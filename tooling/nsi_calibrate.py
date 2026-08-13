#!/usr/bin/env python3
"""Безопасная калибровка Allwinner sunxi-nsi на Orange Pi A733.

Скрипт намеренно имеет очень узкий write-surface: из sysfs он записывает
только ``pmu_timer`` и после каждого окна проверяет, что старое числовое
значение восстановлено. Все остальные NSI файлы читаются только как
счётчики. В частности, файлы управления портами не обнаруживаются и не
передаются в код записи.

На target запускать от root (например, через ``sudo -S``), чтобы запись в
``pmu_timer`` была разрешена. На хосте модуль можно импортировать для
unit-тестов с fake-sysfs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


SCHEMA = "a733-sunxi-nsi-calibration/v1"
THERMAL_LIMIT_C = 85.0
DEFAULT_NSI_ROOT = Path(
    "/sys/devices/platform/soc@3000000/2020000.nsi-controller/"
    "nsi-pmu/hwmon0"
)
DEFAULT_SYSFS_ROOT = Path("/sys")
DEFAULT_SIZES_MIB = (32, 64, 128, 256)
DEFAULT_TIMERS_MS = (100, 250, 500, 1000)
DEFAULT_REPETITIONS = 1


class NSIError(RuntimeError):
    """Ошибка интерфейса NSI; продолжение эксперимента небезопасно."""


class TimerRestoreFailure(NSIError):
    """Старое значение pmu_timer не удалось записать и подтвердить."""


class ThermalAbort(NSIError):
    """Сработал тепловой предохранитель калибровки."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _parse_int_tokens(text: str, *, name: str) -> list[int]:
    tokens = text.split()
    if not tokens:
        raise NSIError(f"пустой sysfs-файл {name}")
    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token, 10))
        except ValueError as exc:
            raise NSIError(f"нечисловой токен в {name}: {token!r}") from exc
    return values


def _parse_timer(text: str) -> int:
    values = _parse_int_tokens(text, name="pmu_timer")
    if len(values) != 1 or values[0] < 0:
        raise NSIError(f"ожидалось одно неотрицательное число pmu_timer, получено {text!r}")
    return values[0]


class SysfsNSI:
    """Минимальный read/write адаптер для фактического sunxi-nsi sysfs.

    ``PMU_READ_FILES`` — белый список. Он не включает ни один файл
    управления портами; это проверяемое свойство, а не соглашение вызывающего
    кода.
    """

    PMU_READ_FILES = (
        "available_pmu",
        "pmu_bandwidth",
        "pmu_bandwidth_rd",
        "pmu_bandwidth_wr",
        "pmu_cmd_rd",
        "pmu_cmd_wr",
        "pmu_latency_rd",
        "pmu_latency_wr",
        "pmu_timer",
    )
    COUNTER_FILES = (
        "pmu_bandwidth",
        "pmu_bandwidth_rd",
        "pmu_bandwidth_wr",
        "pmu_cmd_rd",
        "pmu_cmd_wr",
        "pmu_latency_rd",
        "pmu_latency_wr",
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        # Принимаем только каталог hwmon0, а не произвольный файл. Это также
        # не даёт случайно передать сюда путь с именем port_*.
        if self.root.name.startswith("port_") or self.root.name == "pmu_timer":
            raise NSIError("NSI root должен быть каталогом hwmon, а не control-файлом")
        self.timer_path = self.root / "pmu_timer"
        if self.timer_path.name != "pmu_timer":
            raise NSIError("внутренняя ошибка: разрешён только pmu_timer")

    def read_timer(self) -> int:
        try:
            text = self.timer_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NSIError(f"не удалось прочитать {self.timer_path}: {exc}") from exc
        return _parse_timer(text)

    def _write_timer_raw(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise NSIError(f"недопустимое значение pmu_timer: {value!r}")
        try:
            # sysfs принимает обычную десятичную строку; fsync для sysfs не
            # требуется и на некоторых hwmon mount невозможен.
            with self.timer_path.open("w", encoding="utf-8") as handle:
                handle.write(f"{value}\n")
                handle.flush()
        except OSError as exc:
            raise NSIError(f"не удалось записать pmu_timer={value}: {exc}") from exc

    def write_timer(self, value: int) -> None:
        """Записать pmu_timer и немедленно проверить read-back."""
        self._write_timer_raw(value)
        observed = self.read_timer()
        if observed != value:
            raise NSIError(
                f"pmu_timer read-back отличается: ожидалось {value}, получено {observed}"
            )

    def snapshot(self) -> dict[str, list[int] | list[str]]:
        result: dict[str, list[int] | list[str]] = {}
        for name in self.PMU_READ_FILES:
            path = self.root / name
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise NSIError(f"не удалось прочитать {path}: {exc}") from exc
            if name == "available_pmu":
                words = text.split()
                if not words:
                    raise NSIError("пустой available_pmu")
                result[name] = words
            else:
                result[name] = _parse_int_tokens(text, name=name)
        return result


@contextlib.contextmanager
def timer_window(
    nsi: SysfsNSI,
    value: int,
) -> Iterator[int]:
    """Временно установить timer и гарантированно вернуть старое значение.

    Исключение восстановления намеренно заменяет исключение workload: после
    такого события основной цикл обязан остановиться, чтобы не писать новые
    окна поверх неизвестного состояния железа.
    """
    original = nsi.read_timer()
    try:
        nsi.write_timer(value)
        yield original
    finally:
        try:
            nsi.write_timer(original)
        except BaseException as exc:
            raise TimerRestoreFailure(
                f"КРИТИЧЕСКАЯ ОШИБКА: pmu_timer не восстановлен в {original}"
            ) from exc


def fit_line(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, float | int]:
    """Обычная линейная регрессия без numpy, включая R² и RMSE."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise NSIError("для fit_line нужны минимум две пары координат")
    x = [float(value) for value in x_values]
    y = [float(value) for value in y_values]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx <= 0.0:
        raise NSIError("fit_line: все x одинаковы")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [actual - (slope * a + intercept) for a, actual in zip(x, y)]
    sse = sum(value * value for value in residuals)
    sst = sum((value - y_mean) ** 2 for value in y)
    r2 = 1.0 if sst == 0.0 and sse == 0.0 else (0.0 if sst == 0.0 else 1.0 - sse / sst)
    return {
        "n": len(x),
        "slope": slope,
        "intercept": intercept,
        "r2": max(-1.0, min(1.0, r2)),
        "rmse": math.sqrt(sse / len(x)),
    }


def coefficient_of_variation(values: Iterable[float]) -> float | None:
    values_list = [float(value) for value in values]
    if not values_list:
        return None
    mean = statistics.fmean(values_list)
    if abs(mean) <= 1e-15:
        return None
    return statistics.pstdev(values_list) / abs(mean)


def _safe_fit(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    try:
        return fit_line(x_values, y_values)
    except NSIError as exc:
        return {"error": str(exc), "n": len(y_values)}


def classify_unit_hypothesis(points: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Сравнить гипотезы «объём за окно» и «скорость за окно».

    Здесь намеренно нет утверждения о физических единицах: PMU source уже
    показывает scaling data-unit, но не деление на timer. Классификация лишь
    говорит, какая модель лучше объясняет наблюдаемые числа.
    """
    if len(points) < 4:
        return {"classification": "insufficient", "confidence": "low", "n": len(points)}
    bytes_values = [float(point["bytes"]) for point in points]
    values = [float(point["value"]) for point in points]
    rate_values = [
        float(point["bytes"]) / (float(point["timer_ms"]) / 1000.0)
        for point in points
        if float(point["timer_ms"]) > 0.0
    ]
    if len(rate_values) != len(points):
        return {"classification": "insufficient", "confidence": "low", "n": len(points)}
    volume_fit = _safe_fit(bytes_values, values)
    rate_fit = _safe_fit(rate_values, values)
    volume_r2 = float(volume_fit.get("r2", -1.0))
    rate_r2 = float(rate_fit.get("r2", -1.0))
    gap = abs(rate_r2 - volume_r2)
    if rate_r2 >= 0.50 and rate_r2 > volume_r2 + 0.05:
        classification = "rate"
    elif volume_r2 >= 0.50 and volume_r2 > rate_r2 + 0.05:
        classification = "volume"
    else:
        classification = "inconclusive"
    confidence = "high" if max(rate_r2, volume_r2) >= 0.90 and gap >= 0.10 else (
        "medium" if max(rate_r2, volume_r2) >= 0.50 and gap >= 0.05 else "low"
    )
    return {
        "classification": classification,
        "confidence": confidence,
        "n": len(points),
        "r2_gap": gap,
        "volume_fit": volume_fit,
        "rate_fit": rate_fit,
    }


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text, 10)
    except (OSError, ValueError):
        return None


def telemetry(sysfs_root: Path = DEFAULT_SYSFS_ROOT) -> dict[str, Any]:
    """Снять только read-only thermals/frequencies, доступные в Linux sysfs."""
    thermals: dict[str, float] = {}
    thermal_root = sysfs_root / "class" / "thermal"
    for temp_path in sorted(thermal_root.glob("thermal_zone*/temp")):
        raw = _read_int(temp_path)
        if raw is None:
            continue
        type_path = temp_path.parent / "type"
        name = type_path.read_text(encoding="utf-8").strip() if type_path.exists() else temp_path.parent.name
        thermals[name] = raw / 1000.0
    frequencies: dict[str, int] = {}
    cpu_root = sysfs_root / "devices" / "system" / "cpu"
    for freq_path in sorted(cpu_root.glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        raw = _read_int(freq_path)
        if raw is not None:
            frequencies[str(freq_path.relative_to(sysfs_root))] = raw
    devfreq: dict[str, int] = {}
    for freq_path in sorted((sysfs_root / "class" / "devfreq").glob("*/cur_freq")):
        raw = _read_int(freq_path)
        if raw is not None:
            devfreq[str(freq_path.parent.name)] = raw
    max_temp = max(thermals.values()) if thermals else None
    return {
        "thermal_c": thermals,
        "max_thermal_c": max_temp,
        "cpu_cur_freq_hz": frequencies,
        "devfreq_cur_hz": devfreq,
    }


def _check_thermal(sysfs_root: Path, limit_c: float) -> dict[str, Any]:
    state = telemetry(sysfs_root)
    maximum = state.get("max_thermal_c")
    if maximum is not None and float(maximum) >= limit_c:
        raise ThermalAbort(f"температурный abort: {maximum:.3f} °C >= {limit_c:.3f} °C")
    return state


class JsonlTrace:
    """Append-only trace; каждая запись flush/fsync-ится до продолжения."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "timestamp_utc": _utc_now(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NSIError(f"helper вывел не JSON: {line!r}") from exc
            if not isinstance(value, dict):
                raise NSIError("helper JSON должен быть объектом")
            return value
    raise NSIError("helper не вывел результата")


def _sleep_with_thermal(
    seconds: float,
    nsi: SysfsNSI,
    sysfs_root: Path,
    limit_c: float,
    trace: JsonlTrace,
    **fields: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, seconds)
    last_state: dict[str, Any] = {}
    while True:
        last_state = _check_thermal(sysfs_root, limit_c)
        if time.monotonic() >= deadline:
            return last_state
        trace.record(
            "sample",
            mode=fields.get("mode"),
            phase="wait",
            buffer_mib=fields.get("buffer_mib"),
            timer_ms=fields.get("timer_ms"),
            pmu=nsi.snapshot(),
            telemetry=last_state,
        )
        time.sleep(min(0.050, max(0.001, deadline - time.monotonic())))


def _run_reader(
    helper: Path,
    size_mib: int,
    duration_ms: int,
    nsi: SysfsNSI,
    sysfs_root: Path,
    limit_c: float,
    trace: JsonlTrace,
    **fields: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = [str(helper), str(size_mib), str(duration_ms)]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise NSIError(f"не удалось запустить helper {helper}: {exc}") from exc
    samples: list[dict[str, Any]] = []
    thermal_abort: ThermalAbort | None = None
    while process.poll() is None:
        try:
            state = _check_thermal(sysfs_root, limit_c)
        except ThermalAbort as exc:
            thermal_abort = exc
            process.send_signal(signal.SIGTERM)
            break
        sample = trace.record(
            "sample",
            mode=fields.get("mode", "read"),
            phase="read",
            buffer_mib=size_mib,
            timer_ms=fields.get("timer_ms"),
            pmu=nsi.snapshot(),
            telemetry=state,
        )
        samples.append(sample)
        time.sleep(0.050)
    stdout, stderr = process.communicate()
    if thermal_abort is not None:
        raise thermal_abort
    if process.returncode != 0:
        raise NSIError(
            f"helper завершился с кодом {process.returncode}; stdout={stdout!r}; stderr={stderr!r}"
        )
    result = _json_from_stdout(stdout)
    trace.record(
        "helper_result",
        mode=fields.get("mode", "read"),
        buffer_mib=size_mib,
        timer_ms=fields.get("timer_ms"),
        command=command,
        helper_result=result,
        helper_stdout=stdout,
        helper_stderr=stderr,
    )
    return result, samples


def _master_index(names: Sequence[str]) -> int:
    try:
        return names.index("total")
    except ValueError:
        return len(names) - 1


def _counter_value(snapshot: Mapping[str, Any], name: str, index: int) -> float | None:
    value = snapshot.get(name)
    if not isinstance(value, list) or index >= len(value):
        return None
    try:
        return float(value[index])
    except (TypeError, ValueError):
        return None


def summarize(
    points: Sequence[Mapping[str, Any]],
    master_names: Sequence[str],
    *,
    original_timer_ms: int,
    thermal_limit_c: float,
    status: str,
) -> dict[str, Any]:
    """Собрать fit/R²/CV и классификацию единиц для каждого PMU-сигнала."""
    index = _master_index(master_names)
    signals = list(SysfsNSI.COUNTER_FILES)
    signal_summaries: dict[str, Any] = {}
    for signal_name in signals:
        read_points = []
        idle_values: list[float] = []
        read_values: list[float] = []
        for point in points:
            snapshot = point.get("pmu")
            if not isinstance(snapshot, Mapping):
                continue
            value = _counter_value(snapshot, signal_name, index)
            if value is None:
                continue
            mode = point.get("mode")
            if mode == "idle":
                idle_values.append(value)
            elif mode == "read":
                read_values.append(value)
                helper = point.get("helper_result")
                bytes_read = float(helper.get("bytes_read", 0.0)) if isinstance(helper, Mapping) else 0.0
                read_points.append({
                    "bytes": bytes_read,
                    "timer_ms": float(point.get("timer_ms", 0.0)),
                    "value": value,
                })
        unit = classify_unit_hypothesis(read_points)
        unit["idle_ratio"] = (
            statistics.median(idle_values) / statistics.median(read_values)
            if idle_values and read_values and statistics.median(read_values) != 0.0
            else None
        )
        unit["read_cv"] = coefficient_of_variation(read_values)
        unit["idle_cv"] = coefficient_of_variation(idle_values)
        signal_summaries[signal_name] = unit
    return {
        "schema": SCHEMA,
        "status": status,
        "original_pmu_timer_ms": original_timer_ms,
        "thermal_limit_c": thermal_limit_c,
        "master_names": list(master_names),
        "total_master_index": index,
        "point_count": len(points),
        "signals": signal_summaries,
        "interpretation_ru": {
            "pmu_timer": "исходник ядра и read-back target указывают на миллисекунды",
            "bandwidth_data_unit": "числа уже масштабированы data-unit; деление на timer не подтверждено",
            "classification": "rate/volume — статистическая гипотеза по двум наборам размеров и окон",
            "idle_ratio": "медиана idle-сигнала / медиана read-сигнала; это не физическая единица",
        },
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# E049 — калибровка sunxi-nsi (A733)",
        "",
        f"Статус: **{summary.get('status', 'unknown')}**; исходный `pmu_timer`: "
        f"`{summary.get('original_pmu_timer_ms')}` мс; thermal abort: "
        f"`{summary.get('thermal_limit_c')} °C`.",
        "",
        "Классификация `rate`/`volume` — только fit-гипотеза. Она не заменяет "
        "документирование аппаратных единиц; сырые значения сохранены в `raw.jsonl`.",
        "",
        "| Сигнал (master=total) | Гипотеза | Уверенность | R² volume | R² rate | idle ratio | CV read |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (summary.get("signals") or {}).items():
        if not isinstance(value, Mapping):
            continue
        volume = value.get("volume_fit") or {}
        rate = value.get("rate_fit") or {}
        lines.append(
            f"| `{name}` | {value.get('classification', 'n/a')} | "
            f"{value.get('confidence', 'n/a')} | {volume.get('r2', 'n/a')} | "
            f"{rate.get('r2', 'n/a')} | {value.get('idle_ratio', 'n/a')} | "
            f"{value.get('read_cv', 'n/a')} |"
        )
    lines += [
        "",
        "## Ограничения",
        "",
        "- Инструмент записывает только `pmu_timer`; файлы управления портами не "
        "обнаруживаются и не записываются.",
        "- `bandwidth_*` в текущем ядре масштабируются data-unit (для A733 в DT "
        "IA=16, TA/CPU/RA=64); software source не делит значение на timer.",
        "- Для причинности нужны повторения, холодный/горячий cache и независимый "
        "контроль bytes read; это первый bounded calibration gate.",
    ]
    return "\n".join(lines) + "\n"


def _parse_csv_ints(raw: str, option: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        try:
            value = int(token.strip(), 10)
        except ValueError as exc:
            raise SystemExit(f"{option}: нечисловое значение {token!r}") from exc
        if value <= 0:
            raise SystemExit(f"{option}: значения должны быть положительными")
        values.append(value)
    if not values:
        raise SystemExit(f"{option}: пустой список")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsi-root", type=Path, default=DEFAULT_NSI_ROOT)
    parser.add_argument("--sysfs-root", type=Path, default=DEFAULT_SYSFS_ROOT)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes-mib", default=",".join(map(str, DEFAULT_SIZES_MIB)))
    parser.add_argument("--timers-ms", default=",".join(map(str, DEFAULT_TIMERS_MS)))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--thermal-limit-c", type=float, default=THERMAL_LIMIT_C)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    sizes = _parse_csv_ints(args.sizes_mib, "--sizes-mib")
    timers = _parse_csv_ints(args.timers_ms, "--timers-ms")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions должен быть положительным")
    if args.thermal_limit_c > 85.0 + 1e-9:
        # Порог нельзя ослабить относительно safety requirement родителя;
        # ровно 85 °C — требуемая граница abort.
        raise SystemExit("thermal limit не может быть выше 85 °C")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise SystemExit(f"output-dir уже содержит файлы: {output_dir}; используйте новый каталог")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"e049-{_datetime.datetime.now(_datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    trace = JsonlTrace(output_dir / "raw.jsonl", run_id)
    nsi = SysfsNSI(args.nsi_root)
    original_timer = nsi.read_timer()
    initial = nsi.snapshot()
    names_raw = initial.get("available_pmu")
    master_names = [str(value) for value in names_raw] if isinstance(names_raw, list) else []
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_utc": _utc_now(),
        "nsi_root": str(args.nsi_root),
        "helper": str(args.helper),
        "sizes_mib": list(sizes),
        "timers_ms": list(timers),
        "repetitions": args.repetitions,
        "thermal_limit_c": args.thermal_limit_c,
        "original_pmu_timer_ms": original_timer,
        "initial_pmu": initial,
        "write_surface": ["pmu_timer"],
        "forbidden_write_surface": "all files except exact pmu_timer",
    }
    _write_json(output_dir / "manifest.json", manifest)
    trace.record(
        "start",
        original_pmu_timer_ms=original_timer,
        initial_pmu=initial,
        thermal_limit_c=args.thermal_limit_c,
        sizes_mib=list(sizes),
        timers_ms=list(timers),
        repetitions=args.repetitions,
    )
    points: list[dict[str, Any]] = []
    status = "complete"
    try:
        for repetition in range(args.repetitions):
            for size_mib in sizes:
                for timer_ms in timers:
                    fields = {
                        "repetition": repetition,
                        "buffer_mib": size_mib,
                        "timer_ms": timer_ms,
                    }
                    _check_thermal(args.sysfs_root, args.thermal_limit_c)
                    trace.record("measurement_start", **fields)
                    with timer_window(nsi, timer_ms):
                        thermal_before = _check_thermal(args.sysfs_root, args.thermal_limit_c)
                        _sleep_with_thermal(
                            timer_ms / 1000.0 + 0.050,
                            nsi,
                            args.sysfs_root,
                            args.thermal_limit_c,
                            trace,
                            mode="idle",
                            **fields,
                        )
                        idle_pmu = nsi.snapshot()
                        idle_telemetry = telemetry(args.sysfs_root)
                    idle_point = {
                        **fields,
                        "mode": "idle",
                        "pmu": idle_pmu,
                        "telemetry": idle_telemetry,
                        "helper_result": None,
                        "bytes_read": 0,
                    }
                    points.append(idle_point)
                    trace.record("measurement", **idle_point)

                    # Новое окно сбрасывает counters перед известным read-stream.
                    duration_ms = max(150, timer_ms + 50)
                    with timer_window(nsi, timer_ms):
                        thermal_before_read = _check_thermal(args.sysfs_root, args.thermal_limit_c)
                        helper_result, samples = _run_reader(
                            args.helper,
                            size_mib,
                            duration_ms,
                            nsi,
                            args.sysfs_root,
                            args.thermal_limit_c,
                            trace,
                            mode="read",
                            **fields,
                        )
                        read_pmu = nsi.snapshot()
                        read_telemetry = telemetry(args.sysfs_root)
                    read_point = {
                        **fields,
                        "mode": "read",
                        "pmu": read_pmu,
                        "telemetry": read_telemetry,
                        "thermal_before": thermal_before,
                        "thermal_before_read": thermal_before_read,
                        "helper_result": helper_result,
                        "bytes_read": helper_result.get("bytes_read", 0),
                        "sample_count": len(samples),
                    }
                    points.append(read_point)
                    trace.record("measurement", **read_point)
                    trace.record("measurement_end", **fields, restore_verified=nsi.read_timer() == original_timer)
        if nsi.read_timer() != original_timer:
            raise TimerRestoreFailure("финальная проверка pmu_timer не совпала с исходной")
    except TimerRestoreFailure:
        status = "restore_failure"
        raise
    except ThermalAbort:
        status = "thermal_abort"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        manifest["status"] = status
        manifest["finished_utc"] = _utc_now()
        manifest["point_count"] = len(points)
        _write_json(output_dir / "manifest.json", manifest)
    summary = summarize(
        points,
        master_names,
        original_timer_ms=original_timer,
        thermal_limit_c=args.thermal_limit_c,
        status=status,
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    trace.record("complete", status=status, point_count=len(points), summary="summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_calibration(args)
    except BaseException as exc:
        # Не скрываем failure: если каталог создан, сохраняем отдельный JSON
        # рядом с append-only trace. Это выполняется даже при thermal abort.
        output_dir = getattr(args, "output_dir", None)
        if isinstance(output_dir, Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema": SCHEMA,
                "status": "failed",
                "timestamp_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(output_dir / "failure.json", failure)
            try:
                run_id = "unknown"
                manifest_path = output_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    run_id = str(manifest.get("run_id", run_id))
                JsonlTrace(output_dir / "raw.jsonl", run_id).record(
                    "failure",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            except BaseException:
                # Исходная ошибка важнее вторичной ошибки журналирования.
                pass
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
