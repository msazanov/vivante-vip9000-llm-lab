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
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
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
DEFAULT_WINDOWS_US = (100_000, 250_000, 500_000, 1_000_000)
DEFAULT_REPETITIONS = 1
PINNED_HELPER_SHA256 = "5603b2f4ca8a0f917fdbe5c6716e7182561be1459341e99ab91bd573d7bf27d1"


class NSIError(RuntimeError):
    """Ошибка интерфейса NSI; продолжение эксперимента небезопасно."""


class TimerRestoreFailure(NSIError):
    """Старое значение pmu_timer не удалось записать и подтвердить."""


class ThermalAbort(NSIError):
    """Сработал тепловой предохранитель калибровки."""


def validate_thermal_limit(value: float) -> float:
    """Reject NaN/Inf and limits outside the fail-closed target range."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > THERMAL_LIMIT_C:
        raise NSIError("thermal limit должен быть конечным числом в диапазоне (0, 85]")
    return value


def resolve_pinned_helper(script_path: Path, expected_sha256: str = PINNED_HELPER_SHA256) -> Path:
    """Resolve only the sibling helper whose digest is embedded in this script."""
    script = script_path.resolve(strict=True)
    helper = script.with_name("nsi_sequential_read")
    try:
        helper_real = helper.resolve(strict=True)
        info = helper_real.stat()
    except OSError as exc:
        raise NSIError(f"pinned helper недоступен: {exc}") from exc
    if helper_real.parent != script.parent or helper_real.name != "nsi_sequential_read":
        raise NSIError("pinned helper вышел за каталог скрипта")
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022:
        raise NSIError("pinned helper должен быть regular и не writable для group/other")
    parent_info = helper_real.parent.stat()
    if os.geteuid() == 0 and (
        info.st_uid != 0 or parent_info.st_uid != 0 or parent_info.st_mode & 0o022
    ):
        raise NSIError("root-run требует root-owned helper и защищённый parent directory")
    digest = hashlib.sha256(helper_real.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise NSIError(f"SHA-256 pinned helper не совпал: {digest}")
    return helper_real


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


def _mountinfo_is_sysfs(path: Path) -> bool:
    """Verify that the resolved attribute parent belongs to a sysfs mount."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise NSIError(f"не удалось проверить sysfs identity: {exc}") from exc
    best_mount = ""
    best_type = ""
    resolved = str(path)
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        after_fields = after.split()
        if len(fields) < 5 or not after_fields:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        if (resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/")) and len(mountpoint) > len(best_mount):
            best_mount = mountpoint
            best_type = after_fields[0]
    return best_type == "sysfs"


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

    def __init__(self, root: Path | str, *, allow_test_filesystem: bool = False) -> None:
        self.root = Path(root)
        # Принимаем только каталог hwmon0, а не произвольный файл. Это также
        # не даёт случайно передать сюда путь с именем port_*.
        if self.root.name.startswith("port_") or self.root.name == "pmu_timer":
            raise NSIError("NSI root должен быть каталогом hwmon, а не control-файлом")
        self.timer_path = self.root / "pmu_timer"
        if self.timer_path.name != "pmu_timer":
            raise NSIError("внутренняя ошибка: разрешён только pmu_timer")
        if not allow_test_filesystem:
            resolved_parent = self.root.resolve(strict=True)
            if not str(resolved_parent).startswith("/sys/") or not _mountinfo_is_sysfs(resolved_parent):
                raise NSIError("pmu_timer должен находиться на sysfs mount; fake разрешён только unit-тестам")
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._timer_fd = os.open(self.timer_path, flags)
            info = os.fstat(self._timer_fd)
        except OSError as exc:
            raise NSIError(f"безопасное открытие pmu_timer отклонено: {exc}") from exc
        if not stat.S_ISREG(info.st_mode):
            os.close(self._timer_fd)
            raise NSIError("pmu_timer не является regular sysfs attribute")

    def __del__(self) -> None:
        fd = getattr(self, "_timer_fd", None)
        if isinstance(fd, int):
            try:
                os.close(fd)
            except OSError:
                pass
            self._timer_fd = None

    def _read_timer_raw(self) -> str:
        try:
            os.lseek(self._timer_fd, 0, os.SEEK_SET)
            return os.read(self._timer_fd, 128).decode("ascii", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise NSIError(f"не удалось безопасно прочитать {self.timer_path}: {exc}") from exc

    def read_timer(self) -> int:
        try:
            text = self._read_timer_raw()
        except NSIError:
            raise
        return _parse_timer(text)

    def _write_timer_raw(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise NSIError(f"недопустимое значение pmu_timer: {value!r}")
        try:
            # sysfs принимает обычную десятичную строку; fsync для sysfs не
            # требуется и на некоторых hwmon mount невозможен.
            os.lseek(self._timer_fd, 0, os.SEEK_SET)
            payload = f"{value}\n".encode("ascii")
            written = os.write(self._timer_fd, payload)
            if written != len(payload):
                raise OSError(f"short write {written}/{len(payload)}")
            # Fake regular files need truncation; sysfs rejects/ignores it.
            if not str(self.timer_path).startswith("/sys/"):
                os.ftruncate(self._timer_fd, written)
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


def run_under_watchdog(nsi: SysfsNSI, child: Callable[[], int]) -> dict[str, Any]:
    """Run calibration in a child while the parent owns final restoration.

    SIGINT/SIGTERM/SIGHUP are blocked across the save/fork transition. The
    watchdog parent then forwards them to the child and always restores the
    saved timer after waitpid. SIGKILL cannot be handled in-process; killing
    the *child* is covered because this independent parent remains alive.
    """
    watched = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    original = nsi.read_timer()
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    pid = os.fork()
    if pid == 0:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            for signum in watched:
                signal.signal(signum, signal.SIG_DFL)
            code = int(child())
        except BaseException:
            traceback.print_exc()
            code = 3
        os._exit(max(0, min(255, code)))

    forwarded: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        forwarded.append(signum)
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

    try:
        for signum in watched:
            previous_handlers[signum] = signal.signal(signum, forward)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        _, wait_status = os.waitpid(pid, 0)
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, watched)
        restore_error: BaseException | None = None
        try:
            nsi.write_timer(original)
        except BaseException as exc:
            restore_error = exc
        restore_verified = restore_error is None and nsi.read_timer() == original
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if not restore_verified:
            raise TimerRestoreFailure(
                f"watchdog не восстановил pmu_timer={original}: {restore_error}"
            )
    child_signal = os.WTERMSIG(wait_status) if os.WIFSIGNALED(wait_status) else None
    exit_code = os.waitstatus_to_exitcode(wait_status)
    return {
        "exit_code": exit_code,
        "signal": child_signal,
        "forwarded_signals": forwarded,
        "restore_verified": restore_verified,
        "original_pmu_timer": original,
    }


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
        float(point["bytes"]) / (float(point["active_us"]) / 1_000_000.0)
        for point in points
        if float(point["active_us"]) > 0.0
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
        "active_rate_fit": rate_fit,
        # Backward-compatible alias; the independent variable is now
        # explicitly active duration, never the programmed PMU window.
        "rate_fit": rate_fit,
    }


def _scientific_alignment(helper: object) -> Mapping[str, Any] | None:
    if not isinstance(helper, Mapping):
        return None
    planned = helper.get("planned_bytes")
    actual = helper.get("bytes_read")
    alignment = helper.get("window_alignment")
    if (
        not isinstance(planned, int)
        or isinstance(planned, bool)
        or planned <= 0
        or actual != planned
        or not isinstance(alignment, Mapping)
    ):
        return None
    active = alignment.get("active_us")
    tail = alignment.get("idle_tail_us")
    ratio = alignment.get("elapsed_programmed_ratio")
    if (
        alignment.get("workload_fully_contained") is not True
        or not _finite_positive(active)
        or not _finite_nonnegative(tail)
        or not _finite_positive(ratio)
        or not 1.0 <= float(ratio) <= 1.01
    ):
        return None
    return alignment


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


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


def raw_thermal_maxima(path: Path) -> dict[str, float]:
    """Compute maxima from every persisted raw event, not selected endpoints."""
    maxima: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            state = record.get("telemetry")
            thermal = state.get("thermal_c") if isinstance(state, Mapping) else None
            if not isinstance(thermal, Mapping):
                continue
            for name, value in thermal.items():
                numeric = float(value)
                maxima[str(name)] = max(maxima.get(str(name), -math.inf), numeric)
    return maxima


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


def plan_exact_read_bytes(
    ready: Mapping[str, Any],
    *,
    window_us: int,
) -> dict[str, int | float]:
    """Choose one exact, chunk-aligned read volume below half the PMU window.

    The helper calibration happens before READY, hence outside the PMU window.
    Limiting the predicted active time to half the programmed interval leaves a
    deliberately large scheduling margin and an observable idle tail.
    """

    if not isinstance(window_us, int) or isinstance(window_us, bool) or window_us <= 0:
        raise NSIError("window_us должен быть положительным integer")
    numeric: dict[str, float] = {}
    for name in (
        "buffer_bytes",
        "calibration_bytes",
        "calibration_elapsed_ns",
        "deadline_chunk_bytes",
        "deadline_guard_ns",
    ):
        value = ready.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise NSIError(f"helper READY: {name} должен быть конечным положительным числом")
        numeric[name] = float(value)
    integer_fields = ("buffer_bytes", "calibration_bytes", "deadline_chunk_bytes", "deadline_guard_ns")
    if any(not float(numeric[name]).is_integer() for name in integer_fields):
        raise NSIError("helper READY: byte/guard поля должны быть integer")
    chunk_bytes = int(numeric["deadline_chunk_bytes"])
    buffer_bytes = int(numeric["buffer_bytes"])
    if chunk_bytes % 8 != 0 or buffer_bytes % chunk_bytes != 0:
        raise NSIError("helper READY: chunk/buffer alignment не поддерживается")
    window_ns = window_us * 1000
    guard_ns = int(numeric["deadline_guard_ns"])
    planned_active_budget_ns = min(window_ns // 2, window_ns - 2 * guard_ns)
    if planned_active_budget_ns <= 0:
        raise NSIError("PMU window слишком короткое для deadline guard")
    bytes_per_ns = numeric["calibration_bytes"] / numeric["calibration_elapsed_ns"]
    capacity = int(bytes_per_ns * planned_active_budget_ns)
    planned_bytes = min(buffer_bytes, capacity)
    planned_bytes -= planned_bytes % chunk_bytes
    if planned_bytes < chunk_bytes:
        raise NSIError("калибровка не позволяет безопасно запланировать один bounded chunk")
    return {
        "planned_bytes": planned_bytes,
        "planned_active_budget_ns": planned_active_budget_ns,
        "calibrated_bytes_per_ns": bytes_per_ns,
        "safety_fraction": 0.5,
        "chunk_bytes": chunk_bytes,
        "deadline_guard_ns": guard_ns,
    }


def validate_window_alignment(alignment: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless exact bytes and workload lie inside one PMU window."""

    required = (
        "programmed_window_us",
        "pmu_arm_before_ns",
        "pmu_arm_after_ns",
        "pmu_deadline_earliest_ns",
        "pmu_deadline_latest_ns",
        "pmu_read_start_ns",
        "pmu_read_end_ns",
        "workload_start_ns",
        "workload_end_ns",
        "planned_bytes",
        "bytes_read",
    )
    values: dict[str, int] = {}
    for name in required:
        value = alignment.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
            or int(value) < 0
        ):
            raise NSIError(f"window alignment: {name} должен быть конечным integer >= 0")
        values[name] = int(value)
    window_us = values["programmed_window_us"]
    if window_us <= 0 or values["planned_bytes"] <= 0:
        raise NSIError("window alignment: окно и planned_bytes должны быть положительными")
    if values["bytes_read"] != values["planned_bytes"]:
        raise NSIError("window alignment: helper bytes не совпадают с exact plan")
    window_ns = window_us * 1000
    if values["pmu_deadline_earliest_ns"] != values["pmu_arm_before_ns"] + window_ns:
        raise NSIError("window alignment: неверный earliest PMU deadline")
    if values["pmu_deadline_latest_ns"] != values["pmu_arm_after_ns"] + window_ns:
        raise NSIError("window alignment: неверный latest PMU deadline")
    ordered = (
        values["pmu_arm_before_ns"],
        values["pmu_arm_after_ns"],
        values["workload_start_ns"],
        values["workload_end_ns"],
        values["pmu_deadline_earliest_ns"],
        values["pmu_deadline_latest_ns"],
        values["pmu_read_start_ns"],
        values["pmu_read_end_ns"],
    )
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        raise NSIError("window alignment: workload/PMU timestamps не вложены")
    elapsed_ratio = (
        values["pmu_read_start_ns"] - values["pmu_arm_after_ns"]
    ) / window_ns
    if elapsed_ratio < 1.0 or elapsed_ratio > 1.01:
        raise NSIError(
            f"window alignment: elapsed/programmed={elapsed_ratio:.9f} вне [1.0, 1.01]"
        )
    active_us = (
        values["workload_end_ns"] - values["workload_start_ns"]
    ) / 1000.0
    if active_us <= 0.0:
        raise NSIError("window alignment: active_us должен быть > 0")
    idle_tail_us = (
        values["pmu_deadline_earliest_ns"] - values["workload_end_ns"]
    ) / 1000.0
    checked = dict(alignment)
    checked.update({
        "workload_fully_contained": True,
        "elapsed_programmed_ratio": elapsed_ratio,
        "active_us": active_us,
        "idle_tail_us": idle_tail_us,
        "pmu_arm_latency_us": (
            values["pmu_arm_after_ns"] - values["pmu_arm_before_ns"]
        ) / 1000.0,
    })
    return checked


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
            window_us=fields.get("window_us"),
            pmu=nsi.snapshot(),
            telemetry=last_state,
        )
        time.sleep(min(0.050, max(0.001, deadline - time.monotonic())))


def _monotonic_raw_ns() -> int:
    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return time.clock_gettime_ns(clock_id)


def _wait_briefly_until(deadline_ns: int) -> None:
    """Wait close to an absolute RAW monotonic deadline without 1% overshoot."""
    remaining_ns = deadline_ns - _monotonic_raw_ns()
    if remaining_ns <= 0:
        return
    if remaining_ns > 2_000_000:
        time.sleep(min(0.005, (remaining_ns - 1_000_000) / 1_000_000_000.0))
    else:
        time.sleep(min(0.0001, remaining_ns / 1_000_000_000.0))


def _validate_elapsed_window(alignment: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "programmed_window_us",
        "pmu_arm_before_ns",
        "pmu_arm_after_ns",
        "pmu_deadline_earliest_ns",
        "pmu_deadline_latest_ns",
        "pmu_read_start_ns",
        "pmu_read_end_ns",
    )
    values: dict[str, int] = {}
    for name in required:
        value = alignment.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
            or int(value) < 0
        ):
            raise NSIError(f"PMU elapsed gate: {name} должен быть конечным integer >= 0")
        values[name] = int(value)
    window_us = values["programmed_window_us"]
    if window_us <= 0:
        raise NSIError("PMU elapsed gate: programmed window должен быть > 0")
    window_ns = window_us * 1000
    if (
        values["pmu_deadline_earliest_ns"] != values["pmu_arm_before_ns"] + window_ns
        or values["pmu_deadline_latest_ns"] != values["pmu_arm_after_ns"] + window_ns
    ):
        raise NSIError("PMU elapsed gate: deadlines не соответствуют arm bounds")
    ordered = (
        values["pmu_arm_before_ns"],
        values["pmu_arm_after_ns"],
        values["pmu_deadline_earliest_ns"],
        values["pmu_deadline_latest_ns"],
        values["pmu_read_start_ns"],
        values["pmu_read_end_ns"],
    )
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        raise NSIError("PMU elapsed gate: timestamps не упорядочены")
    ratio = (values["pmu_read_start_ns"] - values["pmu_arm_after_ns"]) / window_ns
    if ratio < 1.0 or ratio > 1.01:
        raise NSIError(f"PMU elapsed gate: elapsed/programmed={ratio:.9f} вне [1.0, 1.01]")
    checked = dict(alignment)
    checked["elapsed_programmed_ratio"] = ratio
    checked["pmu_arm_latency_us"] = (
        values["pmu_arm_after_ns"] - values["pmu_arm_before_ns"]
    ) / 1000.0
    return checked


def _run_idle_window(
    nsi: SysfsNSI,
    programmed_window_us: int,
    sysfs_root: Path,
    limit_c: float,
    trace: JsonlTrace,
    **fields: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure an idle PMU window with the same <=1.01 elapsed gate."""
    window_us = int(programmed_window_us)
    if "window_us" in fields and int(fields["window_us"]) != window_us:
        raise NSIError("idle window_us argument/fields не совпадают")
    samples: list[dict[str, Any]] = []
    pmu_arm_before_ns = _monotonic_raw_ns()
    with timer_window(nsi, window_us):
        pmu_arm_after_ns = _monotonic_raw_ns()
        earliest_ns = pmu_arm_before_ns + window_us * 1000
        latest_ns = pmu_arm_after_ns + window_us * 1000
        next_thermal_ns = pmu_arm_after_ns
        while _monotonic_raw_ns() < latest_ns:
            now_ns = _monotonic_raw_ns()
            if now_ns >= next_thermal_ns:
                samples.append({
                    "phase": "idle",
                    "monotonic_raw_ns": now_ns,
                    "telemetry": _check_thermal(sysfs_root, limit_c),
                })
                next_thermal_ns = now_ns + 50_000_000
            _wait_briefly_until(latest_ns)
        pmu_read_start_ns = _monotonic_raw_ns()
        snapshot = nsi.snapshot()
        pmu_read_end_ns = _monotonic_raw_ns()
        final_telemetry = telemetry(sysfs_root)
    alignment = _validate_elapsed_window({
        "programmed_window_us": window_us,
        "pmu_arm_before_ns": pmu_arm_before_ns,
        "pmu_arm_after_ns": pmu_arm_after_ns,
        "pmu_deadline_earliest_ns": earliest_ns,
        "pmu_deadline_latest_ns": latest_ns,
        "pmu_read_start_ns": pmu_read_start_ns,
        "pmu_read_end_ns": pmu_read_end_ns,
    })
    for sample in samples:
        trace.record(
            "sample",
            mode="idle",
            phase="idle",
            buffer_mib=fields.get("buffer_mib"),
            window_us=window_us,
            monotonic_raw_ns=sample["monotonic_raw_ns"],
            telemetry=sample["telemetry"],
        )
    trace.record(
        "idle_result",
        mode="idle",
        buffer_mib=fields.get("buffer_mib"),
        window_us=window_us,
        window_alignment=alignment,
        telemetry=final_telemetry,
    )
    return samples, snapshot, alignment, final_telemetry


def _run_reader(
    helper: Path,
    size_mib: int,
    programmed_window_us: int,
    nsi: SysfsNSI,
    sysfs_root: Path,
    limit_c: float,
    trace: JsonlTrace,
    **fields: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    command = [str(helper), str(size_mib)]
    start_read_fd, start_write_fd = os.pipe()
    environment = dict(os.environ)
    environment["E049_START_FD"] = str(start_read_fd)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            pass_fds=(start_read_fd,),
        )
    except OSError as exc:
        os.close(start_read_fd)
        os.close(start_write_fd)
        raise NSIError(f"не удалось запустить helper {helper}: {exc}") from exc
    os.close(start_read_fd)
    assert process.stdout is not None
    ready = process.stdout.readline()
    if not ready.startswith("READY "):
        process.kill()
        stdout, stderr = process.communicate()
        os.close(start_write_fd)
        raise NSIError(f"helper не подтвердил synchronized READY: {ready!r} {stdout!r} {stderr!r}")
    try:
        ready_payload = json.loads(ready.removeprefix("READY "))
    except json.JSONDecodeError as exc:
        process.kill()
        stdout, stderr = process.communicate()
        os.close(start_write_fd)
        raise NSIError(f"helper READY не JSON: {ready!r} {stdout!r} {stderr!r}") from exc
    if not isinstance(ready_payload, dict):
        process.kill()
        process.communicate()
        os.close(start_write_fd)
        raise NSIError("helper READY должен быть JSON-объектом")
    window_us = int(programmed_window_us)
    if "window_us" in fields and int(fields["window_us"]) != window_us:
        raise NSIError("window_us argument/fields не совпадают")
    plan = plan_exact_read_bytes(ready_payload, window_us=window_us)
    trace.record(
        "helper_ready",
        mode=fields.get("mode", "read"),
        buffer_mib=size_mib,
        window_us=window_us,
        helper_ready=ready_payload,
        read_plan=plan,
    )
    original = nsi.read_timer()
    blocked = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    pmu_arm_before_ns = 0
    pmu_arm_after_ns = 0
    pmu_deadline_earliest_ns = 0
    pmu_deadline_latest_ns = 0
    try:
        # Counter window begins before the first workload read. The helper was
        # already allocated/touched/calibrated and is blocked on this gate.
        pmu_arm_before_ns = _monotonic_raw_ns()
        nsi.write_timer(window_us)
        pmu_arm_after_ns = _monotonic_raw_ns()
        pmu_deadline_earliest_ns = pmu_arm_before_ns + window_us * 1000
        pmu_deadline_latest_ns = pmu_arm_after_ns + window_us * 1000
        os.write(
            start_write_fd,
            f"{plan['planned_bytes']} {pmu_deadline_earliest_ns}\n".encode("ascii"),
        )
    finally:
        os.close(start_write_fd)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    samples: list[dict[str, Any]] = []
    thermal_abort: ThermalAbort | None = None
    read_pmu: dict[str, Any]
    stdout = ready
    stderr = ""
    result: dict[str, Any] | None = None
    try:
        next_thermal_ns = _monotonic_raw_ns()
        while process.poll() is None:
            now_ns = _monotonic_raw_ns()
            if now_ns >= pmu_deadline_earliest_ns:
                process.send_signal(signal.SIGTERM)
                break
            if now_ns >= next_thermal_ns:
                try:
                    state = _check_thermal(sysfs_root, limit_c)
                except ThermalAbort as exc:
                    thermal_abort = exc
                    process.send_signal(signal.SIGTERM)
                    break
                samples.append({
                    "phase": "active",
                    "monotonic_raw_ns": now_ns,
                    "telemetry": state,
                })
                next_thermal_ns = now_ns + 50_000_000
            _wait_briefly_until(min(pmu_deadline_earliest_ns, now_ns + 1_000_000))
        stdout_tail, stderr = process.communicate(timeout=2)
        stdout = ready + stdout_tail
        if process.returncode == 0:
            result = _json_from_stdout(stdout)
        while _monotonic_raw_ns() < pmu_deadline_latest_ns:
            now_ns = _monotonic_raw_ns()
            if now_ns >= next_thermal_ns:
                state = _check_thermal(sysfs_root, limit_c)
                samples.append({
                    "phase": "idle_tail",
                    "monotonic_raw_ns": now_ns,
                    "telemetry": state,
                })
                next_thermal_ns = now_ns + 50_000_000
            _wait_briefly_until(pmu_deadline_latest_ns)
        pmu_read_start_ns = _monotonic_raw_ns()
        read_pmu = nsi.snapshot()
        pmu_read_end_ns = _monotonic_raw_ns()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        nsi.write_timer(original)
        if nsi.read_timer() != original:
            raise TimerRestoreFailure(f"read-window не восстановил pmu_timer={original}")
    for sample in samples:
        trace.record(
            "sample",
            mode=fields.get("mode", "read"),
            phase=sample["phase"],
            buffer_mib=size_mib,
            window_us=window_us,
            monotonic_raw_ns=sample["monotonic_raw_ns"],
            telemetry=sample["telemetry"],
        )
    if thermal_abort is not None:
        raise thermal_abort
    if process.returncode != 0:
        raise NSIError(
            f"helper завершился с кодом {process.returncode}; stdout={stdout!r}; stderr={stderr!r}"
        )
    if result is None or result.get("status") != "done":
        raise NSIError(f"helper не подтвердил DONE: {result!r}")
    if result.get("deadline_ns") != pmu_deadline_earliest_ns:
        raise NSIError("helper deadline не совпадает с conservative PMU deadline")
    alignment = validate_window_alignment({
        "programmed_window_us": window_us,
        "pmu_arm_before_ns": pmu_arm_before_ns,
        "pmu_arm_after_ns": pmu_arm_after_ns,
        "pmu_deadline_earliest_ns": pmu_deadline_earliest_ns,
        "pmu_deadline_latest_ns": pmu_deadline_latest_ns,
        "pmu_read_start_ns": pmu_read_start_ns,
        "pmu_read_end_ns": pmu_read_end_ns,
        "workload_start_ns": result.get("workload_start_ns"),
        "workload_end_ns": result.get("workload_end_ns"),
        "planned_bytes": plan["planned_bytes"],
        "bytes_read": result.get("bytes_read"),
    })
    trace.record(
        "helper_result",
        mode=fields.get("mode", "read"),
        buffer_mib=size_mib,
        window_us=fields.get("window_us"),
        command=command,
        helper_result=result,
        helper_stdout=stdout,
        helper_stderr=stderr,
        window_alignment=alignment,
    )
    result["window_us"] = window_us
    result["read_plan"] = plan
    result["window_alignment"] = alignment
    return result, samples, read_pmu


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
    original_timer_raw: int,
    thermal_limit_c: float,
    status: str,
) -> dict[str, Any]:
    """Собрать fit/R²/CV и классификацию единиц для каждого PMU-сигнала."""
    index = _master_index(master_names)
    signals = list(SysfsNSI.COUNTER_FILES)
    read_candidates = [point for point in points if point.get("mode") == "read"]
    accepted_read_points = [
        point for point in read_candidates
        if _scientific_alignment(point.get("helper_result")) is not None
    ]
    signal_summaries: dict[str, Any] = {}
    for signal_name in signals:
        read_points = []
        idle_values: list[float] = []
        read_values: list[float] = []
        cell_values: dict[str, list[float]] = {}
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
                helper = point.get("helper_result")
                alignment = _scientific_alignment(helper)
                if alignment is None:
                    continue
                assert isinstance(helper, Mapping)
                read_values.append(value)
                cell_key = f"{point.get('buffer_mib')}MiB@{point.get('window_us')}us"
                cell_values.setdefault(cell_key, []).append(value)
                bytes_read = float(helper["bytes_read"])
                read_points.append({
                    "bytes": bytes_read,
                    "active_us": float(alignment["active_us"]),
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
        unit["cell_cv"] = {
            key: coefficient_of_variation(values)
            for key, values in sorted(cell_values.items())
        }
        signal_summaries[signal_name] = unit
    thermal_maxima: dict[str, float] = {}
    for point in points:
        state = point.get("telemetry")
        thermal = state.get("thermal_c") if isinstance(state, Mapping) else None
        if isinstance(thermal, Mapping):
            for name, value in thermal.items():
                numeric = float(value)
                thermal_maxima[str(name)] = max(thermal_maxima.get(str(name), -math.inf), numeric)
    return {
        "schema": SCHEMA,
        "status": status,
        "original_pmu_timer_raw": original_timer_raw,
        "thermal_limit_c": thermal_limit_c,
        "master_names": list(master_names),
        "selected_channel_index": index,
        "selected_channel_name": master_names[index] if master_names else None,
        "selected_channel_role": "reported aggregate channel; not proven sum",
        "point_count": len(points),
        "alignment_gate": {
            "accepted_points": len(accepted_read_points),
            "rejected_points": len(read_candidates) - len(accepted_read_points),
            "requirements": [
                "bytes_read == planned_bytes",
                "workload_fully_contained == true",
                "1.0 <= elapsed_programmed_ratio <= 1.01",
                "finite active_us > 0",
                "finite idle_tail_us >= 0",
            ],
        },
        "thermal_max_c": thermal_maxima,
        "signals": signal_summaries,
        "interpretation_ru": {
            "pmu_timer": "kernel computes clk_hz/1_000_000*value: sysfs value is microseconds",
            "bandwidth_data_unit": "числа уже масштабированы data-unit; деление на timer не подтверждено",
            "classification": "volume fit uses exact contained bytes; rate fit uses exact bytes / active_us",
            "idle_ratio": "медиана idle-сигнала / медиана read-сигнала; это не физическая единица",
        },
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# E049 — калибровка sunxi-nsi (A733)",
        "",
        f"Статус: **{summary.get('status', 'unknown')}**; исходный `pmu_timer`: "
        f"`{summary.get('original_pmu_timer_raw')}` raw; thermal abort: "
        f"`{summary.get('thermal_limit_c')} °C`.",
        "",
        "Классификация `rate`/`volume` — только fit-гипотеза. Она не заменяет "
        "документирование аппаратных единиц; сырые значения сохранены в `raw.jsonl`.",
        "",
        "| Сигнал (reported aggregate channel) | Гипотеза | Уверенность | R² volume | R² active-rate | idle ratio | CV read |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (summary.get("signals") or {}).items():
        if not isinstance(value, Mapping):
            continue
        volume = value.get("volume_fit") or {}
        rate = value.get("active_rate_fit") or {}
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes-mib", default=",".join(map(str, DEFAULT_SIZES_MIB)))
    parser.add_argument("--windows-us", default=",".join(map(str, DEFAULT_WINDOWS_US)))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--thermal-limit-c", type=float, default=THERMAL_LIMIT_C)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    sizes = _parse_csv_ints(args.sizes_mib, "--sizes-mib")
    windows_us = _parse_csv_ints(args.windows_us, "--windows-us")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions должен быть положительным")
    thermal_limit = validate_thermal_limit(args.thermal_limit_c)
    helper = resolve_pinned_helper(Path(__file__))
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
        "helper": str(helper),
        "helper_sha256": PINNED_HELPER_SHA256,
        "sizes_mib": list(sizes),
        "windows_us": list(windows_us),
        "window_unit": "microseconds",
        "window_unit_evidence": "kernel computes clk_hz/1_000_000 * sysfs_value",
        "repetitions": args.repetitions,
        "thermal_limit_c": thermal_limit,
        "original_pmu_timer_raw": original_timer,
        "initial_pmu": initial,
        "write_surface": ["pmu_timer"],
        "forbidden_write_surface": "all files except exact pmu_timer",
    }
    _write_json(output_dir / "manifest.json", manifest)
    trace.record(
        "start",
        original_pmu_timer_raw=original_timer,
        initial_pmu=initial,
        thermal_limit_c=thermal_limit,
        sizes_mib=list(sizes),
        windows_us=list(windows_us),
        repetitions=args.repetitions,
    )
    points: list[dict[str, Any]] = []
    status = "complete"
    try:
        for repetition in range(args.repetitions):
            for size_mib in sizes:
                for window_us in windows_us:
                    fields = {
                        "repetition": repetition,
                        "buffer_mib": size_mib,
                        "window_us": window_us,
                    }
                    _check_thermal(args.sysfs_root, thermal_limit)
                    trace.record("measurement_start", **fields)
                    thermal_before = _check_thermal(args.sysfs_root, thermal_limit)
                    idle_samples, idle_pmu, idle_alignment, idle_telemetry = _run_idle_window(
                        nsi,
                        window_us,
                        args.sysfs_root,
                        thermal_limit,
                        trace,
                        mode="idle",
                        **fields,
                    )
                    idle_point = {
                        **fields,
                        "mode": "idle",
                        "pmu": idle_pmu,
                        "telemetry": idle_telemetry,
                        "thermal_before": thermal_before,
                        "window_alignment": idle_alignment,
                        "sample_count": len(idle_samples),
                        "helper_result": None,
                        "bytes_read": 0,
                    }
                    points.append(idle_point)
                    trace.record("measurement", **idle_point)

                    # Новое окно сбрасывает counters перед известным read-stream.
                    thermal_before_read = _check_thermal(args.sysfs_root, thermal_limit)
                    helper_result, samples, read_pmu = _run_reader(
                        helper,
                        size_mib,
                        window_us,
                        nsi,
                        args.sysfs_root,
                        thermal_limit,
                        trace,
                        mode="read",
                        **fields,
                    )
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
        original_timer_raw=original_timer,
        thermal_limit_c=thermal_limit,
        status=status,
    )
    summary["thermal_max_c"] = raw_thermal_maxima(trace.path)
    summary["thermal_max_source"] = "all telemetry-bearing events in raw.jsonl"
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    trace.record("complete", status=status, point_count=len(points), summary="summary.json")
    return summary


def write_partial_failure(output_dir: Path, exc: BaseException, *, run_id: str) -> None:
    """Persist failure, raw event and explicitly non-final partial summary."""
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
    JsonlTrace(output_dir / "raw.jsonl", run_id).record("failure", **failure)
    _write_json(output_dir / "summary.partial.json", {
        "schema": SCHEMA,
        "status": "failed",
        "partial": True,
        "unit_classification": "not-computed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    })


def _child_main(args: argparse.Namespace) -> int:
    try:
        summary = run_calibration(args)
    except BaseException as exc:
        # Не скрываем failure: если каталог создан, сохраняем отдельный JSON
        # рядом с append-only trace. Это выполняется даже при thermal abort.
        output_dir = getattr(args, "output_dir", None)
        if isinstance(output_dir, Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            run_id = "unknown"
            manifest_path = output_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                run_id = str(manifest.get("run_id", run_id))
            try:
                write_partial_failure(output_dir, exc, run_id=run_id)
            except BaseException:
                pass
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _emit_watchdog_failure(args: argparse.Namespace, result: Mapping[str, Any]) -> None:
    """Emit machine-readable partial artifacts when the child cannot do so."""
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "signal" if result.get("signal") else "child_failure"
    failure = {
        "schema": SCHEMA,
        "status": status,
        "timestamp_utc": _utc_now(),
        "watchdog": dict(result),
        "note_ru": (
            "SIGKILL нельзя обработать внутри убитого процесса; независимый "
            "родитель восстановил и проверил pmu_timer."
        ),
    }
    _write_json(output_dir / "watchdog_failure.json", failure)
    failure_path = output_dir / "failure.json"
    if not failure_path.exists():
        _write_json(failure_path, failure)
    JsonlTrace(output_dir / "raw.jsonl", "watchdog").record("watchdog_failure", **failure)
    partial = {
        "schema": SCHEMA,
        "status": status,
        "partial": True,
        "watchdog": dict(result),
        "unit_classification": "not-computed",
    }
    _write_json(output_dir / "summary.partial.json", partial)


def _emit_setup_failure(args: argparse.Namespace, exc: BaseException) -> Path:
    """Persist failures that happen before the watchdog child can start."""
    requested: Path = args.output_dir
    output_dir = requested
    if requested.exists() and any(requested.iterdir()):
        # Existing results are immutable evidence. Never append or overwrite
        # them even when the user's setup arguments are invalid.
        while True:
            output_dir = requested.with_name(
                f"{requested.name}.setup-failure-{uuid.uuid4().hex[:8]}"
            )
            if not output_dir.exists():
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema": SCHEMA,
        "status": "setup_failure",
        "timestamp_utc": _utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    _write_json(output_dir / "failure.json", failure)
    JsonlTrace(
        output_dir / "raw.jsonl",
        f"e049-setup-{uuid.uuid4().hex[:8]}",
    ).record("setup_failure", **failure)
    _write_json(output_dir / "summary.partial.json", {
        "schema": SCHEMA,
        "status": "setup_failure",
        "partial": True,
        "unit_classification": "not-computed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    })
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_thermal_limit(args.thermal_limit_c)
        nsi = SysfsNSI(args.nsi_root)
        # Validate the immutable helper before the privileged child is forked.
        resolve_pinned_helper(Path(__file__))
        result = run_under_watchdog(nsi, lambda: _child_main(args))
    except BaseException as exc:
        try:
            evidence_dir = _emit_setup_failure(args, exc)
            print(f"setup evidence: {evidence_dir}", file=sys.stderr)
        except BaseException as evidence_exc:
            print(f"setup evidence write failed: {evidence_exc}", file=sys.stderr)
        print(f"watchdog setup/restore failed: {exc}", file=sys.stderr)
        return 4
    if result["exit_code"] != 0:
        _emit_watchdog_failure(args, result)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
