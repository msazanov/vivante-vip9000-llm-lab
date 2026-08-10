"""Strict collection of CPU and resident VIP9000 benchmark evidence.

The renderer and command line interface turn this immutable model into a
deterministic, reviewable SVG without adding runtime dependencies.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence
import xml.etree.ElementTree as ET


class EvidenceError(ValueError):
    """Raised when saved evidence cannot be safely interpreted."""


@dataclass(frozen=True)
class Series:
    panel: str
    cohort_id: str
    run_id: str
    run_label: str
    metric: str
    status: str
    repository_commit: str
    temperature_c: float
    p10: float
    median: float
    p90: float


@dataclass(frozen=True)
class ChartData:
    cpu: Sequence[Series]
    npu: Sequence[Series]
    omissions: Sequence[tuple[str, int]]
    out_of_scope: int


CPU_CONFIGURATION_FIELDS = {
    "backend", "partition", "context", "batch", "ubatch", "threads",
}
CPU_SOFTWARE_FIELDS = {
    "repository_commit", "runtime", "runtime_commit", "compiler",
    "sdk", "driver", "kernel", "command",
}
CPU_WORKLOAD_FIELDS = {
    "tokenizer", "prompt_suite", "prompt_tokens", "generated_tokens",
    "seed", "deterministic", "warmup_iterations",
}
NPU_ASSET_FIELDS = {
    "vpm_run", "network_binary.nb", "input_0.dat",
    "libNBGlinker.so", "libVIPhal.so",
}
NPU_TARGET_FIELDS = {
    "board", "kernel", "device", "module", "driver_abi",
    "driver_software", "cid", "device_count", "logical_core_count",
}
NPU_RESIDENT_CONFIGURATION_FIELDS = {
    "repository_commit", "profiler_guard_affinity", "host_workload_affinity",
    "device_core", "npu_hz", "loops", "warmup_loops", "measured_loops",
    "profiler_interval_ms", "preload", "npd", "bypass_output", "command",
}

_OMISSION_KEYS = ("failed_status", "missing_metrics", "non_resident", "singleton_cpu_cohort")


def _error(location: str, message: str) -> EvidenceError:
    return EvidenceError(f"{location}: {message}")


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(location, "ожидался JSON-объект")
    return value


def _string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise _error(location, "ожидалась непустая строка")
    return value


def _integer(value: object, location: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(location, "ожидалось целое число")
    if positive and value <= 0:
        raise _error(location, "ожидалось положительное целое число")
    if nonnegative and value < 0:
        raise _error(location, "ожидалось неотрицательное целое число")
    return value


def _bool(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _error(location, "ожидалось логическое значение")
    return value


def _temperature(value: object, location: str) -> float:
    """Validate a saved millidegree Celsius peak and expose degrees Celsius."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(location, "ожидалось целое число millidegrees C")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise _error(location, "температура должна быть конечной и неотрицательной")
    return number / 1000.0


def _check_allowed(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise _error(location, f"неизвестное поле (unknown field) {unknown[0]!r}")


def _required(mapping: dict[str, Any], fields: Sequence[str], location: str) -> None:
    for field in fields:
        if field not in mapping:
            raise _error(location, f"отсутствует обязательное поле {field!r}")


def validate_statistics(value: object, location: str) -> tuple[float, float, float]:
    """Validate and return p10, median, p90 from a statistics object."""
    stats = _mapping(value, location)
    _required(stats, ("p10", "median", "p90"), location)
    values: list[float] = []
    for key in ("p10", "median", "p90"):
        raw = stats[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise _error(f"{location}.{key}", "ожидалось конечное число (finite)")
        number = float(raw)
        if not math.isfinite(number):
            raise _error(f"{location}.{key}", "ожидалось конечное число (finite)")
        if number < 0:
            raise _error(f"{location}.{key}", "ожидалось неотрицательное число (non-negative)")
        values.append(number)
    if not (values[0] <= values[1] <= values[2]):
        raise _error(location, "требуется p10 <= median <= p90")
    return values[0], values[1], values[2]


def _canonical(value: object) -> object:
    """Return JSON-compatible values with stable object key ordering."""
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def cohort_id(signature: Sequence[object]) -> str:
    encoded = json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _signature_sort_key(signature: Sequence[object]) -> str:
    return json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def cpu_signature(row: dict[str, object]) -> Sequence[object]:
    model = _mapping(row.get("model"), "model")
    software = _mapping(row.get("software"), "software")
    workload = _mapping(row.get("workload"), "workload")
    configuration = _mapping(row.get("configuration"), "configuration")
    return (
        ("model.id", model.get("id")),
        ("model.sha256", model.get("sha256")),
        ("software.runtime", software.get("runtime")),
        ("software.runtime_commit", software.get("runtime_commit")),
        ("software.compiler", software.get("compiler")),
        ("software.driver", software.get("driver")),
        ("software.kernel", software.get("kernel")),
        ("software.sdk", software.get("sdk")),
        ("workload", _canonical(workload)),
        ("configuration.context", configuration.get("context")),
        ("configuration.batch", configuration.get("batch")),
        ("configuration.ubatch", configuration.get("ubatch")),
    )


def npu_signature(row: dict[str, object]) -> Sequence[object]:
    assets = _mapping(row.get("asset_sha256"), "asset_sha256")
    target = _mapping(row.get("target"), "target")
    configuration = _mapping(row.get("configuration"), "configuration")
    return tuple(
        [(f"asset_sha256.{field}", assets.get(field)) for field in sorted(NPU_ASSET_FIELDS)]
        + [(f"target.{field}", target.get(field)) for field in sorted(NPU_TARGET_FIELDS)]
        + [
            (f"configuration.{field}", configuration.get(field))
            for field in (
                "device_core", "loops", "warmup_loops", "measured_loops",
                "preload", "npd", "bypass_output",
            )
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(str(path), f"некорректный JSON: {exc}") from exc
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _error(str(path), f"невозможно прочитать ledger: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(f"{path}:{number}", f"некорректный JSON: {exc}") from exc
        rows.append(_mapping(value, f"{path}:{number}"))
    return rows


def _quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _cpu_metric(row: dict[str, Any], metric: str, repetitions: int) -> tuple[float, float, float]:
    statistics = _mapping(row.get("performance"), f"{row.get('run_id', '<unknown>')}.performance")
    all_statistics = _mapping(statistics.get("statistics"), f"{row.get('run_id', '<unknown>')}.performance.statistics")
    location = f"{row.get('run_id', '<unknown>')}.performance.statistics.{metric}"
    block = _mapping(all_statistics.get(metric), location)
    p10, median, p90 = validate_statistics(block, location)
    samples_value = block.get("samples")
    if not isinstance(samples_value, list) or len(samples_value) != repetitions:
        raise _error(location, f"число samples должно совпадать с repetitions ({repetitions})")
    samples: list[float] = []
    for index, sample in enumerate(samples_value):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise _error(f"{location}.samples[{index}]", "ожидалось конечное число (finite)")
        sample_number = float(sample)
        if not math.isfinite(sample_number) or sample_number < 0:
            raise _error(f"{location}.samples[{index}]", "ожидалось конечное неотрицательное число (finite non-negative)")
        samples.append(sample_number)
    expected = (_quantile(samples, 0.1), _quantile(samples, 0.5), _quantile(samples, 0.9))
    for key, actual, recomputed in zip(("p10", "median", "p90"), (p10, median, p90), expected):
        if not math.isclose(actual, recomputed, rel_tol=1e-9, abs_tol=1e-12):
            raise _error(location, f"{key} не согласуется с samples (inconsistent)")
    return p10, median, p90


def _validate_cpu_identity(row: dict[str, Any], location: str) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema_version = _integer(row.get("schema_version"), f"{location}.schema_version", positive=True)
    if schema_version != 1:
        raise _error(f"{location}.schema_version", "поддерживается только schema_version 1")
    run_id = _string(row.get("run_id"), f"{location}.run_id")
    status = _string(row.get("status"), f"{location}.status")
    model = _mapping(row.get("model"), f"{location}.model")
    _required(model, ("id", "sha256"), f"{location}.model")
    _string(model["id"], f"{location}.model.id")
    _string(model["sha256"], f"{location}.model.sha256")
    if "quantization" in model:
        _string(model["quantization"], f"{location}.model.quantization")
    if "size_bytes" in model:
        _integer(model["size_bytes"], f"{location}.model.size_bytes", positive=True)
    configuration = _mapping(row.get("configuration"), f"{location}.configuration")
    software = _mapping(row.get("software"), f"{location}.software")
    workload = _mapping(row.get("workload"), f"{location}.workload")
    _check_allowed(configuration, CPU_CONFIGURATION_FIELDS, f"{location}.configuration")
    _check_allowed(software, CPU_SOFTWARE_FIELDS, f"{location}.software")
    _check_allowed(workload, CPU_WORKLOAD_FIELDS, f"{location}.workload")
    _required(configuration, ("backend", "partition", "context", "batch", "ubatch", "threads"), f"{location}.configuration")
    _required(software, tuple(CPU_SOFTWARE_FIELDS), f"{location}.software")
    _required(workload, tuple(CPU_WORKLOAD_FIELDS), f"{location}.workload")
    _string(configuration["backend"], f"{location}.configuration.backend")
    _string(configuration["partition"], f"{location}.configuration.partition")
    for field in CPU_SOFTWARE_FIELDS:
        _string(software[field], f"{location}.software.{field}")
    for field in ("tokenizer", "prompt_suite"):
        _string(workload[field], f"{location}.workload.{field}")
    for field in ("context", "batch", "ubatch", "threads"):
        _integer(configuration[field], f"{location}.configuration.{field}", positive=True)
    _bool(workload["deterministic"], f"{location}.workload.deterministic")
    for field in ("prompt_tokens", "generated_tokens", "seed", "warmup_iterations"):
        _integer(workload[field], f"{location}.workload.{field}", nonnegative=True)
    return run_id, model, configuration, software, workload


def load_cpu_series(ledger: Path) -> tuple[list[Series], Counter[str]]:
    rows = _read_jsonl(ledger)
    omissions: Counter[str] = Counter()
    by_signature: dict[str, tuple[Sequence[object], list[tuple[dict[str, Any], tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]]]] = {}
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        configuration = row.get("configuration")
        if not isinstance(configuration, dict) or "backend" not in configuration:
            continue
        if isinstance(configuration["backend"], str) and configuration["backend"].casefold() != "cpu":
            continue
        location = f"{ledger}:{index}"
        identity = _validate_cpu_identity(row, location)
        run_id, model, config, software, workload = identity
        if run_id in seen:
            raise _error(location, f"дубликат duplicate CPU run_id {run_id!r}")
        seen.add(run_id)
        status = identity[0] and _string(row["status"], f"{location}.status")
        if status.casefold().startswith("failed"):
            omissions["failed_status"] += 1
            continue
        performance = _mapping(row.get("performance"), f"{run_id}.performance")
        repetitions = performance.get("repetitions")
        if repetitions == 0:
            statistics_value = performance.get("statistics")
            if not isinstance(statistics_value, dict) or not any(
                name in statistics_value for name in ("prompt_tps", "decode_tps")
            ):
                omissions["missing_metrics"] += 1
                continue
        _integer(repetitions, f"{run_id}.performance.repetitions", positive=True)
        statistics = _mapping(performance.get("statistics"), f"{run_id}.performance.statistics")
        metric_names = ("prompt_tps", "decode_tps")
        present = [name in statistics for name in metric_names]
        if not all(present):
            if not any(present):
                raise _error(f"{run_id}.performance.statistics", "отсутствуют throughput metrics")
            raise _error(f"{run_id}.performance.statistics", "неполный набор throughput metrics")
        # Validate before adding the row to a cohort.
        _cpu_metric(row, "prompt_tps", repetitions)
        _cpu_metric(row, "decode_tps", repetitions)
        evidence = _mapping(row.get("evidence"), f"{run_id}.evidence")
        temperature_c = _temperature(
            evidence.get("thermal_peak_millidegrees_c"),
            f"{run_id}.evidence.thermal_peak_millidegrees_c",
        )
        signature = cpu_signature(row)
        key = cohort_id(signature)
        if key not in by_signature:
            by_signature[key] = (signature, [])
        by_signature[key][1].append((row, identity, temperature_c))

    output: list[Series] = []
    for signature, rows_for_signature in sorted(
        (value for value in by_signature.values()),
        key=lambda value: _signature_sort_key(value[0]),
    ):
        cohort_rows = sorted(rows_for_signature, key=lambda item: item[0]["run_id"])
        if len(cohort_rows) < 2:
            omissions["singleton_cpu_cohort"] += len(cohort_rows)
            continue
        cid = cohort_id(signature)
        for row, (_, model, configuration, software, _), temperature_c in cohort_rows:
            label = f"{model['id']} / {configuration['partition']}"
            status = row["status"]
            for metric in ("prompt_tps", "decode_tps"):
                p10, median, p90 = _cpu_metric(row, metric, row["performance"]["repetitions"])
                output.append(Series("cpu", cid, row["run_id"], label, metric, status, software["repository_commit"], temperature_c, p10, median, p90))
    return output, omissions


def _npu_metric(row: dict[str, Any], key: str, loops: int) -> tuple[float, float, float]:
    location = f"{row['run_id']}.{key}"
    block = _mapping(row.get(key), location)
    values = validate_statistics(block, location)
    samples = _integer(block.get("samples"), f"{location}.samples", positive=True)
    if samples != loops:
        raise _error(location, f"samples должны совпадать с measured_loops ({loops})")
    return tuple(value / 1000.0 for value in values)  # microseconds -> milliseconds


def _validate_npu_identity(row: dict[str, Any], directory: Path, location: str) -> tuple[str, str]:
    run_id = _string(row.get("run_id"), f"{location}.run_id")
    status = _string(row.get("status"), f"{location}.status")
    return run_id, status


def load_npu_series(results_dir: Path) -> tuple[list[Series], Counter[str], int]:
    output: list[Series] = []
    omissions: Counter[str] = Counter()
    out_of_scope = 0
    seen: set[str] = set()
    groups: dict[str, tuple[Sequence[object], list[dict[str, Any]]]] = {}
    paths = sorted(results_dir.glob("*/summary.json"))
    for path in paths:
        row = _read_json(path)
        if row.get("schema_version") != "vip9000-capability-run/v1":
            out_of_scope += 1
            continue
        directory = path.parent
        run_id, status = _validate_npu_identity(row, directory, str(path))
        if run_id in seen:
            raise _error(str(path), f"дубликат duplicate NPU run_id {run_id!r}")
        seen.add(run_id)
        if run_id != directory.name:
            raise _error(f"{path}.run_id", "run_id должен совпадать с basename result directory")
        if status.casefold().startswith("failed"):
            omissions["failed_status"] += 1
            continue
        raw_configuration = row.get("configuration")
        configuration = raw_configuration if isinstance(raw_configuration, dict) else {}
        has_host = "host_run_us" in row
        has_device = "device_inference_us" in row
        measured_present = "measured_loops" in configuration
        if not has_host and not has_device and not measured_present:
            omissions["non_resident"] += 1
            continue
        assets = _mapping(row.get("asset_sha256"), f"{path}.asset_sha256")
        target = _mapping(row.get("target"), f"{path}.target")
        _check_allowed(assets, NPU_ASSET_FIELDS, f"{path}.asset_sha256")
        _check_allowed(target, NPU_TARGET_FIELDS, f"{path}.target")
        _check_allowed(configuration, NPU_RESIDENT_CONFIGURATION_FIELDS, f"{path}.configuration")
        _required(assets, tuple(NPU_ASSET_FIELDS), f"{path}.asset_sha256")
        _required(target, tuple(NPU_TARGET_FIELDS), f"{path}.target")
        _required(configuration, tuple(NPU_RESIDENT_CONFIGURATION_FIELDS), f"{path}.configuration")
        for field in NPU_ASSET_FIELDS:
            _string(assets[field], f"{path}.asset_sha256.{field}")
        for field in NPU_TARGET_FIELDS:
            if field in ("device_count", "logical_core_count"):
                _integer(target[field], f"{path}.target.{field}", positive=True)
            else:
                _string(target[field], f"{path}.target.{field}")
        for field in ("device_core", "npu_hz", "loops", "warmup_loops", "measured_loops", "profiler_interval_ms"):
            _integer(configuration[field], f"{path}.configuration.{field}", nonnegative=True)
        for field in ("repository_commit", "profiler_guard_affinity", "host_workload_affinity", "command"):
            _string(configuration[field], f"{path}.configuration.{field}")
        loops = _integer(configuration["loops"], f"{path}.configuration.loops", positive=True)
        warmup = _integer(configuration["warmup_loops"], f"{path}.configuration.warmup_loops", nonnegative=True)
        measured = _integer(configuration["measured_loops"], f"{path}.configuration.measured_loops", positive=True)
        if loops != warmup + measured:
            raise _error(f"{path}.configuration.loops", "loops должен равняться warmup_loops + measured_loops")
        for field in ("preload", "npd", "bypass_output"):
            _bool(configuration[field], f"{path}.configuration.{field}")
        _npu_metric(row, "host_run_us", measured)
        _npu_metric(row, "device_inference_us", measured)
        profiling = _mapping(row.get("profiling"), f"{path}.profiling")
        temperature_c = _temperature(
            profiling.get("npu_peak_millidegrees_c"),
            f"{path}.profiling.npu_peak_millidegrees_c",
        )
        signature = npu_signature(row)
        key = cohort_id(signature)
        if key not in groups:
            groups[key] = (signature, [])
        groups[key][1].append((row, temperature_c))
    for signature, rows_for_signature in sorted(
        (value for value in groups.values()),
        key=lambda value: _signature_sort_key(value[0]),
    ):
        cid = cohort_id(signature)
        for row, temperature_c in sorted(rows_for_signature, key=lambda item: item[0]["run_id"]):
            config = row["configuration"]
            affinity = f"{config['host_workload_affinity']} / {config['profiler_guard_affinity']}"
            frequency = config["npu_hz"]
            label = f"{row['run_id']} ({affinity}, {frequency} Hz)"
            for metric, source in (("host_run_ms", "host_run_us"), ("device_inference_ms", "device_inference_us")):
                p10, median, p90 = _npu_metric(row, source, config["measured_loops"])
                output.append(Series("npu", cid, row["run_id"], label, metric, row["status"], config["repository_commit"], temperature_c, p10, median, p90))
    return output, omissions, out_of_scope


def collect_chart_data(ledger: Path, results_dir: Path) -> ChartData:
    cpu, cpu_omissions = load_cpu_series(ledger)
    npu, npu_omissions, out_of_scope = load_npu_series(results_dir)
    counts = Counter()
    counts.update(cpu_omissions)
    counts.update(npu_omissions)
    return ChartData(cpu=tuple(cpu), npu=tuple(npu), omissions=tuple((key, counts[key]) for key in _OMISSION_KEYS), out_of_scope=out_of_scope)


def nice_axis(maximum: float) -> tuple[float, Sequence[float]]:
    """Return a stable 1/2/5/10 axis ceiling and its zero-based ticks."""
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise EvidenceError("ось: maximum должен быть конечным числом")
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum <= 0:
        raise EvidenceError("ось: maximum должен быть положительным")
    raw_step = maximum / 5.0
    # For subnormal maxima the division can underflow before log10 sees it.
    # A positive finite ceiling is still useful for a degenerate but valid
    # evidence set, so preserve the input as the only representable tick.
    if raw_step <= 0.0:
        return maximum, (0.0, maximum)
    exponent = math.floor(math.log10(raw_step))
    base = 10.0 ** exponent
    if base <= 0.0 or not math.isfinite(base):
        return maximum, (0.0, maximum)
    step = next((multiplier * base for multiplier in (1.0, 2.0, 5.0, 10.0) if multiplier * base >= raw_step), 10.0 * base)
    ceiling = math.ceil(maximum / step - 1e-12) * step
    # Decimal rounding to a fixed number of places would erase values below
    # 1e-12 (including valid subnormal evidence).  Only tidy ordinary-scale
    # floats; preserve tiny values exactly.
    if abs(ceiling) >= 1e-12:
        ceiling = float(round(ceiling, 12))
    if ceiling < maximum:
        ceiling = math.nextafter(maximum, math.inf)
    count = int(round(ceiling / step))
    ticks = list(
        float(round(index * step, 12)) if abs(index * step) >= 1e-12 else float(index * step)
        for index in range(count + 1)
    )
    if ticks[-1] < ceiling:
        ticks[-1] = ceiling
    return ceiling, tuple(ticks)


def _svg_text(parent: ET.Element, text: str, x: float, y: float, **attributes: str) -> ET.Element:
    node = ET.SubElement(parent, "text", {"x": _fmt(x), "y": _fmt(y), **attributes})
    node.text = text
    return node


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _status_label(status: str) -> str:
    if status == "qualified":
        return "qualified"
    if status == "unqualified" or status.endswith("-unqualified"):
        return "unqualified*"
    if status.casefold().startswith("failed"):
        return "failed*"
    return status[:18] + ("…" if len(status) > 18 else "") + "*"


def _series_order(series: Series) -> tuple[str, str, int]:
    metric_order = {
        "prompt_tps": 0,
        "decode_tps": 1,
        "host_run_ms": 0,
        "device_inference_ms": 1,
    }
    return series.cohort_id, series.run_id, metric_order.get(series.metric, 99)


def _metric_label(series: Series) -> str:
    if series.metric == "prompt_tps":
        return f"{series.median:.3f} ток/с"
    if series.metric == "decode_tps":
        if series.median <= 0:
            raise EvidenceError(f"{series.run_id}.{series.metric}: median должен быть положительным")
        return f"{series.median:.3f} ток/с · {1.0 / series.median:.3f} с/ток"
    if series.metric in ("host_run_ms", "device_inference_ms"):
        if series.median <= 0:
            raise EvidenceError(f"{series.run_id}.{series.metric}: median должен быть положительным")
        return f"{series.median:.3f} мс · {1000.0 / series.median:.2f} инф/с"
    raise EvidenceError(f"{series.run_id}.{series.metric}: неизвестная метрика")


def _panel_rows(series: Sequence[Series]) -> list[tuple[str, list[Series]]]:
    # ChartData already carries the collector's deterministic full-signature
    # order.  Preserve the first cohort appearance instead of sorting by the
    # shortened hash cohort_id; only runs and metrics within that cohort are
    # normalized for stable rows.
    by_cohort: dict[str, dict[str, list[Series]]] = {}
    cohort_order: list[str] = []
    for item in series:
        if item.cohort_id not in by_cohort:
            by_cohort[item.cohort_id] = defaultdict(list)
            cohort_order.append(item.cohort_id)
        by_cohort[item.cohort_id][item.run_id].append(item)
    rows: list[tuple[str, list[Series]]] = []
    for cohort in cohort_order:
        for run_id in sorted(by_cohort[cohort]):
            rows.append((run_id, sorted(by_cohort[cohort][run_id], key=_series_order)))
    return rows


def render_svg(data: ChartData) -> bytes:
    """Render ChartData as deterministic, self-contained UTF-8 SVG bytes."""
    all_series = tuple(data.cpu) + tuple(data.npu)
    if not data.cpu or not data.npu:
        raise EvidenceError("график требует одновременно CPU и NPU evidence")
    for item in all_series:
        if not math.isfinite(item.p10) or not math.isfinite(item.median) or not math.isfinite(item.p90):
            raise EvidenceError(f"{item.run_id}.{item.metric}: статистика должна быть конечной")
        if item.median < 0:
            raise EvidenceError(f"{item.run_id}.{item.metric}: median должен быть неотрицательным")
        if item.metric != "prompt_tps" and item.median <= 0:
            raise EvidenceError(f"{item.run_id}.{item.metric}: median должен быть положительным")
        if not (item.p10 <= item.median <= item.p90):
            raise EvidenceError(f"{item.run_id}.{item.metric}: требуется p10 <= median <= p90")
    cpu_max, cpu_ticks = nice_axis(max(item.p90 for item in data.cpu))
    npu_max, npu_ticks = nice_axis(max(item.p90 for item in data.npu))

    unique_runs = len({item.run_id for item in all_series})
    height = max(900, 900 + 72 * (unique_runs - 3))
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": "1200",
        "height": str(height),
        "viewBox": f"0 0 1200 {height}",
        "role": "img",
        "data-repository-commit": ",".join(sorted({item.repository_commit for item in all_series})),
    })
    title = ET.SubElement(root, "title")
    title.text = "Обзор производительности CPU и резидентного VIP9000"
    desc = ET.SubElement(root, "desc")
    desc.text = "Медианы и интервалы p10–p90 из сохранённых свидетельств измерений; статусы и температуры показаны явно."
    style = ET.SubElement(root, "style")
    style.text = "text{font-family: sans-serif; fill:#202124} .axis{stroke:#6b7280;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1} .bar{stroke-width:2}"
    ET.SubElement(root, "rect", {"x": "0", "y": "0", "width": "1200", "height": str(height), "fill": "white"})

    # Keep the annotation and value columns outside the plotting region.  The
    # complete run ID remains readable on the left, while values have a fixed
    # width on the right and can never collide with a bar.
    left, right = 430.0, 920.0
    value_left, value_width = 945.0, 235.0
    panel_rows = {"cpu": _panel_rows(data.cpu), "npu": _panel_rows(data.npu)}
    panel_gap = 45.0
    top_margin, bottom_margin = 78.0, 105.0

    def required_panel_height(row_count: int) -> float:
        return 82.0 + 18.0 + 72.0 * max(1, row_count) + 20.0

    cpu_required = required_panel_height(len(panel_rows["cpu"]))
    npu_required = required_panel_height(len(panel_rows["npu"]))
    available_panel_height = height - top_margin - bottom_margin - panel_gap
    extra_panel_height = max(0.0, available_panel_height - cpu_required - npu_required)
    cpu_height = cpu_required + extra_panel_height / 2.0
    npu_height = npu_required + extra_panel_height - extra_panel_height / 2.0
    cpu_top, cpu_bottom = top_margin, top_margin + cpu_height
    npu_top, npu_bottom = cpu_bottom + panel_gap, cpu_bottom + panel_gap + npu_height
    panel_top = (cpu_top, npu_top)
    panel_bottom = (cpu_bottom, npu_bottom)
    panels = (
        ("cpu", data.cpu, "CPU: пропускная способность LLM — больше лучше", "токенов/с", cpu_max, cpu_ticks, "CPU показывает пропускную способность; декод дополнен временем на токен."),
        ("npu", data.npu, "VIP9000: задержка резидентного запуска — меньше лучше", "мс", npu_max, npu_ticks, "Накладные расходы настройки и процесса, а также корректность не представлены этими столбцами."),
    )
    metric_colors = {
        "cpu": {"prompt_tps": "#2563eb", "decode_tps": "#7c3aed"},
        "npu": {"host_run_ms": "#d97706", "device_inference_ms": "#059669"},
    }
    metric_names = {
        "prompt_tps": "промпт",
        "decode_tps": "декод",
        "host_run_ms": "хост",
        "device_inference_ms": "устройство",
    }
    for (panel_name, series, heading, unit, axis_max, ticks, caveat), top, bottom in zip(panels, panel_top, panel_bottom):
        group = ET.SubElement(root, "g", {"data-panel": panel_name})
        _svg_text(group, heading, 32, top, **{"font-size": "22", "font-weight": "bold"})
        _svg_text(group, caveat, 32, top + 25, **{"font-size": "13", "fill": "#4b5563"})
        _svg_text(group, f"Единица оси: {unit}", left, top + 49, **{"font-size": "14", "font-weight": "bold"})
        legend = ET.SubElement(group, "g", {"data-role": "legend"})
        panel_metrics = ("prompt_tps", "decode_tps") if panel_name == "cpu" else ("host_run_ms", "device_inference_ms")
        for index, metric in enumerate(panel_metrics):
            _svg_text(legend, f"● {metric_names[metric]}", 720 + index * 95, top + 49, **{"data-role": "legend-item", "data-legend-metric": metric, "font-size": "12", "fill": metric_colors[panel_name][metric]})
        _svg_text(legend, "qualified — сплошная; остальные — пунктир", 720, top + 65, **{"font-size": "11", "fill": "#4b5563"})
        axis_y = top + 82
        for tick in ticks:
            x = left + (right - left) * tick / axis_max
            ET.SubElement(group, "line", {"class": "grid", "x1": _fmt(x), "x2": _fmt(x), "y1": _fmt(axis_y), "y2": _fmt(bottom)})
            _svg_text(group, f"{tick:g}", x, axis_y - 5, **{"text-anchor": "middle", "font-size": "11", "fill": "#6b7280"})
        ET.SubElement(group, "line", {"class": "axis", "x1": _fmt(left), "x2": _fmt(left), "y1": _fmt(axis_y), "y2": _fmt(bottom)})
        rows = panel_rows[panel_name]
        row_step = 72.0
        previous_cohort = None
        for row_index, (run_id, items) in enumerate(rows):
            row_y = axis_y + 18 + row_index * row_step + row_step / 2
            cohort = items[0].cohort_id
            if cohort != previous_cohort:
                if previous_cohort is not None:
                    ET.SubElement(group, "line", {"data-role": "cohort-separator", "x1": _fmt(25), "x2": _fmt(right), "y1": _fmt(row_y - row_step / 2), "y2": _fmt(row_y - row_step / 2), "stroke": "#9ca3af", "stroke-dasharray": "4 4"})
                _svg_text(group, f"cohort {cohort}", 32, row_y - row_step / 2 - 4, **{"data-role": "cohort-header", "data-cohort-id": cohort, "font-size": "11", "fill": "#6b7280"})
                previous_cohort = cohort
            representative = items[0]
            detail_label = representative.run_label
            if detail_label.startswith(run_id):
                detail_label = detail_label[len(run_id):].strip(" ()—")
            _svg_text(group, run_id, 32, row_y - 28, **{"data-role": "run-id", "data-run-id": run_id, "font-size": "13", "font-weight": "bold", "textLength": "370", "lengthAdjust": "spacingAndGlyphs"})
            _svg_text(group, detail_label, 32, row_y - 14, **{"data-role": "run-label", "data-run-id": run_id, "font-size": "10", "textLength": "370", "lengthAdjust": "spacingAndGlyphs", "fill": "#4b5563"})
            _svg_text(group, f"пик {'CPU' if panel_name == 'cpu' else 'NPU'} {representative.temperature_c:.1f} °C", 32, row_y, **{"font-size": "12", "fill": "#4b5563"})
            for metric_index, item in enumerate(items):
                y = row_y + 10 + metric_index * 18
                scale = lambda value: left + (right - left) * value / axis_max
                x_p10, x_med, x_p90 = scale(item.p10), scale(item.median), scale(item.p90)
                group_attrs = {"data-panel": panel_name, "data-cohort-id": item.cohort_id, "data-run-id": item.run_id, "data-metric": item.metric, "data-status": item.status, "data-repository-commit": item.repository_commit}
                metric_group = ET.SubElement(group, "g", group_attrs)
                color = metric_colors[panel_name][item.metric]
                bar_attrs = {"class": "bar", "data-role": "median-bar", "x": _fmt(left), "y": _fmt(y - 6), "width": _fmt(max(0.0, x_med - left)), "height": "12", "fill": color, "stroke": color}
                if item.status != "qualified":
                    bar_attrs["stroke-dasharray"] = "6 3"
                ET.SubElement(metric_group, "rect", bar_attrs)
                ET.SubElement(metric_group, "line", {"data-role": "whisker", "x1": _fmt(x_p10), "x2": _fmt(x_p90), "y1": _fmt(y), "y2": _fmt(y), "stroke": color})
                for x in (x_p10, x_p90):
                    ET.SubElement(metric_group, "line", {"data-role": "whisker-cap", "x1": _fmt(x), "x2": _fmt(x), "y1": _fmt(y - 7), "y2": _fmt(y + 7), "stroke": color})
                _svg_text(metric_group, _metric_label(item) + f" · {_status_label(item.status)}", value_left, y + 4, **{"data-role": "median-label", "font-size": "10", "textLength": _fmt(value_width), "lengthAdjust": "spacingAndGlyphs"})
        _svg_text(group, "0", left, bottom + 20, **{"font-size": "11", "text-anchor": "middle"})
    footer = ET.SubElement(root, "g", {"data-role": "footer"})
    omission_text = ", ".join(f"{key}={count}" for key, count in data.omissions)
    _svg_text(footer, f"Источники: benchmarks/results/model-runs.jsonl; benchmarks/results/*/summary.json · пропуски: {omission_text} · out_of_scope={data.out_of_scope}", 32, height - 62, **{"font-size": "12", "textLength": "1135", "lengthAdjust": "spacingAndGlyphs"})
    _svg_text(footer, "Статус без qualified означает наблюдение производительности, а не подтверждение качества модели.", 32, height - 39, **{"font-size": "12", "textLength": "800", "lengthAdjust": "spacingAndGlyphs", "fill": "#4b5563"})
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return payload.rstrip(b"\n") + b"\n"


def publish_atomic(output: Path, payload: bytes) -> None:
    """Publish bytes through a sibling temporary file and durable replace."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
        temporary = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Детерминированный SVG обзор benchmark evidence")
    parser.add_argument("--ledger", type=Path, default=Path("benchmarks/results/model-runs.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/charts/benchmark-overview.svg"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="атомарно заменить результат")
    mode.add_argument("--check", action="store_true", help="проверить актуальность без записи")
    args = parser.parse_args(argv)
    try:
        payload = render_svg(collect_chart_data(args.ledger, args.results_dir))
        if args.check:
            try:
                return 0 if args.output.read_bytes() == payload else 1
            except FileNotFoundError:
                return 1
        if args.force:
            publish_atomic(args.output, payload)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(payload)
        return 0
    except (EvidenceError, OSError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
