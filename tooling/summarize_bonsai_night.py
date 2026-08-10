#!/usr/bin/env python3
"""Aggregate fail-closed Bonsai full-model decode profiling bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator


BONSAI_MODEL_PATH = "/home/orangepi/vip9000-lab/models/Bonsai-27B-Q1_0.gguf"
BONSAI_MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
BONSAI_FILE_SIZE_BYTES = 3_803_452_480
BONSAI_BENCH_MODEL_SIZE_BYTES = 3_792_459_776
BONSAI_MODEL_TYPE = "qwen35 27B Q1_0"
BONSAI_MODEL_N_PARAMS = 26_895_998_464
BONSAI_RUNTIME_COMMIT = "38c66ad0241da4f9fcce541cda8edc219086cec5"
BONSAI_RUNTIME_COMMIT_SHORT = "38c66ad"
BONSAI_BUILD_NUMBER = 9594
BONSAI_GOLDEN_SHA256 = "a9d86b7d298be35cf27a156be87f9aeedbae3602b877e74ced0fdc75b41c4a5f"
CHART_SERIES = "Bonsai-27B Q1_0 / A733 heterogeneous night"


class SummaryError(ValueError):
    """A raw run is incomplete, malformed, or fails the Bonsai evidence gate."""


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"schema error at {context}: expected object")
    return value


def _number(
    value: Any, context: str, *, positive: bool = False, non_negative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"schema error at {context}: expected number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise SummaryError(f"schema error at {context}: expected finite positive number")
    if non_negative and result < 0:
        raise SummaryError(f"schema error at {context}: expected finite non-negative number")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"schema error at {context}: expected integer")
    return value


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    except SummaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"malformed JSON: {path}") from error
    return _mapping(value, str(path))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot read {path}") from error
    try:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise SummaryError(f"malformed JSON at {path}:{line_number}: blank line")
            try:
                value = json.loads(line, object_pairs_hook=_object_pairs)
            except SummaryError:
                raise
            except json.JSONDecodeError as error:
                raise SummaryError(f"malformed JSON at {path}:{line_number}") from error
            yield _mapping(value, f"{path}:{line_number}")
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot read {path}") from error
    finally:
        stream.close()


def _values(value: Any, context: str) -> Iterator[float]:
    """Yield scalar numeric values from the common telemetry map/list shapes."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _values(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _values(item, f"{context}[{index}]")
    else:
        yield _number(value, context)


def _zone_values(value: Any, context: str) -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        entries = value.items()
    elif isinstance(value, list):
        entries = enumerate(value)
    else:
        raise SummaryError(f"schema error at {context}: expected object or array")
    for name, entry in entries:
        zone = _mapping(entry, f"{context}.{name}")
        zone_type = zone.get("type")
        if not isinstance(zone_type, str) or not zone_type:
            raise SummaryError(f"schema error at {context}.{name}.type: expected non-empty string")
        raw_temp = zone.get("millidegrees_c", zone.get("temp_millidegrees_c"))
        if raw_temp is None:
            raw_temp = zone.get("temperature_millidegrees_c")
        if raw_temp is None:
            raise SummaryError(f"schema error at {context}.{name}: missing temperature")
        yield zone_type, _number(raw_temp, f"{context}.{name}.millidegrees_c")


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _command(value: Any, context: str) -> list[str]:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    raise SummaryError(f"schema error at {context}: expected command array")


def _command_option(command: list[str], names: tuple[str, ...], context: str) -> str:
    matches: list[str] = []
    for index, argument in enumerate(command):
        if argument in names:
            if index + 1 >= len(command):
                raise SummaryError(f"schema error at {context}: option has no value")
            matches.append(command[index + 1])
        else:
            for name in names:
                prefix = f"{name}="
                if argument.startswith(prefix):
                    matches.append(argument[len(prefix):])
    if len(matches) != 1 or not matches[0]:
        raise SummaryError(f"schema error at {context}: expected exactly one option")
    return matches[0]


def _positive_command_integer(command: list[str], names: tuple[str, ...], context: str) -> int:
    raw = _command_option(command, names, context)
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise SummaryError(f"schema error at {context}: expected integer") from error
    if value <= 0:
        raise SummaryError(f"schema error at {context}: expected positive integer")
    return value


def _decode_workload(command: list[str]) -> tuple[int, int]:
    if not any(Path(argument).name == "llama-bench" for argument in command):
        raise SummaryError("Bonsai workload requires llama-bench")
    model_path = _command_option(command, ("-m", "--model"), "metadata.command.model")
    if model_path != BONSAI_MODEL_PATH:
        raise SummaryError("Bonsai model path does not match the pinned model")
    prompt_tokens = _command_option(command, ("-p", "--n-prompt"), "metadata.command.n_prompt")
    if prompt_tokens != "0":
        raise SummaryError("Bonsai decode workload requires n_prompt=0")
    n_gen = _positive_command_integer(command, ("-n", "--n-gen"), "metadata.command.n_gen")
    repetitions = _positive_command_integer(command, ("-r", "--repetitions"), "metadata.command.repetitions")
    return n_gen, repetitions


def _canonical_values(value: Any, context: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in value):
            raise SummaryError(f"schema error at {context}: expected scalar array")
        return list(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value]
    raise SummaryError(f"schema error at {context}: expected scalar or array")


def _merge_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return sorted(result, key=lambda item: str(item))


def _guard_summary(path: Path) -> tuple[int | None, int | None]:
    rows = list(_jsonl(path))
    if not rows:
        raise SummaryError("thermal guard evidence is empty")
    if len(rows) < 6:
        raise SummaryError(
            "thermal guard lifecycle is incomplete; expected start, inventory, "
            "child_started, sample, child_exit, exit"
        )
    expected_events = ("start", "inventory", "child_started")
    for index, expected_event in enumerate(expected_events):
        if rows[index].get("event") != expected_event or rows[index].get("kind") is not None:
            raise SummaryError(
                f"thermal guard lifecycle requires {expected_event} at line {index + 1}"
            )
    if rows[-2].get("event") != "child_exit" or rows[-2].get("kind") is not None:
        raise SummaryError("thermal guard lifecycle requires child_exit before exit")
    if rows[-1].get("event") != "exit" or rows[-1].get("kind") is not None:
        raise SummaryError("thermal guard lifecycle requires exit as the final event")
    samples = rows[3:-2]
    if not samples:
        raise SummaryError("thermal guard lifecycle requires at least one sample")
    for index, row in enumerate(samples, 4):
        if row.get("kind") != "sample" or row.get("event") is not None:
            raise SummaryError(f"thermal guard lifecycle has unknown row at line {index}")
    previous_monotonic_ns = -1
    for index, row in enumerate(rows, 1):
        monotonic_ns = _integer(
            row.get("monotonic_ns"), f"thermal-guard.jsonl:{index}.monotonic_ns"
        )
        if monotonic_ns <= previous_monotonic_ns:
            raise SummaryError("thermal guard lifecycle timestamps are not strictly increasing")
        previous_monotonic_ns = monotonic_ns
    child_pid = _integer(rows[2].get("pid"), "thermal-guard.jsonl:3.pid")
    if child_pid <= 0:
        raise SummaryError("thermal guard child_started pid must be positive")
    child_returncode = _integer(
        rows[-2].get("returncode"),
        f"thermal-guard.jsonl:{len(rows) - 1}.returncode",
    )
    if child_returncode != 0:
        raise SummaryError("thermal guard child_exit returncode is nonzero")
    status = _integer(rows[-1].get("status"), f"thermal-guard.jsonl:{len(rows)}.status")
    if status != 0:
        raise SummaryError("thermal guard exit status is nonzero")
    rss_values: list[float] = []
    rss_hwm_values: list[float] = []
    for index, row in enumerate(rows, 1):
        event = row.get("event")
        if event == "abort":
            raise SummaryError(f"thermal guard abort at line {index}")
        process = row.get("process")
        if process is not None:
            process = _mapping(process, f"thermal-guard.jsonl:{index}.process")
            rss = process.get("rss_kib")
            if rss is not None:
                rss_values.append(_number(
                    rss,
                    f"thermal-guard.jsonl:{index}.process.rss_kib",
                    non_negative=True,
                ))
            rss_hwm = process.get("rss_hwm_kib")
            if rss_hwm is not None:
                rss_hwm_values.append(
                    _number(
                        rss_hwm,
                        f"thermal-guard.jsonl:{index}.process.rss_hwm_kib",
                        non_negative=True,
                    )
                )
    return (
        int(max(rss_values)) if rss_values else None,
        int(max(rss_hwm_values)) if rss_hwm_values else None,
    )


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise SummaryError("full-model decode has no samples")
    mean = sum(samples) / len(samples)
    stddev = math.sqrt(sum((sample - mean) ** 2 for sample in samples) / len(samples))
    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "mean": mean,
        "stddev": stddev,
        "median": median,
        "cv": stddev / mean,
    }


def _telemetry_summary(path: Path) -> tuple[
    int, dict[str, int], int, int, int, int, int, int, int | None, int | None
]:
    count = 0
    temperatures: dict[str, int] = {}
    cpu_values: list[float] = []
    npu_values: list[float] = []
    swap_values: list[float] = []
    rss_values: list[float] = []
    rss_hwm_values: list[float] = []
    for index, row in enumerate(_jsonl(path), 1):
        count += 1
        system = row.get("system", row)
        system = _mapping(system, f"telemetry.jsonl:{index}.system")
        zones = _field(system, "thermal_zones", "thermals")
        if zones is None:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: thermal_zones missing")
        zone_count = 0
        for zone_type, temperature in _zone_values(zones, f"telemetry.jsonl:{index}.thermal_zones"):
            zone_count += 1
            temperatures[zone_type] = max(temperatures.get(zone_type, int(temperature)), int(temperature))
        if zone_count == 0:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}.thermal_zones: empty")
        cpu = _field(system, "cpu_frequencies", "cpu_freq_khz", "cpu_freq", "cpu")
        if cpu is None:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: cpu frequencies missing")
        row_cpu_values = list(_values(cpu, f"telemetry.jsonl:{index}.cpu_frequencies"))
        if not row_cpu_values:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: cpu frequencies empty")
        if any(value <= 0 for value in row_cpu_values):
            raise SummaryError(
                f"schema error at telemetry.jsonl:{index}.cpu_frequencies: "
                "expected finite non-negative frequency greater than zero"
            )
        cpu_values.extend(row_cpu_values)
        npu = _field(system, "npu_frequencies", "npu_freq_hz", "npu_frequency_hz", "npu_freq", "npu")
        if npu is None:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: npu frequencies missing")
        row_npu_values = list(_values(npu, f"telemetry.jsonl:{index}.npu_frequencies"))
        if not row_npu_values:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: npu frequencies empty")
        if any(value <= 0 for value in row_npu_values):
            raise SummaryError(
                f"schema error at telemetry.jsonl:{index}.npu_frequencies: "
                "expected finite non-negative frequency greater than zero"
            )
        npu_values.extend(row_npu_values)
        memory = _field(system, "memory")
        if memory is None:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: memory missing")
        memory = _mapping(memory, f"telemetry.jsonl:{index}.memory")
        swap = _field(memory, "SwapFree_kib", "swap_free_kib")
        if swap is None:
            raise SummaryError(f"schema error at telemetry.jsonl:{index}: SwapFree missing")
        swap_values.append(_number(
            swap,
            f"telemetry.jsonl:{index}.SwapFree_kib",
            non_negative=True,
        ))
        process = row.get("process")
        if process is not None:
            process = _mapping(process, f"telemetry.jsonl:{index}.process")
            rss = _field(process, "rss_kib", "peak_rss_kib")
            if rss is not None:
                rss_values.append(_number(
                    rss,
                    f"telemetry.jsonl:{index}.process.rss_kib",
                    non_negative=True,
                ))
            rss_hwm = process.get("rss_hwm_kib")
            if rss_hwm is not None:
                rss_hwm_values.append(
                    _number(
                        rss_hwm,
                        f"telemetry.jsonl:{index}.process.rss_hwm_kib",
                        non_negative=True,
                    )
                )
    if count == 0:
        raise SummaryError("telemetry.jsonl is empty")
    return (
        count,
        temperatures,
        int(min(cpu_values)),
        int(max(cpu_values)),
        int(min(npu_values)),
        int(max(npu_values)),
        int(min(swap_values)),
        int(max(swap_values)),
        int(max(rss_values)) if rss_values else None,
        int(max(rss_hwm_values)) if rss_hwm_values else None,
    )


def _bundle_rows(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.is_dir():
        raise SummaryError(f"run directory does not exist: {run_dir}")
    paths = {name: run_dir / name for name in ("metadata.json", "stdout.log", "telemetry.jsonl", "thermal-guard.jsonl")}
    for name, path in paths.items():
        if not path.is_file():
            raise SummaryError(f"required input is missing: {name}")
    metadata = _json_file(paths["metadata.json"])
    if _integer(metadata.get("schema_version"), "metadata.schema_version") != 1:
        raise SummaryError("schema error at metadata.schema_version: expected 1")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id != run_dir.name:
        raise SummaryError("schema error: metadata.run_id does not match run directory name")
    for name in ("exit_code", "child_return_code"):
        if _integer(metadata.get(name), f"metadata.{name}") != 0:
            raise SummaryError(f"metadata.{name} must be zero")
    command = _command(metadata.get("command"), "metadata.command")
    expected_n_gen, expected_repetitions = _decode_workload(command)
    metadata_commit = metadata.get("build_commit")
    metadata_number = metadata.get("build_number")
    if metadata_commit is not None and (not isinstance(metadata_commit, str) or not metadata_commit):
        raise SummaryError("schema error at metadata.build_commit")
    if metadata_number is not None:
        _integer(metadata_number, "metadata.build_number")

    guard_peak_rss, guard_peak_rss_hwm = _guard_summary(paths["thermal-guard.jsonl"])
    telemetry = _telemetry_summary(paths["telemetry.jsonl"])
    (
        telemetry_count, temperatures, cpu_min, cpu_max, npu_min, npu_max,
        swap_min, swap_max, peak_rss, peak_rss_hwm,
    ) = telemetry

    commits: set[str] = set()
    numbers: set[int] = set()
    decode_rows: list[tuple[dict[str, Any], list[float]]] = []
    for index, row in enumerate(_jsonl(paths["stdout.log"]), 1):
        commit = row.get("build_commit")
        number = row.get("build_number")
        if commit is not None:
            if not isinstance(commit, str) or not commit:
                raise SummaryError(f"schema error at stdout.log:{index}.build_commit")
            commits.add(commit)
        if number is not None:
            numbers.add(_integer(number, f"stdout.log:{index}.build_number"))
        n_prompt = row.get("n_prompt")
        n_gen = row.get("n_gen")
        if n_prompt != 0 or not isinstance(n_gen, int) or isinstance(n_gen, bool) or n_gen <= 0:
            continue
        if n_gen != expected_n_gen:
            raise SummaryError("Bonsai n_gen does not match metadata.command workload")
        expected_model_fields = {
            "model_filename": BONSAI_MODEL_PATH,
            "model_type": BONSAI_MODEL_TYPE,
            "model_size": BONSAI_BENCH_MODEL_SIZE_BYTES,
            "model_n_params": BONSAI_MODEL_N_PARAMS,
        }
        for field, expected in expected_model_fields.items():
            if row.get(field) != expected:
                raise SummaryError(f"Bonsai model identity mismatch at stdout.log:{index}.{field}")
        raw_samples = row.get("samples_ts")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise SummaryError(f"schema error at stdout.log:{index}.samples_ts: expected nonempty array")
        if len(raw_samples) != expected_repetitions:
            raise SummaryError("Bonsai workload samples do not match command repetitions")
        samples = [
            _number(value, f"stdout.log:{index}.samples_ts[{sample_index}]", positive=True)
            for sample_index, value in enumerate(raw_samples)
        ]
        decode_rows.append((row, samples))
    if not decode_rows:
        raise SummaryError("full-model decode rows are missing")
    if metadata_commit is not None:
        commits.add(metadata_commit)
    if metadata_number is not None:
        numbers.add(metadata_number)
    if len(commits) > 1:
        raise SummaryError("build_commit is inconsistent")
    if len(numbers) > 1:
        raise SummaryError("build_number is inconsistent")
    if not commits or not numbers:
        raise SummaryError("build_commit/build_number are required")
    if commits != {BONSAI_RUNTIME_COMMIT_SHORT} or numbers != {BONSAI_BUILD_NUMBER}:
        raise SummaryError("Bonsai runtime identity does not match the pinned build")

    bundle_elapsed_seconds = _number(metadata.get("elapsed_ns"), "metadata.elapsed_ns") / 1_000_000_000
    measured_rss = [value for value in (peak_rss, guard_peak_rss) if value is not None]
    bundle_peak_rss = max(measured_rss) if measured_rss else None
    base_label = metadata.get("label")
    if not isinstance(base_label, str) or not base_label:
        base_label = run_id
    multi = len(decode_rows) > 1
    results: list[dict[str, Any]] = []
    for case_index, (row, samples) in enumerate(decode_rows, 1):
        backends: list[Any] = []
        backend = _field(row, "backends", "backend", "backend_name")
        if backend is None:
            raise SummaryError(f"schema error at stdout.log decode row {case_index}: backend missing")
        parsed_backends = _canonical_values(backend, f"stdout.log decode row {case_index}.backend")
        if any(not isinstance(item, str) or not item for item in parsed_backends):
            raise SummaryError(f"schema error at stdout.log decode row {case_index}.backend: expected non-empty string")
        backends.extend(parsed_backends)
        unique_backends = _merge_unique(backends)
        if not unique_backends:
            raise SummaryError(f"schema error at stdout.log decode row {case_index}: backend empty")
        devices = _merge_unique(_canonical_values(row.get("devices"), f"stdout.log decode row {case_index}.devices"))
        config: dict[str, Any] = {}
        for field in ("n_threads", "cpu_mask", "poll", "n_gpu_layers"):
            value = row.get(field)
            if value is not None:
                if field in {"n_threads", "poll", "n_gpu_layers"}:
                    _integer(value, f"stdout.log decode row {case_index}.{field}")
                config[field] = value
        case_run_id = f"{run_id}--case-{case_index:03d}" if multi else run_id
        label_parts = [base_label]
        for field, label_name in (("poll", "poll"), ("n_threads", "threads"), ("cpu_mask", "mask"), ("n_gpu_layers", "ngl")):
            if field in config:
                label_parts.append(f"{label_name}={config[field]}")
        result: dict[str, Any] = {
            "run_id": case_run_id,
            "label": " ".join(label_parts),
            "command": command,
            "metric_scope": "full_model_decode",
            "unit": "tokens_per_second",
            "backend": unique_backends[0] if len(unique_backends) == 1 else "mixed",
            "backends": unique_backends,
            "n_threads": config.get("n_threads"),
            "cpu_mask": config.get("cpu_mask"),
            "poll": config.get("poll"),
            "n_gpu_layers": config.get("n_gpu_layers"),
            "n_gen": expected_n_gen,
            "repetitions": expected_repetitions,
            "devices": devices,
            "samples": samples,
            **_stats(samples),
            "telemetry_scope": "bundle",
            "bundle_run_id": run_id,
            "bundle_elapsed_seconds": bundle_elapsed_seconds,
            "telemetry_samples": telemetry_count,
            "peak_temperatures_millidegrees_c": temperatures,
            "min_cpu_frequencies_khz": cpu_min,
            "max_cpu_frequencies_khz": cpu_max,
            "min_npu_frequency_hz": npu_min,
            "max_npu_frequency_hz": npu_max,
            "min_swap_free_kib": swap_min,
            "max_swap_free_kib": swap_max,
            "quality": {"status": "not_checked", "exact_match": False},
        }
        if bundle_peak_rss is not None:
            result["peak_rss_kib"] = bundle_peak_rss
        measured_rss_hwm = [
            value for value in (peak_rss_hwm, guard_peak_rss_hwm) if value is not None
        ]
        if measured_rss_hwm:
            result["peak_rss_hwm_kib"] = max(measured_rss_hwm)
        results.append(result)
    return results


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = _bundle_rows(run_dir)
    if len(rows) != 1:
        raise SummaryError("bundle contains multiple decode rows; use summarize_runs")
    return rows[0]


def _file_sha256(path: Path, context: str) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SummaryError(f"cannot read {context}: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def summarize_runs(
    run_dirs: Iterable[Path],
    *,
    reference_run_id: str | None = None,
    candidate_run_id: str | None = None,
    golden_reference_stdout: Path | None = None,
    golden_candidate_stdout: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(_bundle_rows(Path(run_dir)))
    if not rows:
        raise SummaryError("at least one run directory is required")
    rows.sort(key=lambda row: row["run_id"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "series": "Bonsai-27B Q1_0 / A733 full-model decode",
        "model_sha256": BONSAI_MODEL_SHA256,
        "model_size_bytes": BONSAI_FILE_SIZE_BYTES,
        "runtime_commit": BONSAI_RUNTIME_COMMIT,
        "rows": rows,
    }
    quality_arguments = (
        reference_run_id,
        candidate_run_id,
        golden_reference_stdout,
        golden_candidate_stdout,
    )
    if any(argument is not None for argument in quality_arguments):
        if any(argument is None for argument in quality_arguments):
            raise SummaryError("chart-ready mode requires both run ids and both golden stdout files")
        assert reference_run_id is not None
        assert candidate_run_id is not None
        assert golden_reference_stdout is not None
        assert golden_candidate_stdout is not None
        if reference_run_id == candidate_run_id:
            raise SummaryError("chart-ready reference and candidate run ids must be distinct")
        selected_rows: dict[str, dict[str, Any]] = {}
        for run_id, role in (
            (reference_run_id, "reference"),
            (candidate_run_id, "candidate"),
        ):
            matches = [row for row in rows if row["run_id"] == run_id]
            if len(matches) != 1:
                raise SummaryError(f"chart-ready {role} run id must resolve exactly once")
            selected_rows[role] = matches[0]
        workload_fields = ("n_gen", "repetitions")
        if any(
            selected_rows["reference"].get(field)
            != selected_rows["candidate"].get(field)
            for field in workload_fields
        ):
            raise SummaryError(
                "chart-ready reference and candidate workload n_gen/repetitions differ"
            )
        for row in selected_rows.values():
            row["quality"] = {"status": "exact", "exact_match": True}
        reference_row = selected_rows["reference"]
        reference_row["reference"] = True
        reference_sha256 = _file_sha256(
            golden_reference_stdout, "golden reference stdout"
        )
        candidate_sha256 = _file_sha256(
            golden_candidate_stdout, "golden candidate stdout"
        )
        if (
            reference_sha256 != BONSAI_GOLDEN_SHA256
            or candidate_sha256 != BONSAI_GOLDEN_SHA256
        ):
            raise SummaryError("chart-ready exact golden stdout SHA-256 mismatch")
        result.update({
            "series": CHART_SERIES,
            "reference_run_id": reference_run_id,
            "quality_evidence": {
                "method": "deterministic greedy 32-token exact stdout comparison",
                "reference_run_id": golden_reference_stdout.parent.name,
                "candidate_run_id": golden_candidate_stdout.parent.name,
                "reference_sha256": reference_sha256,
                "candidate_sha256": candidate_sha256,
                "exact_match": True,
            },
        })
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Bonsai full-model decode runs")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-run-id")
    parser.add_argument("--candidate-run-id")
    parser.add_argument("--golden-reference-stdout", type=Path)
    parser.add_argument("--golden-candidate-stdout", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = summarize_runs(
            args.run_dirs,
            reference_run_id=args.reference_run_id,
            candidate_run_id=args.candidate_run_id,
            golden_reference_stdout=args.golden_reference_stdout,
            golden_candidate_stdout=args.golden_candidate_stdout,
        )
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
    except FileExistsError as error:
        print(f"summarize_bonsai_night: refusing to overwrite output: {error.filename}", file=sys.stderr)
        return 2
    except (OSError, SummaryError) as error:
        print(f"summarize_bonsai_night: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
