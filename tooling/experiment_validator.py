#!/usr/bin/env python3
"""Validate the evidence layout of one E053 experiment.

The validator is deliberately independent of a model runtime.  It checks the
paper trail around an experiment: Russian README, hypothesis preflight,
provenance, an explicit lifecycle status, and hashes for raw artifacts.  A
``planned`` experiment may exist before any model is run; every other status
must carry the unmodified stdout/stderr, telemetry and trace streams.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker

try:
    from tooling.branch_preflight import (
        duplicate_decision_from_refs,
        experiment_root_topology,
        index_experiment_root_evidence,
        list_refs,
        ref_commit,
        scan_refs,
        tree_binding_sha256,
        untracked_experiment_roots,
    )
except ModuleNotFoundError:  # direct: python3 tooling/experiment_validator.py
    from branch_preflight import (
        duplicate_decision_from_refs,
        experiment_root_topology,
        index_experiment_root_evidence,
        list_refs,
        ref_commit,
        scan_refs,
        tree_binding_sha256,
        untracked_experiment_roots,
    )


MANIFEST_SCHEMA = "e053-experiment-manifest/v1"
SUMMARY_SCHEMA = "e053-experiment-summary/v1"
VALID_STATUSES = {"planned", "passed", "failed", "rejected", "blocked"}
EXECUTED_STATUSES = VALID_STATUSES - {"planned"}
COMMON_PROMPT = "Explain the A733 memory bottleneck in one short sentence."
COMMON_PROMPT_SHA256 = "3422031cc96896c8aff5ea0363ef6cddca9ba485437f56dbfa63d7703f01a7de"
SUMMARY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/schema/e053-experiment-summary.schema.json"
)
TRACE_AB_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/schema/e047-trace-ab-result.schema.json"
)
REQUIRED_FILES = (
    "README.md",
    "hypothesis-preflight.md",
    "commands.txt",
    "environment.json",
    "device.json",
    "data/branch-preflight.json",
    "data/manifest.json",
    "results/summary.json",
)
REQUIRED_RAW_FILES = (
    "raw/stdout.log",
    "raw/stderr.log",
    "raw/telemetry.jsonl",
    "raw/trace.jsonl",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
TELEMETRY_REQUIRED = {
    "schema_version",
    "run_id",
    "timestamp_ns",
    "cpu_temperature_c",
    "cpu_frequency_hz",
    "npu_frequency_hz",
    "rss_bytes",
    "ddr",
}
TRACE_REQUIRED = {
    "schema_version",
    "run_id",
    "step",
    "phase",
    "token_index",
    "layer_index",
    "layer_scope",
    "op_index",
    "op_name",
    "backend",
    "start_ns",
    "end_ns",
    "duration_ns",
    "inputs",
    "outputs",
    "memory",
}
TRACE_BACKENDS = {"cpu", "npu", "gpu", "cpu+npu", "host"}
TRACE_PHASES = {"load", "tokenize", "prefill", "decode", "sample", "sync"}
DIRECT_METHODS = {"hardware_counter", "pmu_counter_delta", "uncore_counter_delta"}
MEMORY_PROVENANCE_REQUIRED = {
    "measurement_method",
    "measurement_source",
    "measurement_confidence",
}
DDR_RELATION_FIELD = "aggregation_relation"
DDR_RUN_TOTAL_SOURCE = (
    "results/summary.json:runs[].memory_accounting.observed_direct_ddr"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def validate_trace_ab_result(value: dict[str, Any]) -> list[str]:
    """Validate cross-field invariants JSON Schema cannot express."""

    errors: list[str] = []
    workload = value.get("workload_identity")
    if not isinstance(workload, dict):
        return ["workload_identity: нужен JSON-объект"]
    workload_hash = _canonical_sha256(workload)
    declared_hash = value.get("workload_identity_sha256")
    identity_valid = declared_hash == workload_hash
    sides: list[dict[str, Any]] = []
    pair_sets: list[set[str]] = []
    token_hashes: set[str] = set()
    for side_name in ("trace_off", "trace_on"):
        side = value.get(side_name)
        if not isinstance(side, dict):
            errors.append(f"{side_name}: нужен JSON-объект")
            identity_valid = False
            continue
        sides.append(side)
        if side.get("workload_identity_sha256") != workload_hash:
            identity_valid = False
        samples = side.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append(f"{side_name}.samples: нужен непустой массив")
            continue
        side_pairs: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict):
                errors.append(f"{side_name}.samples: элемент должен быть объектом")
                continue
            sample_pair_id = sample.get("pair_id")
            if not _nonempty_string(sample_pair_id):
                errors.append(f"{side_name}: у образца отсутствует pair_id")
            elif sample_pair_id in side_pairs:
                errors.append(f"{side_name}: pair_id образца повторяется")
            else:
                side_pairs.add(str(sample_pair_id))
            token_hash = sample.get("token_ids_sha256")
            if isinstance(token_hash, str):
                token_hashes.add(token_hash)
        pair_sets.append(side_pairs)

    if len(pair_sets) == 2 and pair_sets[0] != pair_sets[1]:
        errors.append("trace_off/trace_on: наборы pair_id не совпадают")
        identity_valid = False

    token_stream_equal = value.get("token_stream_equal") is True and len(token_hashes) == 1
    invalid = not identity_valid or not token_stream_equal
    statistics_value = value.get("statistics")
    median_overhead = None
    p95_overhead = None
    if len(sides) == 2 and isinstance(statistics_value, dict):
        try:
            off_values = [float(item["latency_ms"]) for item in sides[0]["samples"]]
            on_values = [float(item["latency_ms"]) for item in sides[1]["samples"]]
            off_median = statistics.median(off_values)
            on_median = statistics.median(on_values)
            off_p95 = _nearest_rank(off_values, 0.95)
            on_p95 = _nearest_rank(on_values, 0.95)
            median_overhead = on_median / off_median - 1.0
            p95_overhead = on_p95 / off_p95 - 1.0
            expected_numbers = {
                "off_median_ms": off_median,
                "on_median_ms": on_median,
                "off_p95_ms": off_p95,
                "on_p95_ms": on_p95,
                "median_overhead_fraction": median_overhead,
                "p95_overhead_fraction": p95_overhead,
            }
            for field, expected in expected_numbers.items():
                actual = statistics_value.get(field)
                if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
                ):
                    errors.append(f"statistics.{field}: значение не вычислено из samples")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            errors.append("statistics: невозможно вычислить median/p95 из samples")
            invalid = True
    else:
        invalid = True

    if invalid:
        expected_classification = "INVALID"
    elif median_overhead is not None and p95_overhead is not None and max(
        median_overhead, p95_overhead
    ) <= 0.01:
        expected_classification = "PASS"
    else:
        expected_classification = "TRACE_ONLY"
    if value.get("classification") != expected_classification:
        errors.append(
            f"classification: ожидается {expected_classification} по identity/token/overhead gate"
        )
    return errors


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: не удалось прочитать JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: корневое значение должно быть JSON-объектом")
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        return None
    return path.as_posix()


def _validate_manifest(
    root: Path,
    manifest: dict[str, Any],
    status: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"data/manifest.json: нужен schema_version {MANIFEST_SCHEMA}")
    if manifest.get("status") != status:
        errors.append("data/manifest.json: status не совпадает с results/summary.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        errors.append("data/manifest.json: files должен быть массивом")
        return {}

    seen: set[str] = set()
    entries_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"data/manifest.json: files[{index}] должен быть объектом")
            continue
        relative = _safe_relative_path(entry.get("path"))
        if relative is None:
            errors.append(f"data/manifest.json: files[{index}].path небезопасен")
            continue
        if relative in seen:
            errors.append(f"data/manifest.json: путь повторяется: {relative}")
        seen.add(relative)
        entries_by_path[relative] = entry
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"data/manifest.json: путь выходит за пределы эксперимента: {relative}")
            continue
        if not path.is_file():
            errors.append(f"data/manifest.json: файл отсутствует: {relative}")
            continue
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            errors.append(f"data/manifest.json: некорректный sha256 для {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"data/manifest.json: sha256 не совпадает для {relative}")
        declared_size = entry.get("size_bytes")
        if declared_size is not None and declared_size != path.stat().st_size:
            errors.append(f"data/manifest.json: size_bytes не совпадает для {relative}")

    if status in EXECUTED_STATUSES:
        listed = seen
        for relative in REQUIRED_RAW_FILES:
            if relative not in listed:
                errors.append(f"data/manifest.json: raw-файл не включён в manifest: {relative}")
    for relative in REQUIRED_FILES:
        if relative != "data/manifest.json" and relative not in seen:
            errors.append(f"data/manifest.json: обязательный файл не включён в manifest: {relative}")
    return entries_by_path


def _validate_capture_provenance(
    path: Path,
    value: dict[str, Any],
    status: str | None,
    errors: list[str],
) -> None:
    if not isinstance(value.get("captured"), bool):
        errors.append(f"{path.name}: поле captured должно быть boolean")
    if not isinstance(value.get("status"), str) or not value["status"].strip():
        errors.append(f"{path.name}: поле status должно быть непустой строкой")
    expected_schema = "e053-environment/v1" if path.name == "environment.json" else "e053-device/v1"
    if value.get("schema_version") != expected_schema:
        errors.append(f"{path.name}: нужен schema_version {expected_schema}")
    if status not in EXECUTED_STATUSES:
        return
    if value.get("captured") is not True:
        errors.append(f"{path.name}: для executed status требуется captured=true")
    if value.get("status") != "captured":
        errors.append(f"{path.name}: для executed status требуется status=captured")
    if path.name == "environment.json":
        _validate_environment(value, errors)
    else:
        _validate_device(value, errors)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_byte_values(
    item: object,
    fields: set[str],
    prefix: str,
    errors: list[str],
    *,
    allowed_methods: set[str],
) -> None:
    _validate_memory_provenance(
        item,
        prefix,
        errors,
        allowed_methods=allowed_methods | {"unavailable"},
    )
    if not isinstance(item, dict):
        return
    missing = fields - item.keys()
    if missing:
        errors.append(f"{prefix}: отсутствуют {', '.join(sorted(missing))}")
    method = item.get("measurement_method")
    unavailable = method == "unavailable"
    for field in fields:
        value = item.get(field)
        if unavailable:
            if value is not None:
                errors.append(f"{prefix}.{field}: при unavailable требуется null")
        elif not _nonnegative_int(value):
            errors.append(f"{prefix}.{field}: нужен integer >= 0 для метода {method}")
    if unavailable and item.get("measurement_confidence") != "unavailable":
        errors.append(f"{prefix}: unavailable требует measurement_confidence=unavailable")


def _validate_environment(value: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(value.get("repo_commit"), str) or not COMMIT_RE.fullmatch(value["repo_commit"]):
        errors.append("environment.json: repo_commit должен быть полным git commit")
    for group, fields in {
        "runtime": ("name", "version", "commit"),
        "compiler": ("name", "version"),
        "sdk": ("used", "name", "version"),
        "driver": ("name", "version"),
        "kernel": ("release", "build"),
    }.items():
        item = value.get(group)
        if not isinstance(item, dict):
            errors.append(f"environment.json: отсутствует объект {group}")
            continue
        for field in fields:
            field_value = item.get(field)
            if group == "sdk" and field == "used":
                if not isinstance(field_value, bool):
                    errors.append("environment.json: sdk.used должен быть boolean")
            elif not _nonempty_string(field_value):
                errors.append(f"environment.json: отсутствует {group}.{field}")
        if group == "runtime" and (
            not isinstance(item.get("commit"), str) or not COMMIT_RE.fullmatch(item["commit"])
        ):
            errors.append("environment.json: runtime.commit должен быть полным git commit")
    if not _nonempty_string(value.get("captured_at_utc")):
        errors.append("environment.json: отсутствует captured_at_utc")


def _validate_device(value: dict[str, Any], errors: list[str]) -> None:
    for field in (
        "device_id",
        "board",
        "soc",
        "npu",
        "os_release",
        "captured_at_utc",
    ):
        if not _nonempty_string(value.get(field)):
            errors.append(f"device.json: отсутствует {field}")
    if not isinstance(value.get("memory_bytes"), int) or value["memory_bytes"] <= 0:
        errors.append("device.json: memory_bytes должен быть положительным integer")


def _validate_summary_schema(summary: dict[str, Any], errors: list[str]) -> None:
    try:
        schema = json.loads(SUMMARY_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"summary schema: не удалось загрузить: {exc}")
        return
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for error in sorted(validator.iter_errors(summary), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "$"
        errors.append(f"results/summary.json schema {location}: {error.message}")


def _validate_run_memory_accounting(value: object, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    groups = {
        "logical": (
            {"logical_bytes"},
            {"static_tensor_accounting", "runtime_tensor_accounting"},
        ),
        "unique_weights": (
            {"unique_weight_bytes"},
            {"static_tensor_accounting", "runtime_tensor_accounting"},
        ),
        "observed_direct_ddr": (
            {"observed_direct_ddr_read_bytes", "observed_direct_ddr_write_bytes"},
            DIRECT_METHODS,
        ),
        "inferred_ddr": (
            {"inferred_ddr_read_bytes", "inferred_ddr_write_bytes"},
            {"analytical_estimate", "counter_model_fit", "simulation"},
        ),
    }
    for group, (byte_fields, methods) in groups.items():
        item = value.get(group)
        if not isinstance(item, dict):
            continue
        allowed = byte_fields | MEMORY_PROVENANCE_REQUIRED
        unexpected = set(item) - allowed
        if unexpected:
            errors.append(
                f"{prefix}.{group}: смешивает классы измерений: {', '.join(sorted(unexpected))}"
            )
        _validate_byte_values(
            item,
            byte_fields,
            f"{prefix}.{group}",
            errors,
            allowed_methods=methods,
        )


def _validate_run_bindings(summary: dict[str, Any], commands_text: str, errors: list[str]) -> set[str]:
    run_ids: set[str] = set()
    commands = {line.strip() for line in commands_text.splitlines() if line.strip() and not line.startswith("#")}
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return run_ids
    any_model_used = False
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        prefix = f"results/summary.json runs[{index}]"
        run_id = run.get("run_id")
        if _nonempty_string(run_id):
            if run_id in run_ids:
                errors.append(f"{prefix}: run_id повторяется")
            run_ids.add(str(run_id))
        model = run.get("model")
        if isinstance(model, dict) and model.get("used") is True:
            any_model_used = True
            if not isinstance(model.get("sha256"), str) or not SHA256_RE.fullmatch(model["sha256"]):
                errors.append(f"{prefix}: при model.used=true нужен model.sha256")
        command = run.get("command")
        workload = run.get("workload")
        _validate_run_memory_accounting(run.get("memory_accounting"), prefix, errors)
        if isinstance(command, dict):
            shell = command.get("shell")
            if not _nonempty_string(shell) or shell not in commands:
                errors.append(f"{prefix}: command.shell отсутствует точной строкой в commands.txt")
            elif command.get("sha256") != hashlib.sha256(shell.encode("utf-8")).hexdigest():
                errors.append(f"{prefix}: command.sha256 не совпадает")
            if isinstance(workload, dict) and command.get("workload_sha256") != _canonical_sha256(
                workload
            ):
                errors.append(f"{prefix}: command.workload_sha256 не связывает команду с workload")
        if isinstance(workload, dict):
            if workload.get("common_prompt_text") != COMMON_PROMPT:
                errors.append(f"{prefix}: common_prompt_text не равен контракту E047")
            if workload.get("common_prompt_sha256") != COMMON_PROMPT_SHA256:
                errors.append(f"{prefix}: common_prompt_sha256 не равен контракту E047")
            rendered = workload.get("rendered_prompt_text")
            if not _nonempty_string(rendered) or workload.get("rendered_prompt_sha256") != hashlib.sha256(
                str(rendered).encode("utf-8")
            ).hexdigest():
                errors.append(f"{prefix}: rendered prompt/hash не связаны")
            token_ids = workload.get("prompt_token_ids")
            if isinstance(token_ids, list):
                expected = _canonical_sha256(token_ids)
                if workload.get("prompt_token_ids_sha256") != expected:
                    errors.append(f"{prefix}: prompt_token_ids_sha256 не совпадает")
    if summary.get("model_used") is not any_model_used:
        errors.append("results/summary.json: model_used не совпадает с runs[].model.used")
    expected_result_status = {
        "passed": "passed",
        "failed": "failed",
        "rejected": "rejected",
        "blocked": "blocked",
    }.get(summary.get("status"))
    if expected_result_status is not None and runs and not any(
        isinstance(run, dict)
        and isinstance(run.get("result"), dict)
        and run["result"].get("status") == expected_result_status
        for run in runs
    ):
        errors.append(
            f"results/summary.json: status={summary.get('status')} требует хотя бы один run result={expected_result_status}"
        )
    return run_ids


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path.name}: не удалось прочитать JSONL: {exc}")
        return []
    if not lines or not any(line.strip() for line in lines):
        errors.append(f"{path.name}: JSONL должен быть непустым")
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: некорректный JSON: {exc.msg}")
            continue
        if not isinstance(event, dict):
            errors.append(f"{path.name}:{line_number}: событие должно быть объектом")
            continue
        events.append(event)
    return events


def _validate_memory_provenance(
    value: object,
    prefix: str,
    errors: list[str],
    *,
    allowed_methods: set[str] | None = None,
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: нужен объект memory/ddr")
        return
    missing = MEMORY_PROVENANCE_REQUIRED - value.keys()
    if missing:
        errors.append(f"{prefix}: отсутствуют {', '.join(sorted(missing))}")
    if value.get("measurement_confidence") not in {"high", "medium", "low", "unavailable"}:
        errors.append(f"{prefix}: measurement_confidence вне enum")
    if not _nonempty_string(value.get("measurement_source")):
        errors.append(f"{prefix}: measurement_source должен быть непустой строкой")
    method = value.get("measurement_method")
    if allowed_methods is not None and (
        not isinstance(method, str) or method not in allowed_methods
    ):
        errors.append(f"{prefix}: measurement_method смешивает классы измерений")


def _validate_telemetry_events(events: list[dict[str, Any]], run_ids: set[str], errors: list[str]) -> None:
    for index, event in enumerate(events, start=1):
        prefix = f"telemetry.jsonl:{index}"
        missing = TELEMETRY_REQUIRED - event.keys()
        if missing:
            errors.append(f"{prefix}: отсутствуют {', '.join(sorted(missing))}")
            continue
        unexpected = set(event) - TELEMETRY_REQUIRED
        if unexpected:
            errors.append(f"{prefix}: неизвестные поля {', '.join(sorted(unexpected))}")
        if event.get("schema_version") != "e047-telemetry-event/v1":
            errors.append(f"{prefix}: неверный schema_version")
        if not _nonempty_string(event.get("run_id")) or event.get("run_id") not in run_ids:
            errors.append(f"{prefix}: неизвестный run_id")
        if not _nonnegative_int(event.get("timestamp_ns")):
            errors.append(f"{prefix}: timestamp_ns должен быть >= 0")
        cpu_temperature = event.get("cpu_temperature_c")
        if not _finite_number(cpu_temperature) or not -40.0 <= float(cpu_temperature) <= 150.0:
            errors.append(f"{prefix}: cpu_temperature_c должен быть finite в диапазоне [-40, 150]")
        frequencies = event.get("cpu_frequency_hz")
        if not isinstance(frequencies, list) or not frequencies:
            errors.append(f"{prefix}: cpu_frequency_hz должен быть непустым массивом")
        elif any(not _nonnegative_int(value) or value > 10_000_000_000 for value in frequencies):
            errors.append(f"{prefix}: cpu_frequency_hz содержит значение вне диапазона")
        npu_frequency = event.get("npu_frequency_hz")
        if npu_frequency is not None and (
            not _nonnegative_int(npu_frequency) or npu_frequency > 10_000_000_000
        ):
            errors.append(f"{prefix}: npu_frequency_hz вне диапазона")
        if not _nonnegative_int(event.get("rss_bytes")):
            errors.append(f"{prefix}: rss_bytes должен быть integer >= 0")
        ddr_fields = {
            "observed_direct_ddr_read_bytes",
            "observed_direct_ddr_write_bytes",
        }
        _validate_byte_values(
            event.get("ddr"),
            ddr_fields,
            f"{prefix}.ddr",
            errors,
            allowed_methods=DIRECT_METHODS,
        )
        ddr = event.get("ddr")
        if isinstance(ddr, dict):
            unexpected = set(ddr) - (
                ddr_fields | MEMORY_PROVENANCE_REQUIRED | {DDR_RELATION_FIELD}
            )
            if unexpected:
                errors.append(f"{prefix}.ddr: смешивает классы измерений")


def _validate_trace_memory(value: object, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: нужен объект memory")
        return
    groups = {
        "logical": (
            {"logical_read_bytes", "logical_write_bytes"},
            {"static_tensor_accounting", "runtime_tensor_accounting"},
        ),
        "unique_weights": (
            {"unique_weight_bytes"},
            {"static_tensor_accounting", "runtime_tensor_accounting"},
        ),
        "observed_direct_ddr": (
            {"observed_direct_ddr_read_bytes", "observed_direct_ddr_write_bytes"},
            DIRECT_METHODS,
        ),
        "inferred_ddr": (
            {"inferred_ddr_read_bytes", "inferred_ddr_write_bytes"},
            {"analytical_estimate", "counter_model_fit", "simulation"},
        ),
    }
    for group, (byte_fields, methods) in groups.items():
        item = value.get(group)
        _validate_byte_values(
            item,
            byte_fields,
            f"{prefix}.{group}",
            errors,
            allowed_methods=methods,
        )
        if isinstance(item, dict):
            allowed = byte_fields | MEMORY_PROVENANCE_REQUIRED
            if group == "observed_direct_ddr":
                allowed = allowed | {DDR_RELATION_FIELD}
            unexpected = set(item) - allowed
            if unexpected:
                errors.append(
                    f"{prefix}.{group}: смешивает классы измерений: "
                    + ", ".join(sorted(unexpected))
                )


def _validate_trace_events(events: list[dict[str, Any]], run_ids: set[str], errors: list[str]) -> None:
    for index, event in enumerate(events, start=1):
        prefix = f"trace.jsonl:{index}"
        missing = TRACE_REQUIRED - event.keys()
        if missing:
            errors.append(f"{prefix}: отсутствуют {', '.join(sorted(missing))}")
            continue
        unexpected = set(event) - TRACE_REQUIRED
        if unexpected:
            errors.append(f"{prefix}: неизвестные поля {', '.join(sorted(unexpected))}")
        if event.get("schema_version") != "e047-trace-event/v1":
            errors.append(f"{prefix}: неверный schema_version")
        if not _nonempty_string(event.get("run_id")) or event.get("run_id") not in run_ids:
            errors.append(f"{prefix}: неизвестный run_id")
        for field in ("step", "token_index", "op_index", "start_ns", "end_ns", "duration_ns"):
            if not _nonnegative_int(event.get(field)):
                errors.append(f"{prefix}: {field} должен быть integer >= 0")
        layer_index = event.get("layer_index")
        layer_scope = event.get("layer_scope")
        if layer_scope == "global":
            if layer_index is not None:
                errors.append(f"{prefix}: layer_scope=global требует layer_index=null")
        elif layer_scope == "layer":
            if not _nonnegative_int(layer_index):
                errors.append(f"{prefix}: layer_scope=layer требует layer_index integer >= 0")
        else:
            errors.append(f"{prefix}: layer_scope должен быть layer/global")
        if not isinstance(event.get("phase"), str) or event.get("phase") not in TRACE_PHASES:
            errors.append(f"{prefix}: phase вне enum")
        if not isinstance(event.get("backend"), str) or event.get("backend") not in TRACE_BACKENDS:
            errors.append(f"{prefix}: backend вне enum")
        if not _nonempty_string(event.get("op_name")):
            errors.append(f"{prefix}: op_name должен быть непустой строкой")
        if all(_nonnegative_int(event.get(field)) for field in ("start_ns", "end_ns", "duration_ns")):
            if event["end_ns"] < event["start_ns"]:
                errors.append(f"{prefix}: end_ns должен быть >= start_ns")
            if event["end_ns"] - event["start_ns"] != event["duration_ns"]:
                errors.append(f"{prefix}: duration_ns != end_ns-start_ns")
        if not isinstance(event.get("inputs"), list) or not isinstance(event.get("outputs"), list):
            errors.append(f"{prefix}: inputs/outputs должны быть массивами")
        elif any(not _nonempty_string(value) for value in event["inputs"] + event["outputs"]):
            errors.append(f"{prefix}: inputs/outputs должны содержать непустые строки")
        _validate_trace_memory(event.get("memory"), f"{prefix}.memory", errors)


def _ddr_relation(
    value: dict[str, Any], prefix: str, errors: list[str]
) -> str | None:
    relation = value.get(DDR_RELATION_FIELD)
    if relation is None:
        return "exact_run_total"
    if not isinstance(relation, dict) or set(relation) != {"kind", "run_total_source"}:
        errors.append(f"{prefix}: aggregation_relation должна явно задать kind и run_total_source")
        return None
    kind = relation.get("kind")
    if kind not in {"exact_run_total", "partition_of_run_total"}:
        errors.append(f"{prefix}: неизвестный aggregation_relation.kind")
        return None
    if relation.get("run_total_source") != DDR_RUN_TOTAL_SOURCE:
        errors.append(f"{prefix}: aggregation_relation не ссылается на summary run total")
        return None
    return str(kind)


def _direct_ddr_values(value: object) -> tuple[object, object, object, object, object] | None:
    if not isinstance(value, dict):
        return None
    return (
        value.get("observed_direct_ddr_read_bytes"),
        value.get("observed_direct_ddr_write_bytes"),
        value.get("measurement_method"),
        value.get("measurement_source"),
        value.get("measurement_confidence"),
    )


def _validate_ddr_stream_against_summary(
    stream_name: str,
    events: list[dict[str, Any]],
    summary_ddr: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    groups_by_run: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        run_id = event.get("run_id")
        if not isinstance(run_id, str):
            continue
        if stream_name == "telemetry.jsonl":
            direct = event.get("ddr")
        else:
            memory = event.get("memory")
            direct = memory.get("observed_direct_ddr") if isinstance(memory, dict) else None
        if isinstance(direct, dict):
            groups_by_run.setdefault(run_id, []).append(direct)

    for run_id, run_total in summary_ddr.items():
        groups = groups_by_run.get(run_id, [])
        if not groups:
            errors.append(f"{stream_name}: direct DDR не связан с run_id={run_id}")
            continue
        expected = _direct_ddr_values(run_total)
        relations: list[str] = []
        for index, group in enumerate(groups, start=1):
            prefix = f"{stream_name} run_id={run_id} event[{index}] direct DDR"
            relation = _ddr_relation(group, prefix, errors)
            if relation is not None:
                relations.append(relation)
            actual = _direct_ddr_values(group)
            if expected is not None and actual is not None and actual[2:] != expected[2:]:
                errors.append(
                    f"{prefix}: method/source/confidence противоречат results/summary.json"
                )
        if len(relations) != len(groups) or len(set(relations)) != 1:
            errors.append(
                f"{stream_name} run_id={run_id}: direct DDR смешивает aggregation relation"
            )
            continue
        if expected is None:
            continue
        if relations[0] == "exact_run_total":
            if any(_direct_ddr_values(group) != expected for group in groups):
                errors.append(
                    f"{stream_name} run_id={run_id}: direct DDR противоречит summary run total"
                )
        else:
            read_values = [group.get("observed_direct_ddr_read_bytes") for group in groups]
            write_values = [group.get("observed_direct_ddr_write_bytes") for group in groups]
            if not all(_nonnegative_int(value) for value in read_values + write_values):
                errors.append(
                    f"{stream_name} run_id={run_id}: direct DDR partitions требуют integer bytes"
                )
                continue
            if sum(read_values) != expected[0] or sum(write_values) != expected[1]:
                errors.append(
                    f"{stream_name} run_id={run_id}: direct DDR partitions не равны summary run total"
                )


def _validate_cross_stream_ddr(
    summary: dict[str, Any],
    telemetry_events: list[dict[str, Any]],
    trace_events: list[dict[str, Any]],
    errors: list[str],
) -> None:
    summary_ddr: dict[str, dict[str, Any]] = {}
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
            continue
        memory = run.get("memory_accounting")
        direct = memory.get("observed_direct_ddr") if isinstance(memory, dict) else None
        if isinstance(direct, dict):
            summary_ddr[run["run_id"]] = direct
    _validate_ddr_stream_against_summary(
        "telemetry.jsonl", telemetry_events, summary_ddr, errors
    )
    _validate_ddr_stream_against_summary(
        "trace.jsonl", trace_events, summary_ddr, errors
    )


def _validate_preflight(root: Path, value: dict[str, Any], summary: dict[str, Any] | None, errors: list[str]) -> None:
    if value.get("schema_version") != "e053-branch-preflight/v3":
        errors.append("data/branch-preflight.json: нужен schema_version e053-branch-preflight/v3")
    if not _nonempty_string(value.get("hypothesis_query")):
        errors.append("data/branch-preflight.json: отсутствует hypothesis_query")
    if summary is not None and str(value.get("experiment_id", "")).upper() != str(
        summary.get("experiment_id", "")
    ).upper():
        errors.append("data/branch-preflight.json: experiment_id не совпадает с summary")
    repository_root = value.get("repository_root")
    if not _nonempty_string(repository_root):
        errors.append("data/branch-preflight.json: отсутствует repository_root")
        return
    repository = Path(str(repository_root)).resolve()
    if not repository.is_dir():
        errors.append("data/branch-preflight.json: repository_root не существует")
        return
    current_refs = sorted(list_refs(repository))
    ref_results = value.get("refs")
    if not isinstance(ref_results, list):
        errors.append("data/branch-preflight.json: refs должен быть массивом")
        return
    scanned_refs = [
        str(item.get("ref")) for item in ref_results if isinstance(item, dict) and _nonempty_string(item.get("ref"))
    ]
    if sorted(scanned_refs) != current_refs:
        errors.append("data/branch-preflight.json: ref snapshot устарел — набор refs изменился")
    expected_remote_refs = sorted(ref for ref in current_refs if ref.startswith("refs/remotes/"))
    if value.get("remote_refs_at_scan") != expected_remote_refs:
        errors.append("data/branch-preflight.json: перечень remote refs устарел или неполон")
    raw_terms = value.get("terms")
    if not isinstance(raw_terms, list) or any(not _nonempty_string(term) for term in raw_terms):
        errors.append("data/branch-preflight.json: terms должен быть непустым массивом строк")
        raw_terms = []
    actual_ref_results = scan_refs(repository, current_refs, [str(term) for term in raw_terms])
    actual_by_ref = {
        str(item.get("ref")): item for item in actual_ref_results if isinstance(item, dict)
    }

    active = value.get("active_worktree")
    if not isinstance(active, dict):
        errors.append("data/branch-preflight.json: отсутствует active_worktree")
        active = {}
    active_ref = active.get("ref")
    upstream_ref = active.get("upstream_ref")
    current_head = ref_commit(repository, "HEAD")
    git_repository = current_head is not None
    if active_ref is None and git_repository:
        errors.append(
            "data/branch-preflight.json: detached HEAD запрещён; "
            "явный immutable commit mode не поддерживается"
        )
    exclusions = active.get("binding_excludes")
    if not isinstance(exclusions, list) or any(_safe_relative_path(item) is None for item in exclusions):
        errors.append("data/branch-preflight.json: binding_excludes должен содержать безопасные пути")
        exclusions = []
    current_experiment_path = None
    try:
        experiment_relative = root.relative_to(repository)
    except ValueError:
        errors.append("data/branch-preflight.json: эксперимент находится вне repository_root")
        expected_exclusions: list[str] = []
    else:
        current_experiment_path = experiment_relative.as_posix()
        expected_exclusions = sorted(
            [
                (experiment_relative / "data/branch-preflight.json").as_posix(),
                (experiment_relative / "data/manifest.json").as_posix(),
            ]
        )
    if git_repository and exclusions != expected_exclusions:
        errors.append("data/branch-preflight.json: binding_excludes разрешает только self-reference файлы")
        exclusions = expected_exclusions
    declared_tree = active.get("tree_binding_sha256")
    tree_source = active.get("tree_binding_source")
    if git_repository and tree_source != "git-index-stage0":
        errors.append("data/branch-preflight.json: tree_binding_source должен быть git-index-stage0")
    if not isinstance(declared_tree, str) or not SHA256_RE.fullmatch(declared_tree):
        errors.append("data/branch-preflight.json: некорректный tree_binding_sha256")

    base_commit = active.get("base_commit")
    if git_repository and isinstance(declared_tree, str) and SHA256_RE.fullmatch(declared_tree):
        binding_source = "index" if current_head == base_commit else "HEAD"
        if tree_binding_sha256(repository, exclusions, source=binding_source) != declared_tree:
            errors.append("data/branch-preflight.json: active Git tree binding не совпадает")
        dirty = subprocess.run(
            ["git", "-C", str(repository), "diff-files", "--name-only", "-z"],
            check=False,
            capture_output=True,
        ).stdout
        dirty_paths = {
            value.decode("utf-8", "surrogateescape")
            for value in dirty.split(b"\0")
            if value
        } - set(exclusions)
        if dirty_paths:
            errors.append(
                "data/branch-preflight.json: tracked worktree отличается от Git index: "
                + ", ".join(sorted(dirty_paths))
            )
        if current_head != base_commit and tree_binding_sha256(
            repository, exclusions, source="index"
        ) != tree_binding_sha256(repository, exclusions, source="HEAD"):
            errors.append("data/branch-preflight.json: Git index не совпадает с доказанным HEAD tree")

    unexpected_untracked_roots = [
        path
        for path in untracked_experiment_roots(repository)
        if path != current_experiment_path
    ]
    if unexpected_untracked_roots:
        errors.append(
            "data/branch-preflight.json: обнаружены untracked experiment roots: "
            + ", ".join(unexpected_untracked_roots)
        )

    bound_index_roots = active.get("bound_index_experiment_roots")
    if not isinstance(bound_index_roots, list):
        errors.append(
            "data/branch-preflight.json: отсутствует bound_index_experiment_roots"
        )
        bound_index_roots = []
    actual_index_roots = index_experiment_root_evidence(repository, exclusions)
    if bound_index_roots != actual_index_roots:
        recorded_paths = {
            str(item.get("canonical_path"))
            for item in bound_index_roots
            if isinstance(item, dict)
        }
        actual_paths = {
            str(item.get("canonical_path"))
            for item in actual_index_roots
            if isinstance(item, dict)
        }
        changed_paths = sorted(recorded_paths ^ actual_paths)
        if not changed_paths:
            changed_paths = sorted(actual_paths | recorded_paths)
        errors.append(
            "data/branch-preflight.json: stage-0 index experiment roots/modes/blob OIDs изменились: "
            + ", ".join(changed_paths)
        )

    if active_ref is not None:
        if active_ref not in current_refs or ref_commit(repository, str(active_ref)) != current_head:
            errors.append("data/branch-preflight.json: active HEAD/ref не совпадает")
        if not isinstance(base_commit, str) or not COMMIT_RE.fullmatch(base_commit):
            errors.append("data/branch-preflight.json: некорректный worktree base commit")
        elif ref_commit(repository, base_commit) != base_commit:
            errors.append("data/branch-preflight.json: worktree base commit отсутствует")
        else:
            ancestor = subprocess.run(
                ["git", "-C", str(repository), "merge-base", "--is-ancestor", base_commit, "HEAD"],
                check=False,
                capture_output=True,
            )
            if ancestor.returncode != 0:
                errors.append("data/branch-preflight.json: worktree base commit не предок HEAD")
        ref_result_by_name = {
            item.get("ref"): item for item in ref_results if isinstance(item, dict)
        }
        active_result = ref_result_by_name.get(active_ref)
        if not isinstance(active_result, dict) or active_result.get("commit") != base_commit:
            errors.append("data/branch-preflight.json: active ref scan не связан с worktree base commit")
        if isinstance(upstream_ref, str):
            upstream_base = active.get("upstream_base_commit")
            upstream_result = ref_result_by_name.get(upstream_ref)
            if (
                not isinstance(upstream_base, str)
                or not COMMIT_RE.fullmatch(upstream_base)
                or not isinstance(upstream_result, dict)
                or upstream_result.get("commit") != upstream_base
            ):
                errors.append("data/branch-preflight.json: upstream scan не связан с upstream base commit")
            else:
                current_upstream = ref_commit(repository, upstream_ref)
                if current_upstream not in {upstream_base, current_head}:
                    errors.append("data/branch-preflight.json: upstream ref сдвинут вне base/HEAD")

    snapshot = value.get("ref_snapshot")
    if not isinstance(snapshot, list):
        errors.append("data/branch-preflight.json: отсутствует ref_snapshot")
        snapshot = []
    excluded_refs = {item for item in (active_ref, upstream_ref) if isinstance(item, str)}
    expected_snapshot_refs = set(current_refs) - excluded_refs
    recorded_snapshot_refs = {
        item.get("ref") for item in snapshot if isinstance(item, dict) and _nonempty_string(item.get("ref"))
    }
    if recorded_snapshot_refs != expected_snapshot_refs:
        errors.append("data/branch-preflight.json: ref snapshot не покрывает точный набор refs")
    for item in snapshot:
        if not isinstance(item, dict):
            errors.append("data/branch-preflight.json: ref snapshot содержит не объект")
            continue
        ref = item.get("ref")
        commit = item.get("commit")
        if active_ref is None and ref == "HEAD" and commit is None:
            continue
        if not _nonempty_string(ref) or not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            errors.append("data/branch-preflight.json: ref snapshot содержит неверный ref/commit")
        elif ref_commit(repository, str(ref)) != commit:
            errors.append(f"data/branch-preflight.json: ref snapshot commit изменился: {ref}")

    experiment_id = str(value.get("experiment_id", ""))
    bound_topology = active.get("bound_experiment_roots")
    if not isinstance(bound_topology, list):
        errors.append("data/branch-preflight.json: отсутствует bound_experiment_roots")
        bound_topology = []
    stored_by_ref = {
        str(item.get("ref")): item for item in ref_results if isinstance(item, dict)
    }
    for ref in current_refs:
        actual_result = actual_by_ref.get(ref)
        if not isinstance(actual_result, dict):
            errors.append(f"data/branch-preflight.json: independent ref rescan не прочитал {ref}")
            continue
        actual_topology = experiment_root_topology(actual_result)
        current_commit = ref_commit(repository, ref)
        if ref == active_ref and current_head != base_commit:
            expected_topology = bound_topology
        elif (
            ref == upstream_ref
            and current_head is not None
            and current_commit == current_head
            and current_head != active.get("upstream_base_commit")
        ):
            expected_topology = bound_topology
        else:
            stored_result = stored_by_ref.get(ref)
            expected_topology = (
                experiment_root_topology(stored_result)
                if isinstance(stored_result, dict)
                else []
            )
        if actual_topology != expected_topology:
            errors.append(
                f"data/branch-preflight.json: independent ref rescan обнаружил изменение experiment roots: {ref}"
            )

        if current_experiment_path is not None:
            prefix = current_experiment_path + "-"
            conflicting = [
                str(item.get("canonical_path"))
                for item in actual_topology
                if isinstance(item, dict)
                and isinstance(item.get("canonical_path"), str)
                and item.get("canonical_path") != current_experiment_path
                and str(item.get("canonical_path")).startswith(prefix)
            ]
            if conflicting:
                errors.append(
                    f"data/branch-preflight.json: independent ref rescan нашёл скрытый duplicate root для {experiment_id}: "
                    + ", ".join(sorted(conflicting))
                )

    expected_decision = duplicate_decision_from_refs(ref_results, experiment_id)
    if value.get("duplicate_decision") != expected_decision:
        errors.append("data/branch-preflight.json: duplicate_decision не совпадает с refs — решение подделано")

    actual_decision = duplicate_decision_from_refs(actual_ref_results, experiment_id)
    expected_candidate_keys = {
        (str(item.get("ref")), str(item.get("canonical_path")))
        for item in expected_decision.get("candidates", [])
        if isinstance(item, dict)
    }
    advanced_refs: set[str] = set()
    if isinstance(active_ref, str) and current_head != base_commit:
        advanced_refs.add(active_ref)
    if (
        isinstance(upstream_ref, str)
        and current_head is not None
        and ref_commit(repository, upstream_ref) == current_head
        and current_head != active.get("upstream_base_commit")
    ):
        advanced_refs.add(upstream_ref)
    for advanced_ref in advanced_refs:
        expected_candidate_keys = {
            key for key in expected_candidate_keys if key[0] != advanced_ref
        }
        bound_decision = duplicate_decision_from_refs(
            [{"ref": advanced_ref, "commit": current_head, "experiment_roots": bound_topology}],
            experiment_id,
        )
        expected_candidate_keys.update(
            (str(item.get("ref")), str(item.get("canonical_path")))
            for item in bound_decision.get("candidates", [])
            if isinstance(item, dict)
        )
    actual_candidate_keys = {
        (str(item.get("ref")), str(item.get("canonical_path")))
        for item in actual_decision.get("candidates", [])
        if isinstance(item, dict)
    }
    if actual_candidate_keys != expected_candidate_keys:
        errors.append(
            "data/branch-preflight.json: independent ref rescan дал иное duplicate evidence"
        )


def _validate_trace_ab_links(
    root: Path,
    summary: dict[str, Any],
    entries_by_path: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    try:
        schema = json.loads(TRACE_AB_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"trace A/B schema: не удалось загрузить: {exc}")
        return
    runs = {
        str(run.get("run_id")): run
        for run in summary.get("runs", [])
        if isinstance(run, dict) and _nonempty_string(run.get("run_id"))
    }
    links = summary.get("trace_ab_results")
    if not isinstance(links, list):
        return
    represented_runs: set[str] = set()
    for link_index, link in enumerate(links):
        prefix = f"trace A/B link[{link_index}]"
        if not isinstance(link, dict):
            errors.append(f"{prefix}: нужен объект")
            continue
        relative = _safe_relative_path(link.get("path"))
        if relative is None:
            errors.append(f"{prefix}: небезопасный path")
            continue
        result_path = root / relative
        if not result_path.is_file():
            errors.append(f"{prefix}: файл отсутствует: {relative}")
            continue
        actual_result_sha = sha256_file(result_path)
        if link.get("sha256") != actual_result_sha:
            errors.append(f"{prefix}: sha256 результата не совпадает")
        manifest_entry = entries_by_path.get(relative)
        if not isinstance(manifest_entry, dict):
            errors.append(f"{prefix}: результат не включён в manifest")
        elif manifest_entry.get("sha256") != actual_result_sha:
            errors.append(f"{prefix}: hash результата в manifest не совпадает")
        result = _read_json(result_path, errors)
        if result is None:
            continue
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        for schema_error in sorted(validator.iter_errors(result), key=lambda item: list(item.path)):
            location = ".".join(str(part) for part in schema_error.path) or "$"
            errors.append(f"{prefix} schema {location}: {schema_error.message}")
        for invariant_error in validate_trace_ab_result(result):
            errors.append(f"{prefix}: {invariant_error}")
        identity = result.get("workload_identity")
        if not isinstance(identity, dict):
            continue
        for side_name, mode in (("trace_off", "off"), ("trace_on", "on")):
            side = result.get(side_name)
            if not isinstance(side, dict) or not isinstance(side.get("samples"), list):
                continue
            for sample_index, sample in enumerate(side["samples"]):
                sample_prefix = f"{prefix} {side_name}.samples[{sample_index}]"
                if not isinstance(sample, dict):
                    continue
                run_id = sample.get("run_id")
                run = runs.get(str(run_id))
                if run is None:
                    errors.append(f"{sample_prefix}: trace A/B run_id не найден в summary")
                    continue
                represented_runs.add(str(run_id))
                binding = run.get("trace_ab")
                if not isinstance(binding, dict):
                    errors.append(f"{sample_prefix}: у run отсутствует trace_ab binding")
                elif (
                    binding.get("result_path") != relative
                    or binding.get("pair_id") != sample.get("pair_id")
                    or binding.get("mode") != mode
                ):
                    errors.append(f"{sample_prefix}: trace A/B run/pair/mode link не совпадает")
                workload = run.get("workload")
                expected_identity = {
                    "variant_id": run.get("variant_id"),
                    "common_prompt_sha256": workload.get("common_prompt_sha256") if isinstance(workload, dict) else None,
                    "rendered_prompt_sha256": workload.get("rendered_prompt_sha256") if isinstance(workload, dict) else None,
                    "prompt_token_ids_sha256": workload.get("prompt_token_ids_sha256") if isinstance(workload, dict) else None,
                    "n_predict": workload.get("n_predict") if isinstance(workload, dict) else None,
                    "seed": workload.get("seed") if isinstance(workload, dict) else None,
                }
                for field, expected in expected_identity.items():
                    if identity.get(field) != expected:
                        errors.append(f"{sample_prefix}: workload identity {field} не совпадает с run")
                raw_artifacts = sample.get("raw_artifacts")
                if not isinstance(raw_artifacts, list):
                    continue
                for raw_value in raw_artifacts:
                    raw_relative = _safe_relative_path(raw_value)
                    if raw_relative is None or not raw_relative.startswith("raw/"):
                        errors.append(f"{sample_prefix}: trace A/B raw path небезопасен")
                        continue
                    raw_path = root / raw_relative
                    if not raw_path.is_file():
                        errors.append(f"{sample_prefix}: trace A/B raw отсутствует: {raw_relative}")
                        continue
                    raw_entry = entries_by_path.get(raw_relative)
                    if not isinstance(raw_entry, dict):
                        errors.append(f"{sample_prefix}: trace A/B raw не включён в manifest: {raw_relative}")
                        continue
                    capture = raw_entry.get("capture")
                    if not isinstance(capture, dict) or capture.get("captured") is not True:
                        errors.append(f"{sample_prefix}: trace A/B raw не имеет capture.captured=true")
                    if raw_entry.get("sha256") != sha256_file(raw_path):
                        errors.append(f"{sample_prefix}: trace A/B raw hash не совпадает: {raw_relative}")
    bound_runs = {
        run_id
        for run_id, run in runs.items()
        if isinstance(run.get("trace_ab"), dict)
        and run["trace_ab"].get("result_path") in {
            link.get("path")
            for link in links
            if isinstance(link, dict) and isinstance(link.get("path"), str)
        }
    }
    if represented_runs != bound_runs:
        errors.append("trace A/B: набор sample run_id не совпадает с привязанными summary runs")


def validate_experiment(experiment_root: Path | str) -> dict[str, Any]:
    """Return a machine-readable validation result for an experiment directory."""

    root = Path(experiment_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return {"valid": False, "status": None, "errors": [f"нет каталога эксперимента: {root}"], "warnings": []}

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"отсутствует обязательный файл: {relative}")

    readme = root / "README.md"
    if readme.is_file():
        try:
            readme_text = readme.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"README.md: не удалось прочитать: {exc}")
        else:
            if not CYRILLIC_RE.search(readme_text):
                errors.append("README.md: документация должна содержать русский текст")

    summary: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    if (root / "results/summary.json").is_file():
        summary = _read_json(root / "results/summary.json", errors)
    if (root / "data/manifest.json").is_file():
        manifest = _read_json(root / "data/manifest.json", errors)

    status = summary.get("status") if summary else None
    if summary is not None:
        _validate_summary_schema(summary, errors)
    if status not in VALID_STATUSES:
        errors.append(
            "results/summary.json: status должен быть одним из "
            + ", ".join(sorted(VALID_STATUSES))
        )
        status = None

    for relative in ("environment.json", "device.json"):
        path = root / relative
        if path.is_file():
            provenance = _read_json(path, errors)
            if provenance is not None:
                _validate_capture_provenance(path, provenance, status, errors)
    for relative in ("commands.txt", "hypothesis-preflight.md"):
        path = root / relative
        if path.is_file():
            try:
                if not path.read_text(encoding="utf-8").strip():
                    errors.append(f"{relative}: файл не должен быть пустым")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"{relative}: не удалось прочитать: {exc}")

    preflight = None
    if (root / "data/branch-preflight.json").is_file():
        preflight = _read_json(root / "data/branch-preflight.json", errors)
        if preflight is not None:
            _validate_preflight(root, preflight, summary, errors)

    entries_by_path: dict[str, dict[str, Any]] = {}
    if status is not None and manifest is not None:
        entries_by_path = _validate_manifest(root, manifest, status, errors)

    if status in EXECUTED_STATUSES:
        commands_text = ""
        try:
            commands_text = (root / "commands.txt").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        run_ids = _validate_run_bindings(summary or {}, commands_text, errors)
        _validate_trace_ab_links(root, summary or {}, entries_by_path, errors)
        for relative in REQUIRED_RAW_FILES:
            raw_path = root / relative
            if not raw_path.is_file():
                errors.append(f"для статуса {status} отсутствует обязательный raw-файл: {relative}")
                continue
            entry = entries_by_path.get(relative)
            capture = entry.get("capture") if isinstance(entry, dict) else None
            if not isinstance(capture, dict) or capture.get("captured") is not True:
                errors.append(f"data/manifest.json: {relative} требует capture.captured=true")
            if raw_path.stat().st_size == 0:
                empty_reason = capture.get("empty_reason") if isinstance(capture, dict) else None
                if relative in {"raw/telemetry.jsonl", "raw/trace.jsonl"}:
                    errors.append(f"{relative}: JSONL не может быть пустым")
                elif not _nonempty_string(empty_reason):
                    errors.append(f"data/manifest.json: пустой {relative} требует capture.empty_reason")
        telemetry_events = _read_jsonl(root / "raw/telemetry.jsonl", errors) if (root / "raw/telemetry.jsonl").is_file() else []
        trace_events = _read_jsonl(root / "raw/trace.jsonl", errors) if (root / "raw/trace.jsonl").is_file() else []
        _validate_telemetry_events(telemetry_events, run_ids, errors)
        _validate_trace_events(trace_events, run_ids, errors)
        _validate_cross_stream_ddr(
            summary or {}, telemetry_events, trace_events, errors
        )

    if manifest is not None and manifest.get("status") == "planned" and status == "planned":
        warnings.append("эксперимент только запланирован; сырые результаты ещё не требуются")

    return {
        "valid": not errors,
        "status": status,
        "experiment_root": str(root),
        "errors": errors,
        "warnings": warnings,
    }


def build_file_manifest(root: Path | str, paths: Iterable[str], *, status: str) -> dict[str, Any]:
    """Create the hash portion of an E053 manifest for existing relative files."""

    root = Path(root).resolve()
    entries = []
    for relative in sorted(set(paths)):
        safe = _safe_relative_path(relative)
        if safe is None:
            raise ValueError(f"небезопасный относительный путь: {relative!r}")
        path = root / safe
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"путь выходит за пределы эксперимента: {relative!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": safe, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": status,
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="вывести только JSON")
    args = parser.parse_args()
    result = validate_experiment(args.experiment)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
