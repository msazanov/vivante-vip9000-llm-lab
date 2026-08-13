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
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker

try:
    from tooling.branch_preflight import list_remote_refs
except ModuleNotFoundError:  # direct: python3 tooling/experiment_validator.py
    from branch_preflight import list_remote_refs


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
    "token_index",
    "layer_index",
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
MEMORY_PROVENANCE_REQUIRED = {
    "measurement_method",
    "measurement_source",
    "measurement_confidence",
}


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
        "logical": {"logical_bytes"},
        "unique_weights": {"unique_weight_bytes"},
        "observed_direct_ddr": {
            "observed_direct_ddr_read_bytes",
            "observed_direct_ddr_write_bytes",
        },
        "inferred_ddr": {"inferred_ddr_read_bytes", "inferred_ddr_write_bytes"},
    }
    for group, byte_fields in groups.items():
        item = value.get(group)
        if not isinstance(item, dict):
            continue
        allowed = byte_fields | MEMORY_PROVENANCE_REQUIRED
        unexpected = set(item) - allowed
        if unexpected:
            errors.append(
                f"{prefix}.{group}: смешивает классы измерений: {', '.join(sorted(unexpected))}"
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
    if allowed_methods is not None and value.get("measurement_method") not in allowed_methods:
        errors.append(f"{prefix}: measurement_method смешивает классы измерений")


def _validate_telemetry_events(events: list[dict[str, Any]], run_ids: set[str], errors: list[str]) -> None:
    for index, event in enumerate(events, start=1):
        prefix = f"telemetry.jsonl:{index}"
        missing = TELEMETRY_REQUIRED - event.keys()
        if missing:
            errors.append(f"{prefix}: отсутствуют {', '.join(sorted(missing))}")
            continue
        if event.get("schema_version") != "e047-telemetry-event/v1":
            errors.append(f"{prefix}: неверный schema_version")
        if event.get("run_id") not in run_ids:
            errors.append(f"{prefix}: неизвестный run_id")
        if not isinstance(event.get("timestamp_ns"), int) or event["timestamp_ns"] < 0:
            errors.append(f"{prefix}: timestamp_ns должен быть >= 0")
        if not isinstance(event.get("cpu_frequency_hz"), list) or not event["cpu_frequency_hz"]:
            errors.append(f"{prefix}: cpu_frequency_hz должен быть непустым массивом")
        _validate_memory_provenance(
            event.get("ddr"),
            f"{prefix}.ddr",
            errors,
            allowed_methods={"hardware_counter", "pmu_counter_delta", "uncore_counter_delta"},
        )


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
            {"hardware_counter", "pmu_counter_delta", "uncore_counter_delta"},
        ),
        "inferred_ddr": (
            {"inferred_ddr_read_bytes", "inferred_ddr_write_bytes"},
            {"analytical_estimate", "counter_model_fit", "simulation"},
        ),
    }
    for group, (byte_fields, methods) in groups.items():
        item = value.get(group)
        _validate_memory_provenance(
            item,
            f"{prefix}.{group}",
            errors,
            allowed_methods=methods,
        )
        if isinstance(item, dict):
            missing = byte_fields - item.keys()
            if missing:
                errors.append(f"{prefix}.{group}: отсутствуют {', '.join(sorted(missing))}")
            unexpected = set(item) - (byte_fields | MEMORY_PROVENANCE_REQUIRED)
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
        if event.get("schema_version") != "e047-trace-event/v1":
            errors.append(f"{prefix}: неверный schema_version")
        if event.get("run_id") not in run_ids:
            errors.append(f"{prefix}: неизвестный run_id")
        for field in ("token_index", "layer_index", "op_index", "start_ns", "end_ns", "duration_ns"):
            if not isinstance(event.get(field), int) or event[field] < 0:
                errors.append(f"{prefix}: {field} должен быть integer >= 0")
        if all(isinstance(event.get(field), int) for field in ("start_ns", "end_ns", "duration_ns")):
            if event["end_ns"] - event["start_ns"] != event["duration_ns"]:
                errors.append(f"{prefix}: duration_ns != end_ns-start_ns")
        if not isinstance(event.get("inputs"), list) or not isinstance(event.get("outputs"), list):
            errors.append(f"{prefix}: inputs/outputs должны быть массивами")
        _validate_trace_memory(event.get("memory"), f"{prefix}.memory", errors)


def _validate_preflight(root: Path, value: dict[str, Any], summary: dict[str, Any] | None, errors: list[str]) -> None:
    if value.get("schema_version") != "e053-branch-preflight/v2":
        errors.append("data/branch-preflight.json: нужен schema_version e053-branch-preflight/v2")
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
    expected_remote_refs = list_remote_refs(Path(str(repository_root)))
    recorded = value.get("remote_refs_at_scan")
    if recorded != expected_remote_refs:
        errors.append("data/branch-preflight.json: перечень remote refs устарел или неполон")
    scanned_refs = {
        item.get("ref") for item in value.get("refs", []) if isinstance(item, dict)
    }
    missing = set(expected_remote_refs) - scanned_refs
    if missing:
        errors.append(
            "data/branch-preflight.json: не просканированы remote refs: "
            + ", ".join(sorted(missing))
        )
    decision = value.get("duplicate_decision")
    if not isinstance(decision, dict) or decision.get("status") not in {
        "duplicate_found",
        "no_duplicate",
    }:
        errors.append("data/branch-preflight.json: отсутствует duplicate_decision")
    elif decision.get("status") == "duplicate_found":
        candidates = decision.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append("data/branch-preflight.json: duplicate_found без candidates")
        else:
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    errors.append(f"data/branch-preflight.json: candidate[{index}] не объект")
                    continue
                for field in ("experiment_id", "path", "ref", "commit"):
                    if not _nonempty_string(candidate.get(field)):
                        errors.append(f"data/branch-preflight.json: candidate[{index}] без {field}")
                if isinstance(candidate.get("commit"), str) and not COMMIT_RE.fullmatch(
                    candidate["commit"]
                ):
                    errors.append(f"data/branch-preflight.json: candidate[{index}] commit неверен")


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
