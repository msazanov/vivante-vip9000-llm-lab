#!/usr/bin/env python3
"""Validate and summarize the Q1 CPU operator evidence bundle.

The runner emits a deliberately small JSONL protocol.  This module is strict
about that protocol and about the result contract: malformed or incomplete
evidence is an error, never an ``unqualified`` success.  Correctness is
checked before any timing qualification is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = "q1-cpu-operator-baseline/v1"
RUN_SCHEMA_VERSION = "q1-cpu-operator-run/v1"
EXPECTED_MODEL_ID = "prism-ml/Ternary-Bonsai-27B-gguf"
EXPECTED_MODEL_SIZE = 3803452480
EXPECTED_MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
PRODUCTION_TENSORS = {
    ("blk.0.ffn_gate.weight", (5120, 17408)): "0f42ca3b81099f540ed67941809ee7fdc0a672563bf852b87db75034135fc6fe",
    ("blk.0.ffn_down.weight", (17408, 5120)): "ab3ef165a5940b14ff1279843af15cbde0fa6cffc2e8eb063be70f7541f0fd04",
}
MEASURED_SAMPLES = 50
CV_LIMIT = 0.02
COSINE_MIN = 0.999999
IDENTITY_PATTERN = re.compile(r"^[0-9]+:[0-9]+$")


class OperatorSummaryError(ValueError):
    """Evidence is missing, malformed, or fails a mandatory gate."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OperatorSummaryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_load(text: str, context: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as error:
        raise OperatorSummaryError(f"{context}: invalid JSON") from error


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    """Publish JSON without overwriting a race winner or leaving partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise OperatorSummaryError(f"refusing to overwrite output: {path}") from error
        os.unlink(temporary)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OperatorSummaryError(f"{context}: expected object")
    return value


def _string(value: Any, context: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise OperatorSummaryError(f"{context}: expected string")
    return value


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise OperatorSummaryError(f"{context}: expected boolean")
    return value


def _integer(value: Any, context: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise OperatorSummaryError(f"{context}: expected integer")
    return value


def _number(value: Any, context: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorSummaryError(f"{context}: expected number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise OperatorSummaryError(f"{context}: expected finite number")
    return result


def _sha(value: Any, context: str) -> str:
    result = _string(value, context).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise OperatorSummaryError(f"{context}: expected lowercase SHA-256 hex")
    return result


def _exact(row: Any, expected: set[str], context: str) -> None:
    if not isinstance(row, dict):
        raise OperatorSummaryError(f"{context}: expected object")
    if any(not isinstance(key, str) for key in row):
        raise OperatorSummaryError(f"{context}: object keys must be strings")
    if set(row) != expected:
        raise OperatorSummaryError(
            f"{context}: fields differ: expected={sorted(expected)} got={sorted(row)}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise OperatorSummaryError(f"cannot read runner JSONL: {path}") from error
    if not lines:
        raise OperatorSummaryError("runner JSONL is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        value = _json_load(line, f"runner JSONL line {line_number}")
        rows.append(_mapping(value, f"runner JSONL line {line_number}"))
    return rows


def _read_f32(path: Path, context: str) -> tuple[list[float], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise OperatorSummaryError(f"cannot read {context}: {path}") from error
    if len(payload) == 0 or len(payload) % 4:
        raise OperatorSummaryError(f"{context}: byte length must be a nonzero multiple of 4")
    values = list(struct.unpack(f"<{len(payload) // 4}f", payload))
    if not all(math.isfinite(value) for value in values):
        raise OperatorSummaryError(f"{context}: contains non-finite F32 element")
    return values, payload


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise OperatorSummaryError("timing: no measured samples")
    mean = sum(values) / len(values)
    population_stddev = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "mean": mean,
        "population_cv": population_stddev / mean if mean else float("inf"),
    }


def _validate_sidecar(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        raise OperatorSummaryError("thermal guard trace is empty")
    def event_row(row: dict[str, Any], event: str, fields: set[str]) -> None:
        _exact(row, fields, f"thermal.{event}")
        if _string(row["event"], f"thermal.{event}.event") != event:
            raise OperatorSummaryError("thermal guard lifecycle is incomplete or reordered")
        _integer(row["monotonic_ns"], f"thermal.{event}.monotonic_ns", nonnegative=True)

    def text(value: Any, context: str) -> str:
        return _string(value, context)

    def path_value(value: Any, context: str) -> str:
        value = _string(value, context)
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
            raise OperatorSummaryError(f"{context}: expected normalized absolute path")
        return value

    def identity_value(value: Any, context: str) -> str:
        value = _string(value, context)
        if IDENTITY_PATTERN.fullmatch(value) is None:
            raise OperatorSummaryError(f"{context}: expected st_dev:st_ino decimal identity")
        device, inode = value.split(":")
        if ((len(device) > 1 and device.startswith("0"))
                or (len(inode) > 1 and inode.startswith("0"))):
            raise OperatorSummaryError(f"{context}: expected canonical decimal identity")
        return value

    def inventory_zone(value: Any, context: str) -> dict[str, Any]:
        row = _mapping(value, context)
        _exact(row, {"path", "type", "identity"}, context)
        path_value(row["path"], f"{context}.path")
        text(row["type"], f"{context}.type")
        identity_value(row["identity"], f"{context}.identity")
        return row

    def sample_zone(value: Any, context: str) -> dict[str, Any]:
        row = _mapping(value, context)
        _exact(row, {"path", "type", "millidegrees_c"}, context)
        path_value(row["path"], f"{context}.path")
        text(row["type"], f"{context}.type")
        _integer(row["millidegrees_c"], f"{context}.millidegrees_c")
        return row

    def inventory_policy(value: Any, context: str) -> dict[str, Any]:
        row = _mapping(value, context)
        _exact(row, {"path", "cur_khz", "max_khz", "governor", "affected_cpus", "identity"}, context)
        path_value(row["path"], f"{context}.path")
        _integer(row["cur_khz"], f"{context}.cur_khz")
        _integer(row["max_khz"], f"{context}.max_khz")
        text(row["governor"], f"{context}.governor")
        if row["affected_cpus"] is not None:
            text(row["affected_cpus"], f"{context}.affected_cpus")
        identity_value(row["identity"], f"{context}.identity")
        return row

    def sample_policy(value: Any, context: str) -> dict[str, Any]:
        row = inventory_policy(value, context)
        return row

    def inventory_cooling(value: Any, context: str) -> dict[str, Any]:
        row = _mapping(value, context)
        _exact(row, {"path", "type", "cur_state", "max_state", "identity"}, context)
        path_value(row["path"], f"{context}.path")
        text(row["type"], f"{context}.type")
        _integer(row["cur_state"], f"{context}.cur_state")
        _integer(row["max_state"], f"{context}.max_state")
        identity_value(row["identity"], f"{context}.identity")
        return row

    def sample_cooling(value: Any, context: str) -> dict[str, Any]:
        return inventory_cooling(value, context)

    index = 0
    start_fields = {"event", "monotonic_ns", "command", "limit_mc", "interval_ms"}
    if index >= len(rows):
        raise OperatorSummaryError("thermal guard lifecycle is incomplete or reordered")
    event_row(rows[index], "start", start_fields)
    if (type(rows[index]["command"]) is not list or not rows[index]["command"]
            or not isinstance(rows[index]["command"][0], str)
            or not rows[index]["command"][0]
            or any(not isinstance(item, str) for item in rows[index]["command"])):
        raise OperatorSummaryError("thermal.start.command must be a string list")
    _integer(rows[index]["limit_mc"], "thermal.start.limit_mc", nonnegative=True)
    _integer(rows[index]["interval_ms"], "thermal.start.interval_ms", nonnegative=True)
    if rows[index]["limit_mc"] <= 0 or rows[index]["interval_ms"] <= 0:
        raise OperatorSummaryError("thermal.start limits must be positive")
    interval_ns = rows[index]["interval_ms"] * 1_000_000
    previous_monotonic_ns = rows[index]["monotonic_ns"]
    index += 1
    if index >= len(rows):
        raise OperatorSummaryError("thermal guard lifecycle is incomplete or reordered")
    event_row(rows[index], "inventory", {"event", "monotonic_ns", "thermal_zones", "cpu_policies", "cooling_devices"})
    for key in ("thermal_zones", "cpu_policies", "cooling_devices"):
        if type(rows[index][key]) is not list:
            raise OperatorSummaryError(f"thermal.inventory.{key} must be an array")
        if not rows[index][key]:
            raise OperatorSummaryError(f"thermal.inventory.{key} must be nonempty")
    inventory_zones = [inventory_zone(value, f"thermal.inventory.thermal_zones[{n}]")
                      for n, value in enumerate(rows[index]["thermal_zones"])]
    inventory_policies = [inventory_policy(value, f"thermal.inventory.cpu_policies[{n}]")
                          for n, value in enumerate(rows[index]["cpu_policies"])]
    inventory_cooling_devices = [inventory_cooling(value, f"thermal.inventory.cooling_devices[{n}]")
                                 for n, value in enumerate(rows[index]["cooling_devices"])]
    for category, values in (("zones", inventory_zones), ("policies", inventory_policies),
                             ("cooling", inventory_cooling_devices)):
        paths = [value["path"] for value in values]
        identities = [value["identity"] for value in values]
        if len(paths) != len(set(paths)) or len(identities) != len(set(identities)):
            raise OperatorSummaryError(f"thermal.inventory.{category} paths/identities must be unique")
    all_paths = [row["path"] for row in inventory_zones + inventory_policies + inventory_cooling_devices]
    all_identities = [row["identity"] for row in inventory_zones + inventory_policies + inventory_cooling_devices]
    if len(all_paths) != len(set(all_paths)) or len(all_identities) != len(set(all_identities)):
        raise OperatorSummaryError("thermal inventory paths/identities must be globally unique")
    zone_by_path = {row["path"]: row for row in inventory_zones}
    policy_by_path = {row["path"]: row for row in inventory_policies}
    cooling_by_path = {row["path"]: row for row in inventory_cooling_devices}
    if rows[index]["monotonic_ns"] < previous_monotonic_ns:
        raise OperatorSummaryError("thermal lifecycle timestamps are not monotonic")
    previous_monotonic_ns = rows[index]["monotonic_ns"]
    index += 1
    if index >= len(rows):
        raise OperatorSummaryError("thermal guard lifecycle is incomplete or reordered")
    event_row(rows[index], "child_started", {"event", "monotonic_ns", "pid"})
    _integer(rows[index]["pid"], "thermal.child_started.pid", nonnegative=True)
    if rows[index]["pid"] <= 0:
        raise OperatorSummaryError("thermal.child_started.pid must be positive")
    child_pid = rows[index]["pid"]
    child_started_monotonic_ns = rows[index]["monotonic_ns"]
    if rows[index]["monotonic_ns"] < previous_monotonic_ns:
        raise OperatorSummaryError("thermal lifecycle timestamps are not monotonic")
    previous_monotonic_ns = rows[index]["monotonic_ns"]
    index += 1
    sample_count = 0
    previous_deadline_ns: int | None = None
    while index < len(rows) and rows[index].get("kind") == "sample":
        row = rows[index]
        _exact(row, {"kind", "sample_seq", "monotonic_ns", "scheduled_deadline_ns", "lateness_ns",
                     "process", "thermal_zones", "cpu_policies", "cooling_devices"}, "thermal.sample")
        if "event" in row:
            raise OperatorSummaryError("thermal guard sample has an unexpected event field")
        _integer(row["sample_seq"], "thermal.sample.sample_seq", nonnegative=True)
        if row["sample_seq"] != sample_count:
            raise OperatorSummaryError("thermal guard sample sequence is not contiguous")
        if _string(row["kind"], "thermal.sample.kind") != "sample":
            raise OperatorSummaryError("thermal.sample.kind must be sample")
        for key in ("monotonic_ns", "scheduled_deadline_ns", "lateness_ns"):
            _integer(row[key], f"thermal.sample.{key}", nonnegative=True)
        if row["monotonic_ns"] < previous_monotonic_ns:
            raise OperatorSummaryError("thermal sample timestamps are not monotonic")
        if row["lateness_ns"] != max(0, row["monotonic_ns"] - row["scheduled_deadline_ns"]):
            raise OperatorSummaryError("thermal sample lateness does not match timestamps")
        if sample_count == 0 and row["scheduled_deadline_ns"] < child_started_monotonic_ns:
            raise OperatorSummaryError("thermal first sample deadline precedes child start")
        if previous_deadline_ns is not None and row["scheduled_deadline_ns"] != previous_deadline_ns + interval_ns:
            raise OperatorSummaryError("thermal sample deadlines are not consistently spaced")
        previous_deadline_ns = row["scheduled_deadline_ns"]
        previous_monotonic_ns = row["monotonic_ns"]
        if type(row["thermal_zones"]) is not list or type(row["cpu_policies"]) is not list \
                or type(row["cooling_devices"]) is not list:
            raise OperatorSummaryError("thermal.sample arrays must be arrays")
        if not row["thermal_zones"] or not row["cpu_policies"] or not row["cooling_devices"]:
            raise OperatorSummaryError("thermal.sample arrays must be nonempty")
        sample_zones = [sample_zone(value, f"thermal.sample.thermal_zones[{n}]")
                        for n, value in enumerate(row["thermal_zones"])]
        if [value["path"] for value in sample_zones] != [value["path"] for value in inventory_zones]:
            raise OperatorSummaryError("thermal sample zone cardinality/order does not match inventory")
        for sample in sample_zones:
            baseline = zone_by_path.get(sample["path"])
            if baseline is None or sample["type"] != baseline["type"]:
                raise OperatorSummaryError("thermal sample zone does not match inventory")
            if sample["millidegrees_c"] >= rows[0]["limit_mc"]:
                raise OperatorSummaryError("thermal sample reached the guard limit")
        sample_policies = [sample_policy(value, f"thermal.sample.cpu_policies[{n}]")
                           for n, value in enumerate(row["cpu_policies"])]
        if [value["path"] for value in sample_policies] != [value["path"] for value in inventory_policies]:
            raise OperatorSummaryError("thermal sample policy cardinality/order does not match inventory")
        for sample in sample_policies:
            baseline = policy_by_path.get(sample["path"])
            if (baseline is None or sample["identity"] != baseline["identity"]
                    or sample["affected_cpus"] != baseline["affected_cpus"]):
                raise OperatorSummaryError("thermal sample policy does not match inventory")
        sample_cooling_devices = [sample_cooling(value, f"thermal.sample.cooling_devices[{n}]")
                                  for n, value in enumerate(row["cooling_devices"])]
        if [value["path"] for value in sample_cooling_devices] != [value["path"] for value in inventory_cooling_devices]:
            raise OperatorSummaryError("thermal sample cooling cardinality/order does not match inventory")
        for sample in sample_cooling_devices:
            baseline = cooling_by_path.get(sample["path"])
            if (baseline is None or sample["type"] != baseline["type"]
                    or sample["identity"] != baseline["identity"]
                    or sample["max_state"] != baseline["max_state"]):
                raise OperatorSummaryError("thermal sample cooling device does not match inventory")
        process = _mapping(row["process"], "thermal.sample.process")
        process_fields = {"pid", "available", "rss_kib", "rss_hwm_kib", "threads",
                          "voluntary_context_switches", "nonvoluntary_context_switches",
                          "user_ticks", "system_ticks", "io_rchar", "io_wchar",
                          "io_read_bytes", "io_write_bytes"}
        if not set(process).issubset(process_fields):
            raise OperatorSummaryError("thermal.sample.process has unknown fields")
        _integer(process.get("pid"), "thermal.sample.process.pid", nonnegative=True)
        _bool(process.get("available"), "thermal.sample.process.available")
        if process["pid"] != child_pid:
            raise OperatorSummaryError("thermal sample process pid does not match child_started")
        for key, value in process.items():
            if key not in {"pid", "available"}:
                _integer(value, f"thermal.sample.process.{key}", nonnegative=True)
        sample_count += 1
        index += 1
    if sample_count == 0:
        raise OperatorSummaryError("thermal guard produced no samples")
    if index + 2 != len(rows):
        raise OperatorSummaryError("thermal guard lifecycle has unknown/reordered events")
    event_row(rows[index], "child_exit", {"event", "monotonic_ns", "returncode"})
    event_row(rows[index + 1], "exit", {"event", "monotonic_ns", "status"})
    if rows[index]["monotonic_ns"] < previous_monotonic_ns \
            or rows[index + 1]["monotonic_ns"] < rows[index]["monotonic_ns"]:
        raise OperatorSummaryError("thermal lifecycle timestamps are not monotonic")
    _integer(rows[index]["returncode"], "thermal.child_exit.returncode")
    _integer(rows[index + 1]["status"], "thermal.exit.status")
    if rows[index]["returncode"] != 0 or rows[index + 1]["status"] != 0:
        raise OperatorSummaryError("thermal guard child did not exit successfully")
    return {"thermal_guard_samples": sample_count}


def _read_telemetry(path: Path) -> dict[str, Any]:
    try:
        value = _json_load(path.read_text(encoding="utf-8"), "telemetry")
    except (OSError, UnicodeError) as error:
        raise OperatorSummaryError(f"cannot read telemetry JSON: {path}") from error
    telemetry = _mapping(value, "telemetry")
    expected = {"schema_version", "swap_delta_kib", "thermal_guard_passed", "throttle_detected",
                "oom_detected", "fault_detected"}
    _exact(telemetry, expected, "telemetry")
    if telemetry["schema_version"] != "q1-cpu-operator-telemetry/v1":
        raise OperatorSummaryError("telemetry.schema_version: unsupported version")
    return telemetry


def _parse_bundle(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any],
                                                        list[dict[str, Any]], dict[str, Any]]:
    records = [_string(row.get("record"), f"runner record {index + 1}.record")
               for index, row in enumerate(rows)]
    if records.count("identity") != 1 or records.count("memory") != 1 or records.count("result") != 1:
        raise OperatorSummaryError("runner JSONL requires exactly one identity, memory, and result")
    if any(record not in {"identity", "memory", "sample", "result"} for record in records):
        raise OperatorSummaryError("runner JSONL contains an unknown record")
    expected_records = ["identity", "memory"] + ["sample"] * (MEASURED_SAMPLES + 1) + ["result"]
    if records != expected_records:
        raise OperatorSummaryError("runner JSONL order must be identity, memory, samples, result")
    identity = next(row for row in rows if row["record"] == "identity")
    memory = next(row for row in rows if row["record"] == "memory")
    result = next(row for row in rows if row["record"] == "result")
    identity_fields = {"schema_version", "record", "run_id", "model", "tensor", "layout",
                       "workload", "weight_buffer_type", "backend", "kernel", "strict_mode",
                       "threads", "logical_gemv_count"}
    memory_fields = {"schema_version", "record", "declared_canonical_q1_bytes",
                     "declared_cpu_repack_bytes", "expanded_weight_ddr_bytes",
                     "activation_f32_bytes", "activation_q8_bytes", "output_f32_bytes",
                     "observed_ddr_read_bytes", "observed_ddr_write_bytes",
                     "physical_resident_weight_copies", "cpu_fallback_count"}
    result_fields = {"schema_version", "record", "output_f32_bytes"}
    for row, expected, name in ((identity, identity_fields, "identity"),
                                (memory, memory_fields, "memory"),
                                (result, result_fields, "result")):
        _exact(row, expected, name)
        if row["schema_version"] != RUN_SCHEMA_VERSION or row["record"] != name:
            raise OperatorSummaryError(f"{name}: unsupported schema_version or record")

    samples = [row for row in rows if row.get("record") == "sample"]
    for row in samples:
        _exact(row, {"schema_version", "record", "kind", "iteration", "host_us", "compute_us"},
               "sample")
        if row["schema_version"] != RUN_SCHEMA_VERSION or row["record"] != "sample":
            raise OperatorSummaryError("sample: unsupported schema_version or record")
        kind = _string(row["kind"], "sample.kind")
        if kind not in {"warmup", "measured"}:
            raise OperatorSummaryError("sample.kind: expected warmup or measured")
        _integer(row["iteration"], "sample.iteration", nonnegative=True)
        _number(row["host_us"], "sample.host_us", nonnegative=True)
        _number(row["compute_us"], "sample.compute_us", nonnegative=True)
    if len(samples) != 1 + MEASURED_SAMPLES:
        raise OperatorSummaryError(f"expected one warmup and {MEASURED_SAMPLES} measured samples")
    warmups = [row for row in samples if row["kind"] == "warmup"]
    measured = [row for row in samples if row["kind"] == "measured"]
    if len(warmups) != 1 or warmups[0]["iteration"] != 0:
        raise OperatorSummaryError("warmup samples must contain exactly iteration 0")
    if samples[0]["kind"] != "warmup" or any(row["kind"] != "measured" for row in samples[1:]):
        raise OperatorSummaryError("runner JSONL order must place warmup before measured samples")
    if [row["iteration"] for row in measured] != list(range(MEASURED_SAMPLES)):
        raise OperatorSummaryError("measured sample indices must be contiguous and unique")
    return identity, memory, measured, result


def _validate_metadata(identity: dict[str, Any], memory: dict[str, Any],
                       telemetry: dict[str, Any], result: dict[str, Any],
                       output_bytes: int) -> None:
    model = _mapping(identity["model"], "identity.model")
    _exact(model, {"id", "sha256", "size_bytes", "quantization"}, "identity.model")
    _sha(model["sha256"], "identity.model.sha256")
    if model["sha256"] != EXPECTED_MODEL_SHA256:
        raise OperatorSummaryError("identity.model.sha256: unexpected pinned model hash")
    if model["id"] != EXPECTED_MODEL_ID or model["size_bytes"] != EXPECTED_MODEL_SIZE:
        raise OperatorSummaryError("identity.model: unexpected pinned model identity")
    _integer(model["size_bytes"], "identity.model.size_bytes", nonnegative=True)
    if model["quantization"] != "Q1_0":
        raise OperatorSummaryError("identity.model.quantization must be Q1_0")

    tensor = _mapping(identity["tensor"], "identity.tensor")
    _exact(tensor, {"name", "sha256", "shape", "packed_bytes"}, "identity.tensor")
    _string(tensor["name"], "identity.tensor.name")
    _sha(tensor["sha256"], "identity.tensor.sha256")
    shape = tensor["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(item) is not int or item <= 0 for item in shape):
        raise OperatorSummaryError("identity.tensor.shape: expected two positive integers")
    shape_tuple = tuple(shape)
    expected_production_sha = PRODUCTION_TENSORS.get((tensor["name"], shape_tuple))
    if expected_production_sha is None:
        raise OperatorSummaryError("identity.tensor: unsupported production tensor tuple")
    if tensor["sha256"] != expected_production_sha:
        raise OperatorSummaryError("identity.tensor.sha256: wrong production tensor hash")
    _integer(tensor["packed_bytes"], "identity.tensor.packed_bytes", nonnegative=True)

    layout = _mapping(identity["layout"], "identity.layout")
    _exact(layout, {"logical_type", "source_layout", "executor_layout"}, "identity.layout")
    for key in layout:
        _string(layout[key], f"identity.layout.{key}")
    if layout["logical_type"] != "GGML_TYPE_Q1_0" or layout["source_layout"] != "GGUF_Q1_0/v1":
        raise OperatorSummaryError("identity.layout: unsupported logical/source layout")
    if layout["executor_layout"] != "CPU_REPACK_Q1_0_4x4":
        raise OperatorSummaryError("identity.layout.executor_layout must be CPU_REPACK_Q1_0_4x4")
    if "VIP" in layout["executor_layout"]:
        raise OperatorSummaryError("CPU operator layout must not be a VIP tile")

    workload = _mapping(identity["workload"], "identity.workload")
    _exact(workload, {"activation_f32_elements", "output_f32_elements"}, "identity.workload")
    for key in workload:
        _integer(workload[key], f"identity.workload.{key}", nonnegative=True)
    k, m = shape_tuple
    if workload["activation_f32_elements"] != k or workload["output_f32_elements"] != m:
        raise OperatorSummaryError("identity.workload dimensions do not match tensor shape")
    for key in ("weight_buffer_type", "backend", "kernel"):
        _string(identity[key], f"identity.{key}")
    _bool(identity["strict_mode"], "identity.strict_mode")
    _integer(identity["threads"], "identity.threads", nonnegative=True)
    _integer(identity["logical_gemv_count"], "identity.logical_gemv_count", nonnegative=True)
    if (identity["weight_buffer_type"], identity["backend"], identity["kernel"], identity["strict_mode"]) != (
            "CPU_REPACK", "ggml-cpu", "q1_0_4x4_q8_0", True):
        raise OperatorSummaryError("identity: CPU_REPACK strict executor is required")
    if identity["logical_gemv_count"] != 1:
        raise OperatorSummaryError("identity.logical_gemv_count must be one")
    if identity["threads"] <= 0:
        raise OperatorSummaryError("identity.execution.threads must be positive")

    for key in ("declared_canonical_q1_bytes", "declared_cpu_repack_bytes", "expanded_weight_ddr_bytes",
                "activation_f32_bytes", "activation_q8_bytes", "output_f32_bytes",
                "physical_resident_weight_copies", "cpu_fallback_count"):
        _integer(memory[key], f"memory.{key}", nonnegative=True)
    if memory["declared_canonical_q1_bytes"] != memory["declared_cpu_repack_bytes"]:
        raise OperatorSummaryError("memory: canonical and CPU_REPACK sizes differ")
    if memory["declared_canonical_q1_bytes"] != tensor["packed_bytes"]:
        raise OperatorSummaryError("memory canonical size does not match tensor packed bytes")
    expected_packed = (k * m // 128) * 18 if k * m % 128 == 0 else -1
    if expected_packed != 12533760 or tensor["packed_bytes"] != expected_packed:
        raise OperatorSummaryError("identity.tensor.packed_bytes: wrong pinned Q1 block size")
    expected_activation_f32 = k * 4
    expected_activation_q8 = (k // 32) * 34 if k % 32 == 0 else -1
    expected_output_f32 = m * 4
    if (memory["activation_f32_bytes"], memory["activation_q8_bytes"], memory["output_f32_bytes"]) != (
            expected_activation_f32, expected_activation_q8, expected_output_f32):
        raise OperatorSummaryError("memory activation/output byte sizes do not match tensor shape")
    if memory["expanded_weight_ddr_bytes"] != 0:
        raise OperatorSummaryError("memory.expanded_weight_ddr_bytes must be zero")
    if memory["physical_resident_weight_copies"] != 1:
        raise OperatorSummaryError("memory.physical_resident_weight_copies must be one")
    if memory["cpu_fallback_count"] != 0:
        raise OperatorSummaryError("memory.cpu_fallback_count must be zero")
    for key in ("observed_ddr_read_bytes", "observed_ddr_write_bytes"):
        if memory[key] is not None:
            _integer(memory[key], f"memory.{key}", nonnegative=True)
            if memory[key] == 0:
                raise OperatorSummaryError(f"memory.{key} must be positive when measured")
    if result["output_f32_bytes"] != output_bytes or memory["output_f32_bytes"] != output_bytes:
        raise OperatorSummaryError("result/output_f32_bytes does not match output artifact")

    for key in ("swap_delta_kib",):
        _integer(telemetry[key], f"telemetry.{key}")
    if telemetry["swap_delta_kib"] != 0:
        raise OperatorSummaryError("telemetry.swap_delta_kib must be zero")
    for key in ("thermal_guard_passed", "throttle_detected", "oom_detected", "fault_detected"):
        _bool(telemetry[key], f"telemetry.{key}")
    if not telemetry["thermal_guard_passed"] or any(telemetry[key] for key in ("throttle_detected", "oom_detected", "fault_detected")):
        raise OperatorSummaryError("telemetry: thermal/throttle/OOM/fault qualification failed")


def summarize(runner_jsonl: Path, golden_f32: Path, output_f32: Path,
              telemetry_path: Path, thermal_guard: Path) -> dict[str, Any]:
    rows = _read_jsonl(runner_jsonl)
    identity, memory, measured, result = _parse_bundle(rows)
    telemetry = _read_telemetry(telemetry_path)
    reference, reference_bytes = _read_f32(golden_f32, "golden F32")
    output, output_bytes = _read_f32(output_f32, "output F32")
    # Correctness is intentionally evaluated before timing/performance gates.
    if len(reference) != len(output):
        raise OperatorSummaryError("correctness: reference/output element counts differ")
    _validate_metadata(identity, memory, telemetry, result, len(output_bytes))
    dot = sum(left * right for left, right in zip(reference, output))
    ref_norm = math.sqrt(sum(value * value for value in reference))
    out_norm = math.sqrt(sum(value * value for value in output))
    if ref_norm == 0.0 or out_norm == 0.0:
        raise OperatorSummaryError("correctness: zero-norm F32 output")
    cosine = dot / (ref_norm * out_norm)
    max_abs_error = max(abs(left - right) for left, right in zip(reference, output))
    reference_abs_max = max(abs(value) for value in reference)
    error_limit = 1e-4 * max(1.0, reference_abs_max)
    correctness_passed = cosine >= COSINE_MIN and max_abs_error <= error_limit
    if not correctness_passed:
        raise OperatorSummaryError("correctness gate failed")
    thermal_info = _validate_sidecar(thermal_guard)

    compute = _stats([_number(row["compute_us"], "sample.compute_us", nonnegative=True) for row in measured])
    host = _stats([_number(row["host_us"], "sample.host_us", nonnegative=True) for row in measured])
    if compute["population_cv"] > CV_LIMIT or host["population_cv"] > CV_LIMIT:
        raise OperatorSummaryError("performance gate failed: population CV exceeds 2%")
    model = identity["model"]
    tensor = identity["tensor"]
    layout = identity["layout"]
    workload = identity["workload"]
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    executor = {"backend": identity["backend"], "kernel": identity["kernel"],
                "weight_buffer_type": identity["weight_buffer_type"], "strict_mode": identity["strict_mode"]}
    execution = {"logical_gemv_count": identity["logical_gemv_count"],
                 "cpu_fallback_count": memory["cpu_fallback_count"], "threads": identity["threads"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "model": model,
        "tensor": tensor,
        "layout": layout,
        "workload": {**workload, "warmup_samples": 1, "measured_samples": MEASURED_SAMPLES},
        "executor": executor,
        "memory": {key: value for key, value in memory.items() if key not in {"schema_version", "record", "cpu_fallback_count"}},
        "execution": execution,
        "timing": {"warmup_samples": 1, "measured_samples": MEASURED_SAMPLES,
                    "host_us": host, "compute_us": compute},
        "correctness": {"reference_sha256": reference_hash, "output_sha256": output_hash,
                        "reference_f32_bytes": len(reference_bytes), "output_f32_bytes": len(output_bytes),
                        "reference_abs_max": reference_abs_max, "cosine": cosine, "max_abs_error": max_abs_error,
                        "thresholds": {"cosine_min": COSINE_MIN, "max_abs_error_max": error_limit},
                        "finite": True, "finite_passed": True,
                        "cosine_passed": cosine >= COSINE_MIN,
                        "max_abs_error_passed": max_abs_error <= error_limit,
                        "passed": correctness_passed},
        "telemetry": {**telemetry, **thermal_info},
        "qualification": {"correctness_passed": True, "sample_count_passed": True,
                           "cv_passed": True, "memory_passed": True, "executor_passed": True,
                           "thermal_passed": telemetry["thermal_guard_passed"], "swap_passed": True,
                           "passed": True},
    }


def validate_summary(value: Any) -> None:
    root = _mapping(value, "summary")
    required = {"schema_version", "run_id", "model", "tensor", "layout", "workload", "executor",
                "memory", "execution", "timing", "correctness", "telemetry", "qualification"}
    _exact(root, required, "summary")
    if root["schema_version"] != SCHEMA_VERSION:
        raise OperatorSummaryError(f"summary.schema_version must be {SCHEMA_VERSION}")
    _string(root["run_id"], "summary.run_id")
    for section in required - {"schema_version", "run_id"}:
        _mapping(root[section], f"summary.{section}")
    memory = root["memory"]
    memory_required = {"declared_canonical_q1_bytes", "declared_cpu_repack_bytes", "expanded_weight_ddr_bytes",
                       "activation_f32_bytes", "activation_q8_bytes", "output_f32_bytes",
                       "observed_ddr_read_bytes", "observed_ddr_write_bytes", "physical_resident_weight_copies"}
    if set(memory) != memory_required:
        raise OperatorSummaryError("summary.memory: exact field set required")
    for key in memory_required:
        if key.startswith("observed_ddr_"):
            if memory[key] is not None:
                _integer(memory[key], f"summary.memory.{key}", nonnegative=True)
        else:
            _integer(memory[key], f"summary.memory.{key}", nonnegative=True)
    if memory["expanded_weight_ddr_bytes"] != 0:
        raise OperatorSummaryError("summary.memory.expanded_weight_ddr_bytes must be zero")
    for key in ("observed_ddr_read_bytes", "observed_ddr_write_bytes"):
        if memory[key] is not None:
            _integer(memory[key], f"summary.memory.{key}", nonnegative=True)
            if memory[key] == 0:
                raise OperatorSummaryError(f"summary.memory.{key} must be positive when measured")
    model = root["model"]
    _exact(model, {"id", "sha256", "size_bytes", "quantization"}, "summary.model")
    if (_sha(model["sha256"], "summary.model.sha256") != EXPECTED_MODEL_SHA256
            or model["id"] != EXPECTED_MODEL_ID or model["size_bytes"] != EXPECTED_MODEL_SIZE):
        raise OperatorSummaryError("summary.model.sha256: wrong pinned model hash")
    _string(model["id"], "summary.model.id")
    _integer(model["size_bytes"], "summary.model.size_bytes", nonnegative=True)
    if model["quantization"] != "Q1_0":
        raise OperatorSummaryError("summary.model.quantization must be Q1_0")
    tensor = root["tensor"]
    _exact(tensor, {"name", "sha256", "shape", "packed_bytes"}, "summary.tensor")
    shape = tensor["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(item) is not int or item <= 0 for item in shape):
        raise OperatorSummaryError("summary.tensor.shape: expected two positive integers")
    expected_hash = PRODUCTION_TENSORS.get((tensor.get("name"), tuple(shape)))
    if expected_hash is None or _sha(tensor["sha256"], "summary.tensor.sha256") != expected_hash:
        raise OperatorSummaryError("summary.tensor: wrong production tensor tuple/hash")
    _integer(tensor["packed_bytes"], "summary.tensor.packed_bytes", nonnegative=True)
    layout = root["layout"]
    _exact(layout, {"logical_type", "source_layout", "executor_layout"}, "summary.layout")
    if layout != {"logical_type": "GGML_TYPE_Q1_0", "source_layout": "GGUF_Q1_0/v1",
                  "executor_layout": "CPU_REPACK_Q1_0_4x4"}:
        raise OperatorSummaryError("summary.layout: unexpected CPU layout")
    k, m = shape
    expected_packed = (k * m // 128) * 18 if k * m % 128 == 0 else -1
    if expected_packed != 12533760 or tensor["packed_bytes"] != expected_packed:
        raise OperatorSummaryError("summary.tensor.packed_bytes: wrong pinned Q1 block size")
    if memory["declared_canonical_q1_bytes"] != tensor["packed_bytes"] \
            or memory["declared_cpu_repack_bytes"] != tensor["packed_bytes"] \
            or memory["activation_f32_bytes"] != k * 4 \
            or memory["activation_q8_bytes"] != (k // 32) * 34 \
            or memory["output_f32_bytes"] != m * 4:
        raise OperatorSummaryError("summary.memory: shape/tensor byte invariant failed")
    workload = root["workload"]
    _exact(workload, {"activation_f32_elements", "output_f32_elements", "warmup_samples", "measured_samples"},
           "summary.workload")
    if workload != {"activation_f32_elements": k, "output_f32_elements": m,
                    "warmup_samples": 1, "measured_samples": 50}:
        raise OperatorSummaryError("summary.workload: dimensions/counts do not match contract")
    executor = root["executor"]
    _exact(executor, {"backend", "kernel", "weight_buffer_type", "strict_mode"}, "summary.executor")
    if executor != {"backend": "ggml-cpu", "kernel": "q1_0_4x4_q8_0",
                    "weight_buffer_type": "CPU_REPACK", "strict_mode": True}:
        raise OperatorSummaryError("summary.executor: unexpected executor")
    execution = root["execution"]
    _exact(execution, {"logical_gemv_count", "cpu_fallback_count", "threads"}, "summary.execution")
    if execution["logical_gemv_count"] != 1 or execution["cpu_fallback_count"] != 0:
        raise OperatorSummaryError("summary.execution: fallback/GEMV gate failed")
    _integer(execution["threads"], "summary.execution.threads", nonnegative=True)
    if execution["threads"] <= 0:
        raise OperatorSummaryError("summary.execution.threads must be positive")
    timing = root["timing"]
    _exact(timing, {"warmup_samples", "measured_samples", "host_us", "compute_us"}, "summary.timing")
    if timing["warmup_samples"] != 1 or timing["measured_samples"] != 50:
        raise OperatorSummaryError("summary.timing: expected one warmup and 50 measured samples")
    stat_fields = {"min", "p10", "median", "p90", "max", "mean", "population_cv"}
    for name in ("host_us", "compute_us"):
        stats = _mapping(timing[name], f"summary.timing.{name}")
        _exact(stats, stat_fields, f"summary.timing.{name}")
        for key in stat_fields:
            _number(stats[key], f"summary.timing.{name}.{key}", nonnegative=True)
        if not (stats["min"] <= stats["p10"] <= stats["median"] <= stats["p90"] <= stats["max"]):
            raise OperatorSummaryError(f"summary.timing.{name}: quantiles are not ordered")
        if not stats["min"] <= stats["mean"] <= stats["max"]:
            raise OperatorSummaryError(f"summary.timing.{name}: mean is outside min/max")
        if stats["population_cv"] > CV_LIMIT:
            raise OperatorSummaryError(f"summary.timing.{name}: CV exceeds 2%")
    correctness = root["correctness"]
    _exact(correctness, {"reference_sha256", "output_sha256", "reference_f32_bytes", "output_f32_bytes",
                         "reference_abs_max", "cosine", "max_abs_error", "thresholds", "finite",
                         "finite_passed", "cosine_passed", "max_abs_error_passed", "passed"},
           "summary.correctness")
    _sha(correctness["reference_sha256"], "summary.correctness.reference_sha256")
    _sha(correctness["output_sha256"], "summary.correctness.output_sha256")
    for key in ("reference_f32_bytes", "output_f32_bytes"):
        _integer(correctness[key], f"summary.correctness.{key}", nonnegative=True)
    _number(correctness["reference_abs_max"], "summary.correctness.reference_abs_max", nonnegative=True)
    _number(correctness["cosine"], "summary.correctness.cosine")
    _number(correctness["max_abs_error"], "summary.correctness.max_abs_error", nonnegative=True)
    thresholds = correctness["thresholds"]
    _exact(thresholds, {"cosine_min", "max_abs_error_max"}, "summary.correctness.thresholds")
    if thresholds["cosine_min"] != COSINE_MIN or not math.isclose(
            thresholds["max_abs_error_max"], 1e-4 * max(1.0, correctness["reference_abs_max"]),
            rel_tol=0.0, abs_tol=1e-12):
        raise OperatorSummaryError("summary.correctness.thresholds: wrong threshold")
    if correctness["output_f32_bytes"] != memory["output_f32_bytes"] \
            or correctness["reference_f32_bytes"] != correctness["output_f32_bytes"] \
            or correctness["reference_abs_max"] < 0:
        raise OperatorSummaryError("summary.correctness: byte/reference invariant failed")
    finite_passed = correctness["finite"] is True and correctness["finite_passed"] is True
    cosine_passed = correctness["cosine"] >= thresholds["cosine_min"]
    max_error_passed = correctness["max_abs_error"] <= thresholds["max_abs_error_max"]
    if correctness["cosine_passed"] is not cosine_passed or correctness["max_abs_error_passed"] is not max_error_passed:
        raise OperatorSummaryError("summary.correctness: decision/value mismatch")
    if not finite_passed or not cosine_passed or not max_error_passed or correctness["passed"] is not True:
        raise OperatorSummaryError("summary.correctness: gate failed")
    telemetry = root["telemetry"]
    _exact(telemetry, {"schema_version", "swap_delta_kib", "thermal_guard_passed", "throttle_detected",
                       "oom_detected", "fault_detected", "thermal_guard_samples"}, "summary.telemetry")
    if telemetry["schema_version"] != "q1-cpu-operator-telemetry/v1":
        raise OperatorSummaryError("summary.telemetry.schema_version: unsupported")
    _integer(telemetry["swap_delta_kib"], "summary.telemetry.swap_delta_kib")
    for key in ("thermal_guard_passed", "throttle_detected", "oom_detected", "fault_detected"):
        _bool(telemetry[key], f"summary.telemetry.{key}")
    if telemetry["swap_delta_kib"] != 0 or telemetry["thermal_guard_passed"] is not True:
        raise OperatorSummaryError("summary.telemetry: swap/thermal gate failed")
    if any(telemetry[key] is not False for key in ("throttle_detected", "oom_detected", "fault_detected")):
        raise OperatorSummaryError("summary.telemetry: fault gate failed")
    _integer(telemetry["thermal_guard_samples"], "summary.telemetry.thermal_guard_samples", nonnegative=True)
    if telemetry["thermal_guard_samples"] == 0:
        raise OperatorSummaryError("summary.telemetry: thermal guard has no samples")
    qualification = root["qualification"]
    qual_fields = {"correctness_passed", "sample_count_passed", "cv_passed", "memory_passed",
                   "executor_passed", "thermal_passed", "swap_passed", "passed"}
    _exact(qualification, qual_fields, "summary.qualification")
    if any(qualification[key] is not True for key in qual_fields):
        raise OperatorSummaryError("summary.qualification: failed gate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_jsonl", type=Path, nargs="?")
    parser.add_argument("--golden-f32", type=Path)
    parser.add_argument("--output-f32", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--thermal-guard", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check is not None:
            if any(value is not None for value in (args.runner_jsonl, args.golden_f32, args.output_f32,
                                                   args.thermal_guard, args.telemetry, args.output,
                                                   )):
                raise OperatorSummaryError("--check is exclusive with runner/output arguments")
            validate_summary(_json_load(args.check.read_text(encoding="utf-8"), "summary"))
            print(f"PASS {SCHEMA_VERSION}")
            return 0
        if (args.runner_jsonl is None or args.golden_f32 is None or args.output_f32 is None
                or args.telemetry is None or args.thermal_guard is None or args.output is None):
            raise OperatorSummaryError("runner, --golden-f32, --output-f32, --telemetry, --thermal-guard, and --output are required")
        summary = summarize(args.runner_jsonl, args.golden_f32, args.output_f32, args.telemetry,
                            args.thermal_guard)
        _write_exclusive_json(args.output, summary)
        print(f"PASS {SCHEMA_VERSION}")
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, OperatorSummaryError) as error:
        print(f"summarize_q1_cpu_operator: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
