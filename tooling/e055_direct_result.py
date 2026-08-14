#!/usr/bin/env python3
"""Validate and plot the bounded direct-OpenSSH E055 target phase.

This module is intentionally separate from the complete 1,260-row promotion
analyzer.  It validates the exact first target microgate (20 runs), derives
paired statistics from immutable raw outputs, and always reports that the
matrix is incomplete.  It never treats PMU event counts as bytes and never
uses a ratio of unequal total run windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PHASE_SCHEMA = "e055-direct-target-analysis/v1"
RAW_MANIFEST_SCHEMA = "e055-direct-raw-manifest/v1"
HARNESS_SCHEMA = "e055-q1-hot-cold-harness/v1"
PMU_SCHEMA = "e049c-arm-pmu/v2"
EXPECTED_HARNESS_PATH = (
    "/tmp/e055-harness-native-7dcdab61-20260814t161000z-c/e055"
)
FILES_PER_RUN = (
    "e049c.json",
    "harness.stdout.raw",
    "harness.stderr.raw",
    "wrapper.stdout.raw",
    "wrapper.stderr.raw",
)
EVENT_CONFIGS = {
    "cpu_cycles": "0x11",
    "instructions": "0x8",
    "stall_backend": "0x24",
}
OUTPUT_CHECKSUMS = {
    "hot_repeat": "0xea018b4296d17e54",
    "cold_conditioned": "0xf0d4224634da5da3",
}
HEX64 = set("0123456789abcdef")
REQUIRED_MATRIX_ROWS = 7 * 2 * 3 * 2 * 3 * 5


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if not _is_int(value) or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a {qualifier}integer")
    return value


def _require_checksum(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 18
        or not value.startswith("0x")
        or any(char not in HEX64 for char in value[2:])
        or value == "0x0000000000000000"
    ):
        raise ValueError(f"{name} must be a canonical nonzero uint64 checksum")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_order(cpu: int, pair: int) -> tuple[str, str]:
    del cpu
    return (
        ("hot_repeat", "cold_conditioned")
        if pair % 2
        else ("cold_conditioned", "hot_repeat")
    )


def _parse_run_name(name: str) -> tuple[int, int, str]:
    parts = name.split("-")
    if len(parts) != 3 or parts[0] not in ("cpu0", "cpu6"):
        raise ValueError(f"noncanonical run directory: {name}")
    if not parts[1].startswith("pair") or not parts[1][4:].isdigit():
        raise ValueError(f"noncanonical pair in run directory: {name}")
    pair = int(parts[1][4:])
    state = {"hot": "hot_repeat", "cold": "cold_conditioned"}.get(parts[2])
    if pair not in range(1, 6) or state is None:
        raise ValueError(f"run is outside the bounded phase: {name}")
    return int(parts[0][3:]), pair, state


def _read_one_json_line(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is not UTF-8") from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(f"{path.name} must contain exactly one nonempty JSON line")
    try:
        value = json.loads(lines[0])
    except ValueError as exc:
        raise ValueError(f"{path.name} is not JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_files(run: Path) -> None:
    actual = sorted(path.name for path in run.iterdir())
    if actual != sorted(FILES_PER_RUN):
        raise ValueError(f"{run.name} has missing or extra raw roles")
    seen_inodes: set[tuple[int, int]] = set()
    for name in FILES_PER_RUN:
        path = run / name
        status = path.lstat()
        if not path.is_file() or path.is_symlink() or status.st_nlink != 1:
            raise ValueError(f"{run.name}/{name} is not a regular single-link file")
        identity = (status.st_dev, status.st_ino)
        if identity in seen_inodes:
            raise ValueError(f"{run.name} reuses one inode for multiple raw roles")
        seen_inodes.add(identity)
    for name in ("harness.stderr.raw", "wrapper.stdout.raw", "wrapper.stderr.raw"):
        if (run / name).stat().st_size != 0:
            raise ValueError(f"{run.name}/{name} must be empty for a valid sample")


def _validate_harness(
    payload: Mapping[str, Any], cpu: int, state: str, *, observed_cpu: bool
) -> dict[str, Any]:
    exact = {
        "schema": HARNESS_SCHEMA,
        "mode": "full_dotprod",
        "cache_state": state,
        "q1_layout": "E039 stock native block_q1_0x4 4x4 DOTPROD",
        "golden_pass": True,
        "golden_cases": 18,
        "target_working_set_bytes": 65536,
        "actual_working_set_bytes": 65728,
        "blocks": 316,
    }
    for key, expected in exact.items():
        value = payload.get(key)
        if isinstance(expected, int) and not isinstance(expected, bool):
            _require_int(value, f"harness {key}")
        if value != expected:
            raise ValueError(f"harness {key} does not match the bounded phase")
    reported_cpu = _require_int(payload.get("cpu"), "harness cpu")
    if reported_cpu != cpu:
        qualifier = "observed CPU" if observed_cpu else "requested CPU label"
        raise ValueError(
            f"harness {qualifier} does not match requested taskset affinity"
        )
    iterations = _require_int(payload.get("iterations"), "harness iterations", positive=True)
    calls = _require_int(payload.get("calls"), "harness calls", positive=True)
    elapsed = _require_int(payload.get("elapsed_ns"), "harness elapsed_ns", positive=True)
    expected_calls = 250 if state == "hot_repeat" else 1
    if iterations != expected_calls or calls != expected_calls:
        raise ValueError("harness calls/iterations do not match cache conditioning")
    _require_int(payload.get("first_call_ns"), "harness first_call_ns", positive=True)
    rate = payload.get("calls_per_second")
    if not _is_finite_number(rate) or float(rate) <= 0.0:
        raise ValueError("harness calls_per_second must be positive and finite")
    expected_rate = calls * 1e9 / elapsed
    if not math.isclose(float(rate), expected_rate, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("harness calls_per_second disagrees with elapsed/calls")
    logical = payload.get("logical_bytes_per_call")
    if logical != {
        "q1_packed_bytes": 22752,
        "q8_bytes": 42976,
        "total_input_bytes": 65728,
        "output_bytes": 16,
        "dot_products": 161792,
    }:
        raise ValueError("harness logical work descriptor is not exact")
    checksum = _require_checksum(payload.get("checksum"), "harness output")
    if checksum != OUTPUT_CHECKSUMS[state]:
        raise ValueError("harness output checksum does not match the exact kernel fixture")
    condition = payload.get("cold_conditioning")
    if not isinstance(condition, Mapping) or condition.get("verified_touched") is not True:
        raise ValueError("harness conditioning was not verified")
    if state == "cold_conditioned":
        expected_condition = {
            "strategy": "verified_write_read_each_64B_line",
            "requested_bytes": 67108864,
            "actual_bytes": 67108864,
            "line_bytes": 64,
            "lines_touched": 1048576,
            "checksum": "0x8d3ea13d15850279",
            "verified_touched": True,
            "warmup_calls": 0,
        }
    else:
        expected_condition = {
            "strategy": "verified_kernel_warmup",
            "requested_bytes": 65728,
            "actual_bytes": 65728,
            "line_bytes": 64,
            "lines_touched": 1027,
            "verified_touched": True,
            "warmup_calls": 16,
        }
        _require_checksum(condition.get("checksum"), "hot conditioning")
        expected_condition["checksum"] = condition.get("checksum")
    for key in ("requested_bytes", "actual_bytes", "line_bytes", "lines_touched", "warmup_calls"):
        _require_int(condition.get(key), f"conditioning {key}")
    if dict(condition) != expected_condition:
        raise ValueError("harness conditioning metadata is not canonical")
    if payload.get("sync") != {
        "requested": True,
        "started": True,
        "acknowledged": True,
        "ended": True,
        "sequence": "S/A/E",
    }:
        raise ValueError("harness synchronization markers are not exact")
    if payload.get("qualification") != "unqualified_harness_output_requires_E049c_join":
        raise ValueError("harness qualification label is missing")
    return {
        "elapsed_ns": elapsed,
        "calls": calls,
        "checksum": checksum,
        "ns_per_traversal": elapsed / calls,
        "traversals_per_second": calls * 1e9 / elapsed,
    }


def _validate_pmu(
    payload: Mapping[str, Any],
    cpu: int,
    state: str,
    harness: Mapping[str, Any],
    *,
    require_observed_cpu: bool,
) -> dict[str, int]:
    if payload.get("schema_version") != PMU_SCHEMA:
        raise ValueError("PMU schema is not E049c v2")
    exact = {
        "status": "ok",
        "sample_valid": True,
        "event_source": "armv8_pmuv3_raw_config",
        "counter_semantics": "event counts only; no DDR-byte conversion",
        "event_group": "core",
        "software_group_size_limit": 4,
        "event_group_size": 3,
        "min_running_ratio": 0.95,
        "failure_reason": None,
    }
    for key, expected in exact.items():
        value = payload.get(key)
        if key in ("software_group_size_limit", "event_group_size"):
            _require_int(value, f"PMU {key}")
        if key == "sample_valid" and not isinstance(value, bool):
            raise ValueError("PMU sample_valid must be Boolean")
        if value != expected:
            raise ValueError(f"PMU {key} is not canonical")
    pid = _require_int(payload.get("pid"), "PMU pid", positive=True)
    pgid = _require_int(payload.get("process_group"), "PMU process_group", positive=True)
    if pid != pgid:
        raise ValueError("PMU pid and process group must match")
    expected_calls = 250 if state == "hot_repeat" else 1
    expected_command = [
        "taskset", "-c", str(cpu), EXPECTED_HARNESS_PATH,
        "--mode", "full_dotprod", "--cache-state", state,
        "--working-set-bytes", "65536",
    ]
    if not require_observed_cpu:
        expected_command.extend(["--cpu", str(cpu)])
    expected_command.extend(["--iterations", str(expected_calls)])
    if state == "cold_conditioned":
        expected_command.extend(["--budget-ms", "0"])
    expected_command.extend([
        "--warmup", "16" if state == "hot_repeat" else "0",
        "--thrash-bytes", "67108864", "--sync",
    ])
    if payload.get("command") != expected_command:
        if require_observed_cpu and "--cpu" in payload.get("command", []):
            raise ValueError(
                "requested CPU label is forbidden; harness must report sched_getcpu"
            )
        raise ValueError("PMU child command is not the exact direct phase argv")
    if payload.get("sync") != {
        "mode": "start_ack_end",
        "started": True,
        "acknowledged": True,
        "ended": True,
    }:
        raise ValueError("PMU synchronization markers are not exact")
    measured = _require_int(
        payload.get("measured_elapsed_ns"), "PMU measured_elapsed_ns", positive=True
    )
    thermal = payload.get("thermal")
    if not isinstance(thermal, Mapping):
        raise ValueError("PMU thermal object is missing")
    if (
        thermal.get("guard_enabled") is not True
        or thermal.get("checked") is not True
        or thermal.get("readable") is not True
        or thermal.get("tripped") is not False
        or not _is_finite_number(thermal.get("limit_c"))
        or float(thermal["limit_c"]) != 85.0
        or not _is_finite_number(thermal.get("max_observed_c"))
        or float(thermal["max_observed_c"]) > 85.0
    ):
        raise ValueError("PMU thermal safety gate did not pass")
    exit_status = payload.get("exit")
    if not isinstance(exit_status, Mapping):
        raise ValueError("PMU exit object is missing")
    _require_int(exit_status.get("code"), "PMU exit code")
    _require_int(exit_status.get("raw_wait_status"), "PMU wait status")
    if exit_status != {"code": 0, "raw_wait_status": 0}:
        raise ValueError("PMU child did not exit cleanly")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != len(EVENT_CONFIGS):
        raise ValueError("PMU core group is incomplete")
    values: dict[str, int] = {}
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("PMU event is malformed")
        name = event.get("name")
        if name not in EVENT_CONFIGS or name in seen:
            raise ValueError("PMU event name is missing, duplicated, or unexpected")
        seen.add(name)
        if event.get("config") != EVENT_CONFIGS[name]:
            raise ValueError("PMU event config does not match E049c")
        if (
            event.get("support") != "supported"
            or event.get("count_semantics") != "event_count_not_bytes"
            or event.get("sample_valid") is not True
            or event.get("errno") != 0
            or event.get("error") is not None
        ):
            raise ValueError("PMU event status is not valid")
        value = _require_int(event.get("value"), f"PMU {name} count")
        enabled = _require_int(event.get("time_enabled_ns"), f"PMU {name} enabled", positive=True)
        running = _require_int(event.get("time_running_ns"), f"PMU {name} running", positive=True)
        ratio = event.get("running_ratio")
        if not isinstance(ratio, float) or not math.isfinite(ratio) or ratio != 1.0:
            raise ValueError("PMU running ratio must be an exact runtime float 1.0")
        if running != enabled:
            raise ValueError("PMU running/enabled windows differ despite ratio 1")
        if enabled + 1_000_000 < int(harness["elapsed_ns"]):
            raise ValueError("PMU event window is implausibly shorter than harness elapsed")
        values[name] = value
    if measured + 1_000_000 < int(harness["elapsed_ns"]):
        raise ValueError("PMU measured window is implausibly shorter than harness elapsed")
    return values


def _phase_run_dirs(raw_dir: Path, require_complete: bool) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise ValueError("direct phase raw path must be a real directory")
    run_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if any(not path.is_dir() for path in raw_dir.iterdir()):
        raise ValueError("direct phase raw root may contain only run directories")
    expected_names = {
        f"cpu{cpu}-pair{pair}-{state}"
        for cpu in (0, 6)
        for pair in range(1, 6)
        for state in ("hot", "cold")
    }
    actual_names = {path.name for path in run_dirs}
    if require_complete and actual_names != expected_names:
        raise ValueError("direct phase is not the exact 20-run bounded matrix")
    if not run_dirs:
        raise ValueError("direct phase contains no runs")
    return run_dirs


def _derive_run(run: Path, *, require_observed_cpu: bool) -> dict[str, Any]:
    cpu, pair, state = _parse_run_name(run.name)
    _validate_files(run)
    harness_payload = _read_one_json_line(run / "harness.stdout.raw")
    harness = _validate_harness(
        harness_payload, cpu, state, observed_cpu=require_observed_cpu
    )
    try:
        pmu_payload = json.loads((run / "e049c.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{run.name}/e049c.json is not valid UTF-8 JSON") from exc
    if not isinstance(pmu_payload, Mapping):
        raise ValueError(f"{run.name}/e049c.json must contain an object")
    events = _validate_pmu(
        pmu_payload,
        cpu,
        state,
        harness,
        require_observed_cpu=require_observed_cpu,
    )
    return {
        "run_id": run.name,
        "cpu": cpu,
        "requested_cpu": cpu,
        "observed_cpu": cpu if require_observed_cpu else None,
        "cpu_evidence": (
            "taskset_requested_plus_post_measurement_sched_getcpu"
            if require_observed_cpu
            else "historical_requested_label_only"
        ),
        "pair_index": pair,
        "pair_order": "->".join(_expected_order(cpu, pair)),
        "cache_state": state,
        "elapsed_ns": harness["elapsed_ns"],
        "calls": harness["calls"],
        "ns_per_traversal": harness["ns_per_traversal"],
        "traversals_per_second": harness["traversals_per_second"],
        "output_checksum": harness["checksum"],
        "temperature_c": float(pmu_payload["thermal"]["max_observed_c"]),
        "pmu_counts": events,
        "raw_sha256": {name: _sha256(run / name) for name in FILES_PER_RUN},
        "raw_size_bytes": {name: (run / name).stat().st_size for name in FILES_PER_RUN},
    }


def _diagnostic_run(run: Path) -> dict[str, Any] | None:
    """Recover timing for a plotted invalid point without qualifying the sample."""

    try:
        cpu, pair, state = _parse_run_name(run.name)
        harness = _read_one_json_line(run / "harness.stdout.raw")
        calls = _require_int(harness.get("calls"), "diagnostic calls", positive=True)
        elapsed = _require_int(harness.get("elapsed_ns"), "diagnostic elapsed", positive=True)
    except (OSError, ValueError):
        return None
    return {
        "run_id": run.name,
        "cpu": cpu,
        "pair_index": pair,
        "cache_state": state,
        "calls": calls,
        "elapsed_ns": elapsed,
        "ns_per_traversal": elapsed / calls,
    }


def audit_phase(
    raw_dir: Path,
    *,
    require_complete: bool = True,
    require_observed_cpu: bool = True,
) -> dict[str, Any]:
    """Audit every raw run and retain exact fail-closed qualification errors."""

    raw_dir = Path(raw_dir)
    run_dirs = _phase_run_dirs(raw_dir, require_complete)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, str]] = set()
    for run in run_dirs:
        key = _parse_run_name(run.name)
        if key in seen_keys:
            raise ValueError("direct phase contains a duplicate run key")
        seen_keys.add(key)
        try:
            rows.append(
                _derive_run(run, require_observed_cpu=require_observed_cpu)
            )
        except (OSError, ValueError) as exc:
            invalid.append({
                "run_id": run.name,
                "reason": str(exc),
                "diagnostic_timing": _diagnostic_run(run),
            })
    return {
        "phase": raw_dir.parent.name if raw_dir.name == "raw" else raw_dir.name,
        "raw_run_count": len(run_dirs),
        "qualified_rows": sorted(
            rows, key=lambda row: (row["cpu"], row["pair_index"], row["cache_state"])
        ),
        "invalid_runs": sorted(invalid, key=lambda item: item["run_id"]),
    }


def validate_phase(
    raw_dir: Path,
    *,
    require_complete: bool = True,
    require_observed_cpu: bool = True,
) -> list[dict[str, Any]]:
    """Derive rows only when every raw sample passes the accepted qualifier."""

    audit = audit_phase(
        raw_dir,
        require_complete=require_complete,
        require_observed_cpu=require_observed_cpu,
    )
    if audit["invalid_runs"]:
        failures = "; ".join(
            f"{item['run_id']}: {item['reason']}" for item in audit["invalid_runs"]
        )
        raise ValueError(f"{len(audit['invalid_runs'])} raw sample(s) failed: {failures}")
    return audit["qualified_rows"]


def validate_run(
    run_dir: Path, *, require_observed_cpu: bool = True
) -> dict[str, Any]:
    """Validate one freshly copied direct run before any next target sample."""

    row = _derive_run(
        Path(run_dir), require_observed_cpu=require_observed_cpu
    )
    return {
        "schema": "e055-direct-live-qualification/v1",
        "status": "PASS",
        "run_id": row["run_id"],
        "cpu": row["cpu"],
        "requested_cpu": row["requested_cpu"],
        "observed_cpu": row["observed_cpu"],
        "cpu_evidence": row["cpu_evidence"],
        "pair_index": row["pair_index"],
        "cache_state": row["cache_state"],
        "ns_per_traversal": row["ns_per_traversal"],
        "max_temperature_c": row["temperature_c"],
        "raw_sha256": row["raw_sha256"],
    }


def _median(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot summarize an empty set")
    return float(statistics.median(items))


def analyze_phase(audit: Mapping[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired diagnostics without promoting an invalid/incomplete matrix."""

    if isinstance(audit, list):
        rows = audit
        raw_run_count = len(rows)
        invalid_runs: list[dict[str, Any]] = []
        phase = "unbound-list-input"
    else:
        rows = audit.get("qualified_rows")
        raw_run_count = audit.get("raw_run_count")
        invalid_runs = audit.get("invalid_runs")
        phase = audit.get("phase")
    if not isinstance(rows, list) or not _is_int(raw_run_count) or not isinstance(invalid_runs, list):
        raise ValueError("phase audit is malformed")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase audit is missing its phase identity")
    if raw_run_count != 20:
        raise ValueError("analysis requires the exact 20-run bounded phase")
    cpu_results: dict[str, Any] = {}
    for cpu in (0, 6):
        cpu_rows = [row for row in rows if row["cpu"] == cpu]
        pairs: list[dict[str, Any]] = []
        for pair_index in range(1, 6):
            pair_rows = [row for row in cpu_rows if row["pair_index"] == pair_index]
            if len(pair_rows) != 2:
                continue
            hot = next((row for row in pair_rows if row["cache_state"] == "hot_repeat"), None)
            cold = next((row for row in pair_rows if row["cache_state"] == "cold_conditioned"), None)
            if hot is None or cold is None:
                raise ValueError("every pair must contain exactly hot and cold")
            event_ratios = {
                name: (cold["pmu_counts"][name] / cold["calls"])
                / (hot["pmu_counts"][name] / hot["calls"])
                for name in EVENT_CONFIGS
            }
            pairs.append({
                "pair_index": pair_index,
                "order": hot["pair_order"],
                "hot_ns_per_traversal": hot["ns_per_traversal"],
                "cold_ns_per_traversal": cold["ns_per_traversal"],
                "cold_hot_penalty": cold["ns_per_traversal"] / hot["ns_per_traversal"],
                "pmu_event_count_ratios_per_traversal": event_ratios,
            })
        hot_ns = [pair["hot_ns_per_traversal"] for pair in pairs]
        cold_ns = [pair["cold_ns_per_traversal"] for pair in pairs]
        if not pairs:
            raise ValueError(f"CPU{cpu} has no complete qualified pair")
        cpu_results[str(cpu)] = {
            "pairs": pairs,
            "qualified_pair_count": len(pairs),
            "median_hot_ns_per_traversal": _median(hot_ns),
            "median_cold_ns_per_traversal": _median(cold_ns),
            "median_hot_traversals_per_second": 1e9 / _median(hot_ns),
            "median_cold_traversals_per_second": 1e9 / _median(cold_ns),
            "median_pair_cold_hot_penalty": _median(
                pair["cold_hot_penalty"] for pair in pairs
            ),
            "median_pmu_event_count_ratios_per_traversal": {
                name: _median(
                    pair["pmu_event_count_ratios_per_traversal"][name]
                    for pair in pairs
                )
                for name in EVENT_CONFIGS
            },
        }
    return {
        "schema": PHASE_SCHEMA,
        "status": (
            "PASS_BOUNDED_MICROGATE"
            if not invalid_runs
            else "INVALID_PHASE_RAW_SAMPLE_FAILED_QUALIFIER"
        ),
        "phase": phase,
        "raw_run_count": raw_run_count,
        "valid_run_count": len(rows),
        "invalid_run_count": len(invalid_runs),
        "invalid_runs": invalid_runs,
        "chart_annotation": (
            "20/20 qualified; taskset CPU matched post-measurement sched_getcpu\n"
            "PMU event ratios are excluded from kernel-cache inference"
            if not invalid_runs
            else (
                "X = invalid raw sample qualifier; only qualified pairs are summarized\n"
                "PMU event ratios are excluded from kernel-cache inference"
            )
        ),
        "normalization": "elapsed_ns / calls",
        "scope": {
            "cpus": [0, 6],
            "working_set_bytes": 65536,
            "actual_working_set_bytes": 65728,
            "mode": "full_dotprod",
            "pmu_group": "core",
            "pairs_per_cpu": 5,
        },
        "max_temperature_c": max(row["temperature_c"] for row in rows),
        "cpus": cpu_results,
        "pmu_interpretation": {
            "unit": "raw event counts per traversal; never bytes",
            "qualified_for_kernel_cache_attribution": False,
            "reason": (
                "cold_conditioned performs exactly one measured traversal while hot_repeat "
                "performs 250; fixed S/A/E, launcher, and counter-window overhead remains in "
                "the single cold count and dominates the normalized PMU ratios"
            ),
        },
        "promotion": {
            "eligible": False,
            "qualified_matrix_rows": len(rows),
            "required_matrix_rows": REQUIRED_MATRIX_ROWS,
            "threshold_fraction": 0.04,
            "reason": (
                (
                    "one or more raw runs failed qualification and only one of 126 "
                    "phase cells was attempted; the complete matrix is mandatory"
                )
                if invalid_runs
                else (
                    "the bounded 20-run microgate passed, but only one of 126 phase "
                    "cells was attempted; the complete matrix is mandatory"
                )
            ),
        },
    }


def build_raw_manifest(result_dir: Path) -> dict[str, Any]:
    """Bind all non-generated evidence files by canonical path, size, and SHA-256."""

    allowed_roots = ("raw", "artifacts", "failures", "controls", "build-evidence")
    files: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for root_name in allowed_roots:
        root = result_dir / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            status = path.lstat()
            if path.is_symlink() or status.st_nlink != 1:
                raise ValueError(f"manifest evidence is not a regular single-link file: {path}")
            identity = (status.st_dev, status.st_ino)
            if identity in seen_inodes:
                raise ValueError("manifest evidence contains a hardlink alias")
            seen_inodes.add(identity)
            relative = path.relative_to(result_dir).as_posix()
            files.append({
                "path": relative,
                "size_bytes": status.st_size,
                "sha256": _sha256(path),
            })
    return {
        "schema": RAW_MANIFEST_SCHEMA,
        "phase": result_dir.name,
        "transport": "direct /usr/bin/ssh and /usr/bin/scp argv only",
        "device_attestation": False,
        "files": files,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id", "cpu", "pair_index", "pair_order", "cache_state", "calls",
        "elapsed_ns", "ns_per_traversal", "traversals_per_second", "temperature_c",
        "cpu_cycles", "instructions", "stall_backend", "output_checksum",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{name: row[name] for name in fieldnames if name in row},
                **row["pmu_counts"],
            })


def _write_chart(path: Path, analysis: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=160)
    colors = {0: "#2563eb", 6: "#dc2626"}
    markers = {"hot": "o", "cold": "s"}
    for cpu in (0, 6):
        pairs = analysis["cpus"][str(cpu)]["pairs"]
        x = [item["pair_index"] for item in pairs]
        axes[0].plot(
            x, [item["hot_ns_per_traversal"] / 1000 for item in pairs],
            marker=markers["hot"], linestyle="-", color=colors[cpu],
            label=f"CPU{cpu} hot (250 traversals)",
        )
        axes[0].plot(
            x, [item["cold_ns_per_traversal"] / 1000 for item in pairs],
            marker=markers["cold"], linestyle="--", color=colors[cpu],
            label=f"CPU{cpu} cold-conditioned (1 traversal)",
        )
        penalties = [item["cold_hot_penalty"] for item in pairs]
        axes[1].plot(x, penalties, marker="o", color=colors[cpu], label=f"CPU{cpu} pairs")
        median = analysis["cpus"][str(cpu)]["median_pair_cold_hot_penalty"]
        axes[1].axhline(
            median, color=colors[cpu], linestyle="--", alpha=0.65,
            label=f"CPU{cpu} median {median:.3f}×",
        )
    invalid_labeled = False
    for issue in analysis.get("invalid_runs", []):
        timing = issue.get("diagnostic_timing")
        if not isinstance(timing, Mapping):
            continue
        axes[0].scatter(
            [timing["pair_index"]], [timing["ns_per_traversal"] / 1000],
            marker="X", s=95, linewidths=1.3, edgecolors="#111827",
            color=colors[timing["cpu"]], zorder=6,
            label="invalid PMU timing qualifier" if not invalid_labeled else None,
        )
        invalid_labeled = True
    axes[0].set_title("Q1 64 KiB full_dotprod: normalized latency")
    axes[0].set_xlabel("Alternating pair index")
    axes[0].set_ylabel("Microseconds per traversal (lower is faster)")
    axes[0].set_xticks(range(1, 6))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="#111827", linewidth=1, label="no penalty (1.0×)")
    axes[1].set_title("Paired cold/hot latency ratio")
    axes[1].set_xlabel("Alternating pair index")
    axes[1].set_ylabel("Cold / hot ns per traversal")
    axes[1].set_xticks(range(1, 6))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "E055 direct A733 microgate — raw pair dispersion is retained\n"
        + str(analysis["chart_annotation"]),
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(path)
    plt.close(fig)


def write_outputs(result_dir: Path) -> None:
    audit = audit_phase(result_dir / "raw")
    rows = audit["qualified_rows"]
    analysis = analyze_phase(audit)
    (result_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(result_dir / "samples.csv", rows)
    _write_chart(result_dir / "e055-paired-hot-cold.png", analysis)
    manifest = build_raw_manifest(result_dir)
    (result_dir / "raw-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path, nargs="?")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir is not None:
        if args.result_dir is not None:
            parser.error("result_dir and --run-dir are mutually exclusive")
        try:
            result = validate_run(args.run_dir.resolve())
        except (OSError, ValueError) as exc:
            print(json.dumps({
                "schema": "e055-direct-live-qualification/v1",
                "status": "FAIL",
                "reason": str(exc),
            }, sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.result_dir is None:
        parser.error("result_dir or --run-dir is required")
    write_outputs(args.result_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
