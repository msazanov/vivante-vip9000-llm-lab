#!/usr/bin/env python3
"""Summarize an immutable profiling/thermal run bundle deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Iterator


class SummaryError(ValueError):
    """The run bundle is missing or does not match the consumed schema."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"schema error at {context}: expected object")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"schema error at {context}: expected integer")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError(f"schema error at {context}: expected non-empty string")
    return value


def _json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"malformed JSON: {path}") from error
    return _mapping(value, str(path))


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise SummaryError(f"malformed JSON at {path}:{line_number}: blank line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SummaryError(f"malformed JSON at {path}:{line_number}") from error
                yield _mapping(value, f"{path}:{line_number}")
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot read {path}") from error


def _stats(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    median = statistics.median(ordered)
    if isinstance(median, float) and median.is_integer():
        median = int(median)
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def _append_stats(target: dict[str, list[int]], key: str, value: Any, context: str) -> None:
    target.setdefault(key, []).append(_integer(value, context))


def _consume_thermal_zones(
    value: Any,
    thermal_values: dict[str, list[int]],
    context: str,
) -> None:
    if isinstance(value, dict):
        entries = value.items()
    elif isinstance(value, list):
        entries = enumerate(value)
    else:
        raise SummaryError(f"schema error at {context}: expected object or array")
    for name, entry in entries:
        zone = _mapping(entry, f"{context}.{name}")
        zone_type = _text(zone.get("type"), f"{context}.{name}.type")
        temperature = _integer(zone.get("millidegrees_c"), f"{context}.{name}.millidegrees_c")
        _append_stats(thermal_values, zone_type, temperature, f"{context}.{name}.millidegrees_c")


def _consume_policies(value: Any, policy_values: dict[str, dict[str, Any]], context: str) -> None:
    if not isinstance(value, list):
        raise SummaryError(f"schema error at {context}: expected array")
    for index, entry in enumerate(value):
        policy = _mapping(entry, f"{context}[{index}]")
        path = _text(policy.get("path"), f"{context}[{index}].path")
        bucket = policy_values.setdefault(path, {"cur_khz": [], "max_khz": [], "governors": set()})
        bucket["cur_khz"].append(_integer(policy.get("cur_khz"), f"{context}[{index}].cur_khz"))
        bucket["max_khz"].append(_integer(policy.get("max_khz"), f"{context}[{index}].max_khz"))
        bucket["governors"].add(_text(policy.get("governor"), f"{context}[{index}].governor"))


def _consume_cooling(value: Any, cooling_values: dict[str, list[int]], context: str) -> None:
    if not isinstance(value, list):
        raise SummaryError(f"schema error at {context}: expected array")
    for index, entry in enumerate(value):
        device = _mapping(entry, f"{context}[{index}]")
        device_type = _text(device.get("type"), f"{context}[{index}].type")
        _append_stats(cooling_values, device_type, device.get("cur_state"), f"{context}[{index}].cur_state")


def _consume_process(value: Any, process_values: dict[str, list[int]], context: str) -> None:
    process = _mapping(value, context)
    for field in ("rss_kib", "rss_hwm_kib", "threads"):
        if field in process and process[field] is not None:
            _append_stats(process_values, field, process[field], f"{context}.{field}")


def _consume_memory(
    value: Any,
    mem_available: list[int],
    swap_used: list[int],
    swap_free: list[int],
    context: str,
) -> None:
    memory = _mapping(value, context)
    if "MemAvailable_kib" in memory:
        mem_available.append(_integer(memory["MemAvailable_kib"], f"{context}.MemAvailable_kib"))
    total = memory.get("SwapTotal_kib")
    free = memory.get("SwapFree_kib")
    used = memory.get("SwapUsed_kib")
    if total is not None:
        total = _integer(total, f"{context}.SwapTotal_kib")
    if free is not None:
        free = _integer(free, f"{context}.SwapFree_kib")
        swap_free.append(free)
    if used is not None:
        swap_used.append(_integer(used, f"{context}.SwapUsed_kib"))
    elif total is not None and free is not None:
        if free > total:
            raise SummaryError(f"schema error at {context}: SwapFree_kib exceeds SwapTotal_kib")
        swap_used.append(total - free)


def _peak_process(process_values: dict[str, list[int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for source, output in (
        ("rss_kib", "peak_rss_kib"),
        ("rss_hwm_kib", "peak_rss_hwm_kib"),
        ("threads", "peak_threads"),
    ):
        if source in process_values:
            result[output] = max(process_values[source])
    return result


def summarize_run(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise SummaryError(f"run directory does not exist: {run_dir}")
    required = {name: run_dir / name for name in ("metadata.json", "telemetry.jsonl", "thermal-guard.jsonl")}
    for name, path in required.items():
        if not path.is_file():
            raise SummaryError(f"required input is missing: {name}")

    metadata = _json_file(required["metadata.json"])
    schema_version = _integer(metadata.get("schema_version"), "metadata.schema_version")
    if schema_version != 1:
        raise SummaryError("schema error at metadata.schema_version: expected 1")
    run_id = _text(metadata.get("run_id"), "metadata.run_id")
    if run_id != run_dir.name:
        raise SummaryError("schema error: metadata.run_id does not match run directory name")
    elapsed_ns = _integer(metadata.get("elapsed_ns"), "metadata.elapsed_ns")
    exit_code = _integer(metadata.get("exit_code"), "metadata.exit_code")
    if elapsed_ns < 0:
        raise SummaryError("schema error at metadata.elapsed_ns: expected non-negative integer")

    thermal_values: dict[str, list[int]] = {}
    policy_values: dict[str, dict[str, Any]] = {}
    cooling_values: dict[str, list[int]] = {}
    profiler_process_values: dict[str, list[int]] = {}
    workload_process_values: dict[str, list[int]] = {}
    mem_available: list[int] = []
    swap_used: list[int] = []
    swap_free: list[int] = []
    guard_events: dict[str, int] = {}
    telemetry_count = 0
    guard_sample_count = 0

    for index, row in enumerate(_jsonl_rows(required["telemetry.jsonl"])):
        telemetry_count += 1
        context = f"telemetry.jsonl:{index + 1}"
        system = _mapping(row.get("system"), f"{context}.system")
        if "thermal_zones" in system:
            _consume_thermal_zones(system["thermal_zones"], thermal_values, f"{context}.system.thermal_zones")
        if "cpu_policies" in system:
            _consume_policies(system["cpu_policies"], policy_values, f"{context}.system.cpu_policies")
        if "cooling_devices" in system:
            _consume_cooling(system["cooling_devices"], cooling_values, f"{context}.system.cooling_devices")
        if "memory" in system:
            _consume_memory(
                system["memory"],
                mem_available,
                swap_used,
                swap_free,
                f"{context}.system.memory",
            )
        if "process" in row:
            _consume_process(
                row["process"], profiler_process_values, f"{context}.process"
            )

    for index, row in enumerate(_jsonl_rows(required["thermal-guard.jsonl"])):
        context = f"thermal-guard.jsonl:{index + 1}"
        if "event" in row:
            event = _text(row["event"], f"{context}.event")
            guard_events[event] = guard_events.get(event, 0) + 1
            continue
        if row.get("kind", "sample") != "sample":
            raise SummaryError(f"schema error at {context}.kind: expected sample")
        guard_sample_count += 1
        _consume_thermal_zones(row.get("thermal_zones"), thermal_values, f"{context}.thermal_zones")
        if "cpu_policies" in row:
            _consume_policies(row["cpu_policies"], policy_values, f"{context}.cpu_policies")
        if "cooling_devices" in row:
            _consume_cooling(row["cooling_devices"], cooling_values, f"{context}.cooling_devices")
        if "memory" in row:
            _consume_memory(
                row["memory"], mem_available, swap_used, swap_free, f"{context}.memory"
            )
        if "process" in row:
            _consume_process(
                row["process"], workload_process_values, f"{context}.process"
            )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "exit_code": exit_code,
        "elapsed_ns": elapsed_ns,
        "sample_counts": {
            "telemetry": telemetry_count,
            "thermal_guard": guard_sample_count,
        },
        "profiler_child": _peak_process(profiler_process_values),
        "workload_child": _peak_process(workload_process_values),
        "thermal_by_type": {key: _stats(values) for key, values in sorted(thermal_values.items())},
        "cpu_policies": {
            path: {
                "cur_khz": _stats(bucket["cur_khz"]),
                "max_khz": _stats(bucket["max_khz"]),
                "governors": sorted(bucket["governors"]),
            }
            for path, bucket in sorted(policy_values.items())
        },
        "cooling_by_type": {key: _stats(values) for key, values in sorted(cooling_values.items())},
        "memory": {},
        "thermal_guard_events": {key: guard_events[key] for key in sorted(guard_events)},
    }
    if mem_available:
        summary["memory"]["mem_available_kib_min"] = min(mem_available)
    if swap_free:
        summary["memory"]["swap_free_kib_min"] = min(swap_free)
    if swap_used:
        summary["memory"]["swap_used_kib_delta"] = swap_used[-1] - swap_used[0]
        summary["memory"]["swap_used_kib_peak"] = max(swap_used)
    elif swap_free:
        summary["memory"]["swap_used_kib_delta"] = swap_free[0] - swap_free[-1]
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an immutable target run bundle")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="exclusive output file; stdout is used by default")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = summarize_run(args.run_dir)
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
    except FileExistsError as error:
        print(f"summarize_target_run: refusing to overwrite output: {error.filename}", file=sys.stderr)
        return 2
    except (OSError, SummaryError) as error:
        print(f"summarize_target_run: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
