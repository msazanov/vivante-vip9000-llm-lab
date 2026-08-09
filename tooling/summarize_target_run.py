#!/usr/bin/env python3
"""Summarize an immutable profiling/thermal run bundle deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
import tempfile
from typing import Any, Iterator


MAX_SERIES = 256
MAX_GOVERNORS_PER_POLICY = 32
MAX_EVENT_TYPES = 128
STATS_BATCH_SIZE = 4096
SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class SummaryError(ValueError):
    """The run bundle is missing or does not match the consumed schema."""


class DiskStatsStore:
    """Exact statistics backed by a temporary SQLite file, not sample-sized RAM."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="vip9000-run-summary-")
        database = Path(self._temporary.name) / "stats.sqlite3"
        self._connection = sqlite3.connect(database)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute("PRAGMA cache_size=-2048")
        self._connection.execute("CREATE TABLE samples (series INTEGER NOT NULL, value INTEGER NOT NULL)")
        self._next_series = 0
        self._pending: list[tuple[int, int]] = []

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            self._temporary.cleanup()

    def __enter__(self) -> DiskStatsStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def new_series(self) -> int:
        if self._next_series >= MAX_SERIES:
            raise SummaryError(f"schema error: more than {MAX_SERIES} metric series")
        series = self._next_series
        self._next_series += 1
        return series

    def add(self, series: int, value: int, context: str) -> None:
        if not -(1 << 63) <= value < (1 << 63):
            raise SummaryError(f"schema error at {context}: integer is outside signed 64-bit range")
        self._pending.append((series, value))
        if len(self._pending) >= STATS_BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            self._connection.executemany("INSERT INTO samples VALUES (?, ?)", self._pending)
            self._pending.clear()

    def stats(self, series: int) -> dict[str, int | float]:
        self._flush()
        count, minimum, maximum = self._connection.execute(
            "SELECT COUNT(*), MIN(value), MAX(value) FROM samples WHERE series = ?",
            (series,),
        ).fetchone()
        if count == 0:
            return {}
        if count % 2:
            median: int | float = self._connection.execute(
                "SELECT value FROM samples WHERE series = ? ORDER BY value LIMIT 1 OFFSET ?",
                (series, count // 2),
            ).fetchone()[0]
        else:
            middle = self._connection.execute(
                "SELECT value FROM samples WHERE series = ? ORDER BY value LIMIT 2 OFFSET ?",
                (series, count // 2 - 1),
            ).fetchall()
            median = (middle[0][0] + middle[1][0]) / 2
            if median.is_integer():
                median = int(median)
        return {"min": minimum, "median": median, "max": maximum}


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


def _phase_path(run_dir: Path, metadata: dict[str, Any]) -> Path:
    files = _mapping(metadata.get("files"), "metadata.files")
    relative_text = _text(files.get("phases"), "metadata.files.phases")
    raw_parts = relative_text.split("/")
    if (
        relative_text.startswith("/")
        or "\\" in relative_text
        or any(part in ("", ".", "..") for part in raw_parts)
        or any(SAFE_PATH_COMPONENT.fullmatch(part) is None for part in raw_parts)
    ):
        raise SummaryError("schema error at metadata.files.phases: unsafe relative path")
    relative = PurePosixPath(relative_text)
    run_resolved = run_dir.resolve()
    candidate = (run_dir / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(run_resolved)
    except ValueError as error:
        raise SummaryError("schema error at metadata.files.phases: unsafe relative path") from error
    if not candidate.is_file():
        raise SummaryError(f"required input is missing: {relative_text}")
    if "phase_file" in metadata:
        phase_file = _mapping(metadata["phase_file"], "metadata.phase_file")
        declared_path = _text(phase_file.get("path"), "metadata.phase_file.path")
        if declared_path != relative_text:
            raise SummaryError("schema error: metadata.phase_file.path conflicts with metadata.files.phases")
        if "size_bytes" in phase_file:
            declared_size = _integer(phase_file["size_bytes"], "metadata.phase_file.size_bytes")
            if declared_size < 0 or declared_size != candidate.stat().st_size:
                raise SummaryError("schema error: metadata.phase_file.size_bytes does not match phases file")
    return candidate


def _consume_phases(path: Path) -> tuple[int, dict[str, int]]:
    count = 0
    events: dict[str, int] = {}
    previous_monotonic_ns: int | None = None
    for index, row in enumerate(_jsonl_rows(path), 1):
        context = f"{path.name}:{index}"
        event = _text(row.get("event"), f"{context}.event")
        monotonic_ns = _integer(row.get("monotonic_ns"), f"{context}.monotonic_ns")
        if monotonic_ns < 0:
            raise SummaryError(f"schema error at {context}.monotonic_ns: expected non-negative integer")
        if previous_monotonic_ns is not None and monotonic_ns < previous_monotonic_ns:
            raise SummaryError(f"schema error at {context}.monotonic_ns: timestamps are not monotonic")
        if "step" in row:
            _integer(row["step"], f"{context}.step")
        previous_monotonic_ns = monotonic_ns
        if event not in events and len(events) >= MAX_EVENT_TYPES:
            raise SummaryError(f"schema error: more than {MAX_EVENT_TYPES} phase event types")
        events[event] = events.get(event, 0) + 1
        count += 1
    if count == 0:
        raise SummaryError(f"required phase evidence is empty: {path.name}")
    return count, events


def _append_stats(
    store: DiskStatsStore,
    target: dict[str, int],
    key: str,
    value: Any,
    context: str,
) -> None:
    if key not in target:
        target[key] = store.new_series()
    store.add(target[key], _integer(value, context), context)


def _consume_thermal_zones(
    value: Any,
    store: DiskStatsStore,
    thermal_values: dict[str, int],
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
        _append_stats(store, thermal_values, zone_type, temperature, f"{context}.{name}.millidegrees_c")


def _consume_policies(
    value: Any,
    store: DiskStatsStore,
    policy_values: dict[str, dict[str, Any]],
    context: str,
) -> None:
    if not isinstance(value, list):
        raise SummaryError(f"schema error at {context}: expected array")
    for index, entry in enumerate(value):
        policy = _mapping(entry, f"{context}[{index}]")
        path = _text(policy.get("path"), f"{context}[{index}].path")
        if path not in policy_values:
            policy_values[path] = {
                "cur_khz": store.new_series(),
                "max_khz": store.new_series(),
                "governors": set(),
            }
        bucket = policy_values[path]
        store.add(
            bucket["cur_khz"],
            _integer(policy.get("cur_khz"), f"{context}[{index}].cur_khz"),
            f"{context}[{index}].cur_khz",
        )
        store.add(
            bucket["max_khz"],
            _integer(policy.get("max_khz"), f"{context}[{index}].max_khz"),
            f"{context}[{index}].max_khz",
        )
        bucket["governors"].add(_text(policy.get("governor"), f"{context}[{index}].governor"))
        if len(bucket["governors"]) > MAX_GOVERNORS_PER_POLICY:
            raise SummaryError(
                f"schema error at {context}[{index}].governor: more than "
                f"{MAX_GOVERNORS_PER_POLICY} distinct values"
            )


def _consume_cooling(
    value: Any,
    store: DiskStatsStore,
    cooling_values: dict[str, int],
    context: str,
) -> None:
    if not isinstance(value, list):
        raise SummaryError(f"schema error at {context}: expected array")
    for index, entry in enumerate(value):
        device = _mapping(entry, f"{context}[{index}]")
        device_type = _text(device.get("type"), f"{context}[{index}].type")
        _append_stats(
            store,
            cooling_values,
            device_type,
            device.get("cur_state"),
            f"{context}[{index}].cur_state",
        )


def _consume_process(value: Any, process_peaks: dict[str, int], context: str) -> None:
    process = _mapping(value, context)
    for field, output in (
        ("rss_kib", "peak_rss_kib"),
        ("rss_hwm_kib", "peak_rss_hwm_kib"),
        ("threads", "peak_threads"),
    ):
        if field in process and process[field] is not None:
            observed = _integer(process[field], f"{context}.{field}")
            process_peaks[output] = max(process_peaks.get(output, observed), observed)


def _consume_memory(
    value: Any,
    aggregate: dict[str, int],
    context: str,
) -> None:
    memory = _mapping(value, context)
    if "MemAvailable_kib" in memory:
        available = _integer(memory["MemAvailable_kib"], f"{context}.MemAvailable_kib")
        aggregate["mem_available_kib_min"] = min(
            aggregate.get("mem_available_kib_min", available), available
        )
    total = memory.get("SwapTotal_kib")
    free = memory.get("SwapFree_kib")
    used = memory.get("SwapUsed_kib")
    if total is not None:
        total = _integer(total, f"{context}.SwapTotal_kib")
    if free is not None:
        free = _integer(free, f"{context}.SwapFree_kib")
        aggregate.setdefault("swap_free_kib_first", free)
        aggregate["swap_free_kib_last"] = free
        aggregate["swap_free_kib_min"] = min(aggregate.get("swap_free_kib_min", free), free)
    if used is not None:
        used = _integer(used, f"{context}.SwapUsed_kib")
    elif total is not None and free is not None:
        if free > total:
            raise SummaryError(f"schema error at {context}: SwapFree_kib exceeds SwapTotal_kib")
        used = total - free
    if used is not None:
        aggregate.setdefault("swap_used_kib_first", used)
        aggregate["swap_used_kib_last"] = used
        aggregate["swap_used_kib_peak"] = max(aggregate.get("swap_used_kib_peak", used), used)


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
    phase_count, phase_events = _consume_phases(_phase_path(run_dir, metadata))

    with DiskStatsStore() as stats_store:
        thermal_values: dict[str, int] = {}
        policy_values: dict[str, dict[str, Any]] = {}
        cooling_values: dict[str, int] = {}
        profiler_process_peaks: dict[str, int] = {}
        workload_process_peaks: dict[str, int] = {}
        memory_aggregate: dict[str, int] = {}
        guard_events: dict[str, int] = {}
        telemetry_count = 0
        guard_sample_count = 0

        for index, row in enumerate(_jsonl_rows(required["telemetry.jsonl"])):
            telemetry_count += 1
            context = f"telemetry.jsonl:{index + 1}"
            system = _mapping(row.get("system"), f"{context}.system")
            if "thermal_zones" in system:
                _consume_thermal_zones(
                    system["thermal_zones"],
                    stats_store,
                    thermal_values,
                    f"{context}.system.thermal_zones",
                )
            if "cpu_policies" in system:
                _consume_policies(
                    system["cpu_policies"],
                    stats_store,
                    policy_values,
                    f"{context}.system.cpu_policies",
                )
            if "cooling_devices" in system:
                _consume_cooling(
                    system["cooling_devices"],
                    stats_store,
                    cooling_values,
                    f"{context}.system.cooling_devices",
                )
            if "memory" in system:
                _consume_memory(system["memory"], memory_aggregate, f"{context}.system.memory")
            if "process" in row:
                _consume_process(row["process"], profiler_process_peaks, f"{context}.process")

        for index, row in enumerate(_jsonl_rows(required["thermal-guard.jsonl"])):
            context = f"thermal-guard.jsonl:{index + 1}"
            if "event" in row:
                event = _text(row["event"], f"{context}.event")
                if event not in guard_events and len(guard_events) >= MAX_EVENT_TYPES:
                    raise SummaryError(f"schema error: more than {MAX_EVENT_TYPES} guard event types")
                guard_events[event] = guard_events.get(event, 0) + 1
                continue
            if row.get("kind", "sample") != "sample":
                raise SummaryError(f"schema error at {context}.kind: expected sample")
            guard_sample_count += 1
            _consume_thermal_zones(
                row.get("thermal_zones"), stats_store, thermal_values, f"{context}.thermal_zones"
            )
            if "cpu_policies" in row:
                _consume_policies(
                    row["cpu_policies"], stats_store, policy_values, f"{context}.cpu_policies"
                )
            if "cooling_devices" in row:
                _consume_cooling(
                    row["cooling_devices"], stats_store, cooling_values, f"{context}.cooling_devices"
                )
            if "memory" in row:
                _consume_memory(row["memory"], memory_aggregate, f"{context}.memory")
            if "process" in row:
                _consume_process(row["process"], workload_process_peaks, f"{context}.process")

        memory_summary = {
            key: value
            for key, value in memory_aggregate.items()
            if key in ("mem_available_kib_min", "swap_free_kib_min", "swap_used_kib_peak")
        }
        if "swap_used_kib_first" in memory_aggregate:
            memory_summary["swap_used_kib_delta"] = (
                memory_aggregate["swap_used_kib_last"] - memory_aggregate["swap_used_kib_first"]
            )
        elif "swap_free_kib_first" in memory_aggregate:
            memory_summary["swap_used_kib_delta"] = (
                memory_aggregate["swap_free_kib_first"] - memory_aggregate["swap_free_kib_last"]
            )

        return {
            "run_id": run_id,
            "exit_code": exit_code,
            "elapsed_ns": elapsed_ns,
            "sample_counts": {
                "telemetry": telemetry_count,
                "thermal_guard": guard_sample_count,
                "phases": phase_count,
            },
            "profiler_child": profiler_process_peaks,
            "workload_child": workload_process_peaks,
            "thermal_by_type": {
                key: stats_store.stats(series) for key, series in sorted(thermal_values.items())
            },
            "cpu_policies": {
                path: {
                    "cur_khz": stats_store.stats(bucket["cur_khz"]),
                    "max_khz": stats_store.stats(bucket["max_khz"]),
                    "governors": sorted(bucket["governors"]),
                }
                for path, bucket in sorted(policy_values.items())
            },
            "cooling_by_type": {
                key: stats_store.stats(series) for key, series in sorted(cooling_values.items())
            },
            "memory": memory_summary,
            "thermal_guard_events": {key: guard_events[key] for key in sorted(guard_events)},
            "phase_events": {key: phase_events[key] for key in sorted(phase_events)},
        }


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
    except (OSError, sqlite3.Error, SummaryError) as error:
        print(f"summarize_target_run: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
