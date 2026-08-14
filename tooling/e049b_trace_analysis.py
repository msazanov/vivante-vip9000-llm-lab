#!/usr/bin/env python3
"""E049b post-processing for the published E048 per-op traces.

This module deliberately performs all expensive work off the decode hot path:
it validates the deterministic zstd publication manifest, streams each JSONL
trace, joins worker events by node, and reports wall intervals rather than a
sum of parallel worker durations.  The resulting attribution is therefore
``TRACE_ONLY`` evidence: it explains the captured trace, but does not claim a
new inference speed or a direct DDR measurement.

The event format is E048 ``e048-trace/v1``.  E048 did not include stable
allocation identities or a direct DDR PMU counter; this analyzer consequently
keeps observed DDR read/write fields as ``None`` and labels all byte totals as
logical descriptors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from tooling.e048_trace import TRACE_SCHEMA, validate_trace_events
except ModuleNotFoundError:  # direct ``python tooling/e049b_trace_analysis.py``
    from e048_trace import TRACE_SCHEMA, validate_trace_events


ANALYZER_SCHEMA = "e049b-trace-analysis/v1"
PACKED_MANIFEST_SCHEMA = "e048-compat-ab-packed/v1"
EXPECTED_PAIRS = tuple(f"pair{i:02d}" for i in range(1, 6))
Q1_VARIANT_RE = re.compile(r"q1_0_4x(?P<width>\d+)_q8_0")


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge non-empty half-open timestamp intervals.

    Worker spans overlap because the CPU executes one graph node in parallel.
    The union, rather than a sum of worker durations, is the wall-time
    quantity used by this experiment.
    """

    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        old_start, old_end = merged[-1]
        if start <= old_end:
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    """Return duration of an already merged (or mergeable) interval set."""

    return sum(end - start for start, end in merge_intervals(intervals))


def subtract_intervals(
    base: Iterable[tuple[int, int]],
    subtract: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract interval union ``subtract`` from interval union ``base``."""

    result: list[tuple[int, int]] = []
    cuts = merge_intervals(subtract)
    for start, end in merge_intervals(base):
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                result.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return [(start, end) for start, end in result if end > start]


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, matching the E048 convention."""

    if not values:
        raise ValueError("percentile требует непустой выборки")
    if not 0 <= p <= 1:
        raise ValueError("percentile должен быть между 0 и 1")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_samples(values: Sequence[float]) -> dict[str, Any]:
    """Summarize one value per comparable run.

    E049b has five runs.  ``cv`` is population standard deviation divided by
    the mean; the raw samples remain in the machine-readable result.
    """

    if not values:
        raise ValueError("нельзя суммировать пустую выборку")
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    stdev = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
    return {
        "count": len(numeric),
        "samples": numeric,
        "mean": mean,
        "median": statistics.median(numeric),
        "p10": percentile(numeric, 0.10),
        "p90": percentile(numeric, 0.90),
        "stdev_population": stdev,
        "cv": (stdev / mean) if mean else None,
    }


def family_for_q1_variant(variant: str | None) -> str:
    """Map the trace variant to an explicit Q1 GEMV family."""

    match = Q1_VARIANT_RE.fullmatch(variant or "")
    if not match:
        return "q1_gemv_unknown"
    return f"q1_gemv_4x{match.group('width')}"


def _canonical_node_family(op_name: str | None, fused_nodes: int) -> str:
    op = (op_name or "UNKNOWN").upper()
    if fused_nodes > 1:
        return f"fused:{op}"
    return op


def _node_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row.get("token_seq", 0)), int(row.get("node_index", -1))


def _validate_event_clock(event: Mapping[str, Any], errors: list[str]) -> None:
    kind = event.get("event")
    if kind not in {"node_worker", "q1_kernel", "phase_worker"}:
        return
    start = event.get("start_ns")
    end = event.get("end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        errors.append(f"{kind} имеет неверный интервал")


def _validate_run_structure(events: Sequence[dict[str, Any]]) -> list[str]:
    """Validate E048 events plus token and overflow invariants."""

    errors = list(validate_trace_events(events))
    begins: dict[int, dict[str, Any]] = {}
    ends: dict[int, dict[str, Any]] = {}
    for event in events:
        _validate_event_clock(event, errors)
        seq = int(event.get("token_seq", 0))
        if event.get("event") == "token_begin":
            if seq in begins:
                errors.append(f"token {seq}: повторный token_begin")
            begins[seq] = event
        elif event.get("event") == "token_end":
            if seq in ends:
                errors.append(f"token {seq}: повторный token_end")
            ends[seq] = event
            if event.get("overflow_count", 0) != 0:
                errors.append(f"token {seq}: overflow_count не равен нулю")
    if not begins:
        errors.append("нет token_begin")
    if set(begins) != set(ends):
        errors.append("множества token_begin и token_end различаются")
    return errors


def _as_interval(event: Mapping[str, Any], start_field: str, end_field: str) -> tuple[int, int]:
    return int(event[start_field]), int(event[end_field])


def aggregate_run_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one E048 run into token, node and logical-byte rows.

    ``categories_ns`` is a disjoint partition of the token wall interval:

    * ``q1_gemv`` — union of all Q1 kernel spans (parallel workers merged);
    * ``f32_to_q8`` — union of explicit conversion phase spans;
    * ``other_compute`` — worker compute union after removing the two explicit
      kernel classes;
    * ``barrier_sync`` — worker ``end→post_barrier_end`` waits, after removing
      compute classes;
    * ``unattributed`` — token gaps not represented by E048 events.

    Node rows retain the wall interval and the sum of worker compute times for
    diagnostic contrast.  The latter is *not* used for latency aggregation.
    """

    errors = _validate_run_structure(events)
    if errors:
        raise ValueError("; ".join(errors))

    tokens: dict[int, dict[str, Any]] = {}
    node_workers: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    q1_by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    phase_by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        seq = int(event.get("token_seq", 0))
        kind = event["event"]
        if kind == "token_begin":
            tokens.setdefault(seq, {})["begin"] = event
        elif kind == "token_end":
            tokens.setdefault(seq, {})["end"] = event
        elif kind == "node_worker":
            node_workers[(seq, int(event["node_index"]))].append(event)
        elif kind == "q1_kernel":
            q1_by_node[(seq, int(event["node_index"]))].append(event)
        elif kind == "phase_worker":
            phase_by_node[(seq, int(event["node_index"]))].append(event)

    node_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for seq in sorted(tokens):
        begin = tokens[seq]["begin"]
        end = tokens[seq]["end"]
        token_start = int(begin["start_ns"])
        token_end = int(end["end_ns"])
        compute_intervals: list[tuple[int, int]] = []
        barrier_intervals: list[tuple[int, int]] = []
        q1_intervals: list[tuple[int, int]] = []
        phase_intervals: list[tuple[int, int]] = []
        q1_variants: dict[str, list[tuple[int, int]]] = defaultdict(list)
        logical_q1 = logical_q8 = logical_q8_unique_descriptor = output_bytes = 0

        seq_node_keys = sorted(key for key in node_workers if key[0] == seq)
        for key in seq_node_keys:
            workers = node_workers[key]
            first = workers[0]
            wall_start = min(int(row["start_ns"]) for row in workers)
            wall_end = max(int(row["post_barrier_end_ns"]) for row in workers)
            worker_compute = [_as_interval(row, "start_ns", "end_ns") for row in workers]
            worker_barrier = [
                (int(row["end_ns"]), int(row["post_barrier_end_ns"]))
                for row in workers
                if int(row["post_barrier_end_ns"]) > int(row["end_ns"])
            ]
            compute_intervals.extend(worker_compute)
            barrier_intervals.extend(worker_barrier)
            q1_rows = q1_by_node.get(key, [])
            phase_rows = phase_by_node.get(key, [])
            node_q1 = []
            node_phase = []
            node_q1_by_variant: dict[str, list[tuple[int, int]]] = defaultdict(list)
            node_logical_q1 = node_logical_q8 = 0
            node_logical_q8_unique_descriptor = node_output_bytes = 0
            for row in q1_rows:
                interval = _as_interval(row, "start_ns", "end_ns")
                node_q1.append(interval)
                q1_intervals.append(interval)
                family = family_for_q1_variant(row.get("variant"))
                q1_variants[family].append(interval)
                node_q1_by_variant[family].append(interval)
                logical_q1 += int(row.get("q1_packed_weight_read_bytes", 0))
                logical_q8 += int(row.get("q8_activation_logical_read_bytes", 0))
                logical_q8_unique_descriptor += int(row.get("q8_activation_unique_read_bytes", 0))
                output_bytes += int(row.get("output_write_bytes", 0))
                node_logical_q1 += int(row.get("q1_packed_weight_read_bytes", 0))
                node_logical_q8 += int(row.get("q8_activation_logical_read_bytes", 0))
                node_logical_q8_unique_descriptor += int(row.get("q8_activation_unique_read_bytes", 0))
                node_output_bytes += int(row.get("output_write_bytes", 0))
            for row in phase_rows:
                interval = _as_interval(row, "start_ns", "end_ns")
                node_phase.append(interval)
                phase_intervals.append(interval)
            node_rows.append({
                "token_seq": seq,
                "node_index": int(first["node_index"]),
                "layer_index": first.get("layer_index"),
                "op_name": first.get("op_name"),
                "tensor_id": first.get("tensor_id"),
                "fused_nodes": int(first.get("fused_nodes") or 1),
                "fused": int(first.get("fused_nodes") or 1) > 1,
                "family": _canonical_node_family(first.get("op_name"), int(first.get("fused_nodes") or 1)),
                "workers": len(workers),
                "wall_start_ns": wall_start,
                "wall_end_ns": wall_end,
                "wall_duration_ns": wall_end - wall_start,
                "worker_sum_duration_ns": sum(end - start for start, end in worker_compute),
                "worker_compute_wall_ns": interval_duration(worker_compute),
                "barrier_wall_ns": interval_duration(worker_barrier),
                "q1_wall_ns": interval_duration(node_q1),
                "f32_to_q8_wall_ns": interval_duration(node_phase),
                "q1_variants_ns": {
                    family: interval_duration(spans)
                    for family, spans in sorted(node_q1_by_variant.items())
                },
                "logical_read_bytes": sum(int(row.get("logical_read_bytes", 0)) for row in workers),
                "logical_write_bytes": sum(int(row.get("logical_write_bytes", 0)) for row in workers),
                "unique_read_bytes_descriptor": sum(int(row.get("unique_read_bytes", 0)) for row in workers),
                "unique_write_bytes_descriptor": sum(int(row.get("unique_write_bytes", 0)) for row in workers),
                "q1_logical_read_bytes": node_logical_q1,
                "q8_logical_read_bytes": node_logical_q8,
                "q8_unique_read_bytes_descriptor": node_logical_q8_unique_descriptor,
                "q1_output_write_bytes": node_output_bytes,
            })

        # Make a disjoint category partition. Q1 takes precedence over an
        # explicitly recorded conversion phase, then other compute, then
        # barrier/synchronisation. This avoids double counting nested spans.
        q1_union = merge_intervals(q1_intervals)
        phase_exclusive = subtract_intervals(phase_intervals, q1_union)
        explicit_compute = merge_intervals([*q1_union, *phase_exclusive])
        compute_union = merge_intervals(compute_intervals)
        other_compute = subtract_intervals(compute_union, explicit_compute)
        barrier_exclusive = subtract_intervals(barrier_intervals, [*explicit_compute, *other_compute])
        covered = merge_intervals([*q1_union, *phase_exclusive, *other_compute, *barrier_exclusive])
        token_interval = [(token_start, token_end)]
        unattributed = subtract_intervals(token_interval, covered)
        categories = {
            "q1_gemv": interval_duration(q1_union),
            "f32_to_q8": interval_duration(phase_exclusive),
            "other_compute": interval_duration(other_compute),
            "barrier_sync": interval_duration(barrier_exclusive),
            "unattributed": interval_duration(unattributed),
        }
        variant_counts: dict[str, int] = defaultdict(int)
        for (q1_token, _q1_node), q1_rows_for_node in q1_by_node.items():
            if q1_token != seq:
                continue
            for q1_row in q1_rows_for_node:
                variant_counts[family_for_q1_variant(q1_row.get("variant"))] += 1
        token_rows.append({
            "token_seq": seq,
            "steady_decode": bool(begin.get("steady_decode", False)),
            "n_tokens": begin.get("n_tokens"),
            "start_ns": token_start,
            "end_ns": token_end,
            "duration_ns": token_end - token_start,
            "categories_ns": categories,
            "logical_bytes": {
                "q1_packed_weight_read_bytes": logical_q1,
                "q8_activation_logical_read_bytes": logical_q8,
                "q8_activation_unique_read_bytes_descriptor": logical_q8_unique_descriptor,
                "q1_output_write_bytes": output_bytes,
            },
            "observed_ddr_read_bytes": None,
            "observed_ddr_write_bytes": None,
            "counter_method": "none",
            "q1_variant_event_counts": dict(sorted(variant_counts.items())),
        })

    return {"tokens": token_rows, "nodes": node_rows}


def _iter_zstd_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        for line_number, line in enumerate(process.stdout, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSONL parse error: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: event не объект")
            yield value
    finally:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd завершился с RC={return_code} для {path}: {stderr.strip()}")


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Decompress one published trace and retain parsed events for analysis."""

    return list(_iter_zstd_jsonl(path))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _decompress_bytes(path: Path) -> bytes:
    process = subprocess.run(
        ["zstd", "-q", "-dc", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(f"zstd manifest file failed for {path}: {process.stderr.decode(errors='replace')}")
    return process.stdout


def read_packed_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the E048 TSV manifest without trusting its comments."""

    rows: list[dict[str, Any]] = []
    columns: list[str] | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        if raw.startswith("# columns="):
            columns = raw[len("# columns="):].split("\t")
            continue
        if raw.startswith("#"):
            continue
        if columns is None:
            raise ValueError(f"{path}:{line_number}: отсутствует columns header")
        fields = raw.split("\t")
        if len(fields) != len(columns):
            raise ValueError(f"{path}:{line_number}: число TSV полей не совпадает")
        row = dict(zip(columns, fields))
        for name in ("uncompressed_bytes", "compressed_bytes"):
            row[name] = int(row[name])
        rows.append(row)
    if columns != [
        "source_relative_path", "git_artifact", "uncompressed_sha256",
        "uncompressed_bytes", "compressed_sha256", "compressed_bytes",
    ]:
        raise ValueError(f"{path}: неизвестная схема packed manifest")
    if not rows:
        raise ValueError(f"{path}: manifest пуст")
    return rows


def validate_packed_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every compressed and decompressed artifact listed in the manifest."""

    rows = read_packed_manifest(manifest_path)
    failures: list[dict[str, Any]] = []
    checked_bytes = 0
    for row in rows:
        artifact = repo_root / row["git_artifact"]
        failure: dict[str, Any] = {"git_artifact": row["git_artifact"]}
        if not artifact.is_file():
            failure["error"] = "missing"
            failures.append(failure)
            continue
        compressed_sha, compressed_bytes = _sha256_file(artifact)
        checked_bytes += compressed_bytes
        if compressed_bytes != row["compressed_bytes"] or compressed_sha != row["compressed_sha256"]:
            failure["error"] = "compressed_hash_or_size_mismatch"
            failure["actual_compressed_sha256"] = compressed_sha
            failure["actual_compressed_bytes"] = compressed_bytes
            failures.append(failure)
            continue
        payload = _decompress_bytes(artifact)
        uncompressed_sha = _sha256_bytes(payload)
        if len(payload) != row["uncompressed_bytes"] or uncompressed_sha != row["uncompressed_sha256"]:
            failure["error"] = "uncompressed_hash_or_size_mismatch"
            failure["actual_uncompressed_sha256"] = uncompressed_sha
            failure["actual_uncompressed_bytes"] = len(payload)
            failures.append(failure)
    try:
        manifest_display = str(manifest_path.relative_to(repo_root))
    except ValueError:
        manifest_display = str(manifest_path)
    return {
        "schema": PACKED_MANIFEST_SCHEMA,
        "path": manifest_display,
        "entries": len(rows),
        "checked_compressed_bytes": checked_bytes,
        "valid": not failures,
        "failures": failures,
    }


def _load_trace_runs(repo_root: Path, trace_paths: Sequence[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in trace_paths:
        events = load_trace(path)
        errors = _validate_run_structure(events)
        if errors:
            raise ValueError(f"{path}: {'; '.join(errors)}")
        aggregate = aggregate_run_events(events)
        pair_match = re.search(r"pair(\d+)-on-e048-trace", path.name)
        pair = f"pair{int(pair_match.group(1)):02d}" if pair_match else path.stem
        steady = [row for row in aggregate["tokens"] if row["steady_decode"]]
        if not steady:
            raise ValueError(f"{path}: нет steady_decode token")
        try:
            path_display = str(path.relative_to(repo_root))
        except ValueError:
            path_display = str(path)
        runs.append({
            "pair": pair,
            "path": path_display,
            "events": len(events),
            "aggregate": aggregate,
            "steady_tokens": steady,
        })
    if len(runs) != 5:
        raise ValueError(f"ожидалось пять trace-on runs, найдено {len(runs)}")
    if {run["pair"] for run in runs} != set(EXPECTED_PAIRS):
        raise ValueError("trace-on pairs должны быть pair01..pair05")
    return sorted(runs, key=lambda row: row["pair"])


def _run_category_sample(run: Mapping[str, Any], category: str) -> float:
    representative = _representative_token(run)
    return float(representative["categories_ns"][category] / 1e6)


def _run_token_sample(run: Mapping[str, Any]) -> float:
    return float(_representative_token(run)["duration_ns"] / 1e6)


def _representative_token(run: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose the median-wall steady token so stacked categories sum exactly."""

    ordered = sorted(run["steady_tokens"], key=lambda row: int(row["duration_ns"]))
    return ordered[len(ordered) // 2]


def _run_bytes_sample(run: Mapping[str, Any], key: str) -> float:
    return float(statistics.median(
        row["logical_bytes"][key] for row in run["steady_tokens"]
    ))


def _build_category_rows(runs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = ("q1_gemv", "f32_to_q8", "other_compute", "barrier_sync", "unattributed")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for run in runs:
        row = {
            "run": run["pair"],
            "steady_token_count": len(run["steady_tokens"]),
            "token_wall_median_ms": _run_token_sample(run),
            "q1_packed_weight_read_bytes_per_token": _run_bytes_sample(
                run, "q1_packed_weight_read_bytes"
            ),
            "q8_activation_logical_read_bytes_per_token": _run_bytes_sample(
                run, "q8_activation_logical_read_bytes"
            ),
            "q8_activation_unique_read_bytes_descriptor_per_token": _run_bytes_sample(
                run, "q8_activation_unique_read_bytes_descriptor"
            ),
            "observed_ddr_read_bytes_per_token": None,
            "observed_ddr_write_bytes_per_token": None,
        }
        for category in categories:
            row[f"{category}_ms"] = _run_category_sample(run, category)
        rows.append(row)
    for key in ["token_wall_median_ms", *[f"{category}_ms" for category in categories]]:
        summary[key] = summarize_samples([row[key] for row in rows])
    for key in [
        "q1_packed_weight_read_bytes_per_token",
        "q8_activation_logical_read_bytes_per_token",
        "q8_activation_unique_read_bytes_descriptor_per_token",
    ]:
        summary[key] = summarize_samples([row[key] for row in rows])
    return rows, summary


def _group_node_rows(runs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make one median steady-token sample per node/group for each run."""

    per_run: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        steady_sequences = {
            int(token["token_seq"])
            for token in run["aggregate"]["tokens"]
            if token["steady_decode"]
        }
        for node in run["aggregate"]["nodes"]:
            if int(node["token_seq"]) not in steady_sequences:
                continue
            layer = node["layer_index"] if node["layer_index"] is not None else -1
            key = f"layer={layer}|op={node['op_name'] or 'UNKNOWN'}|family={node['family']}"
            grouped[key].append(node)
        for key, nodes in grouped.items():
            first = nodes[0]
            per_run[(run["pair"], key)] = {
                "run": run["pair"],
                "group": key,
                "layer_index": first["layer_index"] if first["layer_index"] is not None else -1,
                "op_name": first["op_name"] or "UNKNOWN",
                "family": first["family"],
                "fused": bool(first["fused"]),
                "fused_nodes": first["fused_nodes"],
                "node_count": len(nodes),
                "wall_median_ms": statistics.median(node["wall_duration_ns"] / 1e6 for node in nodes),
                "compute_wall_median_ms": statistics.median(node["worker_compute_wall_ns"] / 1e6 for node in nodes),
                "barrier_wall_median_ms": statistics.median(node["barrier_wall_ns"] / 1e6 for node in nodes),
                "logical_read_bytes_median": statistics.median(node["logical_read_bytes"] for node in nodes),
                "logical_write_bytes_median": statistics.median(node["logical_write_bytes"] for node in nodes),
                "q1_logical_read_bytes_median": statistics.median(node["q1_logical_read_bytes"] for node in nodes),
                "q8_logical_read_bytes_median": statistics.median(node["q8_logical_read_bytes"] for node in nodes),
                "q1_wall_median_ms": statistics.median(node["q1_wall_ns"] / 1e6 for node in nodes),
                "f32_to_q8_wall_median_ms": statistics.median(node["f32_to_q8_wall_ns"] / 1e6 for node in nodes),
            }
    groups = sorted({key for _, key in per_run})
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for key in groups:
        samples = [per_run[(run["pair"], key)] for run in runs]
        # Every graph node is expected in every comparable run. Missing nodes
        # are reported, never silently replaced by zero.
        if len(samples) != len(runs):
            raise ValueError(f"group {key} отсутствует в части пяти runs")
        first = samples[0]
        row = {
            "group": key,
            "layer_index": first["layer_index"],
            "op_name": first["op_name"],
            "family": first["family"],
            "fused": first["fused"],
            "fused_nodes": first["fused_nodes"],
            "run_samples": samples,
        }
        for field in (
            "wall_median_ms", "compute_wall_median_ms", "barrier_wall_median_ms",
            "logical_read_bytes_median", "logical_write_bytes_median",
            "q1_logical_read_bytes_median", "q8_logical_read_bytes_median",
            "q1_wall_median_ms", "f32_to_q8_wall_median_ms",
        ):
            row[field] = summarize_samples([sample[field] for sample in samples])
        summary[key] = row
        rows.append(row)
    return rows, summary


def _load_trace_off_summary(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    eval_data = data["metrics"]["eval"]
    off_ms = [float(value) for value in eval_data["off_ms"]]
    count = int(data["runs"]["off"][0]["timers"]["eval"]["count"])
    per_token = [value / count for value in off_ms]
    try:
        source_display = str(path.relative_to(repo_root)) if repo_root is not None else str(path)
    except ValueError:
        source_display = str(path)
    return {
        "source": source_display,
        "timer": "trace-off profile eval",
        "count_per_eval": count,
        "run_samples_ms": per_token,
        "stats": summarize_samples(per_token),
        "classification": "TRACE_ONLY",
        "note_ru": "Эта линия — trace-off whole-workload eval/число decode runs; она не является per-token hardware counter и используется только как диагностический A/B ориентир.",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    runs: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
    category_summary: Mapping[str, Any],
    node_rows: Sequence[Mapping[str, Any]],
    node_summary: Mapping[str, Any],
    trace_off: Mapping[str, Any],
) -> dict[str, Any]:
    """Write machine JSON/CSV and the two non-singleton diagnostic charts."""

    data_dir = output_dir / "data"
    chart_dir = output_dir / "charts"
    data_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(data_dir / "e049b-run-category.csv", category_rows)
    node_csv: list[dict[str, Any]] = []
    for row in node_rows:
        flat = {key: value for key, value in row.items() if key != "run_samples"}
        for field, stats in row.items():
            if field in {"run_samples"} or not isinstance(stats, Mapping):
                continue
            flat[f"{field}_median"] = stats.get("median")
            flat[f"{field}_p10"] = stats.get("p10")
            flat[f"{field}_p90"] = stats.get("p90")
            flat[f"{field}_cv"] = stats.get("cv")
        node_csv.append(flat)
    _write_csv(data_dir / "e049b-layer-op.csv", node_csv)

    q1_per_token = category_summary["q1_packed_weight_read_bytes_per_token"]["median"]
    q8_per_token = category_summary["q8_activation_logical_read_bytes_per_token"]["median"]
    q8_unique_per_token = category_summary[
        "q8_activation_unique_read_bytes_descriptor_per_token"
    ]["median"]
    q1_variants: dict[str, int] = defaultdict(int)
    for run in runs:
        for token in run["steady_tokens"]:
            for family, count in token["q1_variant_event_counts"].items():
                q1_variants[family] += int(count)
    summary = {
        "schema": ANALYZER_SCHEMA,
        "experiment_id": "E049b-trace-analysis",
        "classification": "TRACE_ONLY",
        "source_experiment": "E048-per-op-trace",
        "source_commit": "f5bd54eebbfc12c7399a64fd0f64ab56fa9172f5",
        "trace_runs": [
            {"pair": run["pair"], "path": run["path"], "events": run["events"],
             "steady_token_count": len(run["steady_tokens"])}
            for run in runs
        ],
        "manifest_validation": manifest,
        "steady_token_definition": "token_begin.steady_decode == true (token_seq 3..5 in each accepted E048 run)",
        "statistics": {
            "run_sample": "median of three steady tokens within each run",
            "across_runs": "five run samples",
            "percentile": "linear interpolation",
            "cv": "population standard deviation / mean",
        },
        "latency": {
            "category_summary": category_summary,
            "layer_op_summary": node_summary,
            "trace_off_reference": trace_off,
        },
        "logical_bytes_per_token": {
            "q1_packed_weight_read_bytes": {
                "value": q1_per_token,
                "decimal_gb": q1_per_token / 1e9,
                "binary_gib": q1_per_token / (1024 ** 3),
                "meaning_ru": "логическое чтение packed Q1-весов; это не измеренный DDR-трафик",
            },
            "q8_activation_logical_read_bytes": {
                "value": q8_per_token,
                "decimal_gb": q8_per_token / 1e9,
                "binary_gib": q8_per_token / (1024 ** 3),
                "meaning_ru": "логические повторные чтения активации для output-групп",
            },
            "q8_activation_unique_read_bytes_descriptor": {
                "value": q8_unique_per_token,
                "decimal_mb": q8_unique_per_token / 1e6,
                "status": "event_descriptor_only_unproven_at_experiment_level",
            },
            "observed_ddr_read_bytes": None,
            "observed_ddr_write_bytes": None,
            "counter_method": "none",
        },
        "phase_presence": {
            "q1_event_count": sum(
                sum(row["q1_variant_event_counts"].values())
                for run in runs for row in run["steady_tokens"]
            ),
            "q1_variant_event_counts": dict(sorted(q1_variants.items())),
            "f32_to_q8_event_count": sum(
                1 for run in runs for row in run["aggregate"]["nodes"]
                if row["token_seq"] in {
                    int(token["token_seq"]) for token in run["steady_tokens"]
                } and row["f32_to_q8_wall_ns"] > 0
            ),
            "f32_to_q8_status": "not_observed_in_e048_capture",
        },
        "charts": {
            "stacked_category_latency": "charts/e049b-category-latency-stack.png",
            "top_layer_op_xy": "charts/e049b-top-layer-op-xy.png",
        },
        "output_manifest": "data/e049b-output-manifest.json",
    }
    (data_dir / "e049b-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_charts(chart_dir, category_rows, node_rows, trace_off)
    output_files = [
        "data/e049b-summary.json",
        "data/e049b-run-category.csv",
        "data/e049b-layer-op.csv",
        "charts/e049b-category-latency-stack.png",
        "charts/e049b-top-layer-op-xy.png",
    ]
    output_manifest = {
        "schema": "e049b-output-manifest/v1",
        "source_experiment": "E048-per-op-trace",
        "files": [
            {
                "path": relative,
                "sha256": _sha256_file(output_dir / relative)[0],
                "size_bytes": _sha256_file(output_dir / relative)[1],
            }
            for relative in output_files
        ],
    }
    (data_dir / "e049b-output-manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _write_charts(
    chart_dir: Path,
    category_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    trace_off: Mapping[str, Any],
) -> None:
    """Create exactly two multi-sample charts; never plot a singleton result."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = (
        ("q1_gemv_ms", "Q1 GEMV", "#3366cc"),
        ("f32_to_q8_ms", "F32→Q8", "#dc3912"),
        ("other_compute_ms", "прочие вычисления", "#ff9900"),
        ("barrier_sync_ms", "barrier/sync", "#109618"),
        ("unattributed_ms", "неатрибутировано", "#999999"),
    )
    labels = [str(row["run"]) for row in category_rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    bottom = [0.0] * len(labels)
    for field, label, color in categories:
        values = [float(row[field]) for row in category_rows]
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom = [old + value for old, value in zip(bottom, values)]
    off_values = [float(value) for value in trace_off["run_samples_ms"]]
    if len(off_values) == len(labels):
        ax.plot(x, off_values, "k--o", linewidth=1.5, markersize=4,
                label="trace-off eval/3 (A/B reference; TRACE_ONLY)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("мс на steady-токен (median трёх токенов)")
    ax.set_title("E049b: разложение wall-time по пяти сопоставимым трассам")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.01, 0.01, "E048 trace-on; logical attribution, DDR bytes не измерены; TRACE_ONLY",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(chart_dir / "e049b-category-latency-stack.png")
    plt.close(fig)

    ranked_all = sorted(node_rows, key=lambda row: row["wall_median_ms"]["median"], reverse=True)
    ranked: list[Mapping[str, Any]] = []
    family_counts: dict[str, int] = defaultdict(int)
    # A plain top-15 list would contain only same-shaped MUL_MAT nodes and
    # collapse to one x coordinate. Keep two representatives per family so
    # the XY relationship remains informative and labels remain readable.
    for row in ranked_all:
        if row["wall_median_ms"]["median"] < 0.18:
            # Below 0.18 ms the labels form an unreadable cloud at this scale;
            # those rows remain in the machine CSV/JSON and are not discarded.
            continue
        family = str(row["family"])
        if family_counts[family] >= 2:
            continue
        ranked.append(row)
        family_counts[family] += 1
        if len(ranked) == 12:
            break
    fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
    for index, row in enumerate(ranked):
        x_value = row["logical_read_bytes_median"]["median"] / 1e9
        y_value = row["wall_median_ms"]["median"]
        label = f"L{row['layer_index']} {row['op_name']}"
        if row["fused"]:
            label += " [fused]"
        ax.scatter(x_value, y_value, s=42, alpha=0.85)
        offset_y = 5 + (index % 4) * 9
        ax.annotate(label, (x_value, y_value), xytext=(5, offset_y),
                    textcoords="offset points", fontsize=7,
                    arrowprops={"arrowstyle": "-", "linewidth": 0.35, "alpha": 0.45})
    ax.set_xlabel("логическое чтение узла, GB (не DDR measurement)")
    ax.set_ylabel("wall latency, мс (median по пяти run samples)")
    ax.set_title("E049b: top layer/op по wall latency — bytes против времени")
    ax.grid(alpha=0.25)
    fig.text(0.01, 0.01, "Каждая точка агрегирует пять run samples; worker durations не суммируются",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(chart_dir / "e049b-top-layer-op-xy.png")
    plt.close(fig)


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    experiment = repo_root / "experiments/E048-per-op-trace"
    output = repo_root / "experiments/E049b-trace-analysis"
    manifest = experiment / "data/compat-ab-packed-manifest.tsv"
    return experiment, output, manifest


def run_analysis(repo_root: Path, output_dir: Path, manifest_path: Path, *, skip_manifest: bool = False) -> dict[str, Any]:
    experiment_dir = repo_root / "experiments/E048-per-op-trace"
    trace_paths = [
        experiment_dir / "raw/compat-ab/compressed" / f"{pair}-on-e048-trace.jsonl.zst"
        for pair in EXPECTED_PAIRS
    ]
    missing = [str(path) for path in trace_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("отсутствуют trace: " + ", ".join(missing))
    manifest_result = {
        "schema": PACKED_MANIFEST_SCHEMA,
        "path": str(manifest_path),
        "valid": True,
        "skipped": True,
        "failures": [],
    }
    if not skip_manifest:
        manifest_result = validate_packed_manifest(repo_root, manifest_path)
        if not manifest_result["valid"]:
            raise ValueError(f"packed manifest invalid: {manifest_result['failures'][:3]}")
    runs = _load_trace_runs(repo_root, trace_paths)
    category_rows, category_summary = _build_category_rows(runs)
    node_rows, node_summary = _group_node_rows(runs)
    trace_off = _load_trace_off_summary(experiment_dir / "data/compat-ab-summary.json", repo_root)
    return write_outputs(
        output_dir,
        manifest=manifest_result,
        runs=runs,
        category_rows=category_rows,
        category_summary=category_summary,
        node_rows=node_rows,
        node_summary=node_summary,
        trace_off=trace_off,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--skip-manifest-validation", action="store_true",
                        help="только для локальной разработки; опубликованный прогон обязан валидировать manifest")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    _, default_output, default_manifest = _default_paths(repo_root)
    output_dir = (args.output_dir or default_output).resolve()
    manifest = (args.manifest or default_manifest).resolve()
    try:
        summary = run_analysis(
            repo_root, output_dir, manifest,
            skip_manifest=args.skip_manifest_validation,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"E049b ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "experiment": summary["experiment_id"],
        "classification": summary["classification"],
        "output_dir": str(output_dir),
        "q1_logical_bytes_per_token": summary["logical_bytes_per_token"]["q1_packed_weight_read_bytes"]["value"],
        "q8_logical_bytes_per_token": summary["logical_bytes_per_token"]["q8_activation_logical_read_bytes"]["value"],
        "observed_ddr_read_bytes": summary["logical_bytes_per_token"]["observed_ddr_read_bytes"],
        "observed_ddr_write_bytes": summary["logical_bytes_per_token"]["observed_ddr_write_bytes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
