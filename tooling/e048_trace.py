#!/usr/bin/env python3
"""Host-side validation and aggregation for the E048 llama.cpp trace.

The target patch deliberately records compact fixed-size events.  This module
does all string-heavy validation, byte-range unioning and statistics after the
run so the instrumentation itself does not write JSON or take locks on the
decode hot path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Sequence


TRACE_SCHEMA = "e048-trace/v1"
DIRECT_DDR_COUNTER_METHODS = frozenset({
    "nsi-pmu-direct",
    "perf-direct-ddr",
    "vendor-ddr-pmu-direct",
})


def q1_gemv_byte_ledger(*, n: int, nc: int) -> dict[str, int]:
    """Return the Q1_0×Q8_0 GEMV logical/unique byte ledger.

    The current A733 kernels process four output rows per packed block.  One
    Q1 block covers 128 values and is 72 bytes after repacking (four FP16
    scales plus 64 sign bytes).  The corresponding four Q8_0 activation
    blocks are 4×34 bytes.  Activation is logically revisited for every output
    group, but the unique activation vector is counted once.
    """

    if n <= 0 or n % 128 != 0:
        raise ValueError("Q1 GEMV n must be a positive multiple of 128")
    if nc <= 0 or nc % 4 != 0:
        raise ValueError("Q1 GEMV nc must be a positive multiple of 4")

    blocks = n // 128
    output_groups = nc // 4
    q1_weight_bytes = blocks * output_groups * 72
    q8_vector_bytes = blocks * 4 * 34
    return {
        "q1_packed_weight_read_bytes": q1_weight_bytes,
        "q8_activation_logical_read_bytes": output_groups * q8_vector_bytes,
        "q8_activation_unique_read_bytes": q8_vector_bytes,
        "output_write_bytes": nc * 4,
        "q1_block_count": blocks * output_groups,
        "q8_block_count": blocks * 4,
    }


def f32_to_q8_byte_ledger(n: int) -> dict[str, int]:
    """Describe the scratch conversion preceding a Q1 dot product."""

    if n <= 0 or n % 32 != 0:
        raise ValueError("Q8_0 conversion n must be a positive multiple of 32")
    q8_bytes = (n // 32) * 34
    return {
        "f32_source_read_bytes": n * 4,
        "q8_scratch_write_bytes": q8_bytes,
        "q8_scratch_read_bytes": q8_bytes,
    }


def unique_byte_ranges(ranges: Iterable[tuple[int, int]]) -> int:
    """Return the union length of half-open byte ranges."""

    normalized = []
    for start, end in ranges:
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range: {(start, end)!r}")
        if end == start:
            continue
        normalized.append((start, end))
    normalized.sort()
    total = 0
    current_start = current_end = None
    for start, end in normalized:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return total


def _nonnegative_int(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{field} должен быть неотрицательным integer")


def validate_trace_events(events: Sequence[dict[str, Any]]) -> list[str]:
    """Validate raw compact events without requiring a model runtime."""

    errors: list[str] = []
    if not events:
        return ["trace не содержит событий"]
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event[{index}] должен быть объектом")
            continue
        if event.get("schema") != TRACE_SCHEMA:
            errors.append(f"event[{index}]: неверная schema")
        kind = event.get("event")
        if kind not in {"token_begin", "token_end", "node_worker", "q1_kernel", "phase_worker", "trace_summary"}:
            errors.append(f"event[{index}]: неизвестный тип {kind!r}")

        for field in ("observed_ddr_read_bytes", "observed_ddr_write_bytes"):
            value = event.get(field)
            if value is not None:
                _nonnegative_int(value, field, errors)
                if event.get("counter_method") not in DIRECT_DDR_COUNTER_METHODS:
                    errors.append(f"{field} разрешён только с прямым DDR counter_method")
        if kind == "token_begin":
            _nonnegative_int(event.get("start_ns"), "start_ns", errors)
        elif kind == "token_end":
            _nonnegative_int(event.get("end_ns"), "end_ns", errors)
        elif kind == "node_worker":
            for field in ("start_ns", "end_ns", "post_barrier_end_ns"):
                _nonnegative_int(event.get(field), field, errors)
            if all(isinstance(event.get(field), int) for field in ("start_ns", "end_ns", "post_barrier_end_ns")):
                if event["end_ns"] < event["start_ns"]:
                    errors.append("node_worker: end_ns раньше start_ns")
                if event["post_barrier_end_ns"] < event["end_ns"]:
                    errors.append("node_worker: post_barrier_end_ns раньше end_ns")
            for field in ("logical_read_bytes", "logical_write_bytes"):
                _nonnegative_int(event.get(field), field, errors)
        elif kind == "q1_kernel":
            for field in ("start_ns", "end_ns", "q1_packed_weight_read_bytes",
                          "q8_activation_logical_read_bytes", "q8_activation_unique_read_bytes",
                          "output_write_bytes"):
                _nonnegative_int(event.get(field), field, errors)
            if all(isinstance(event.get(field), int) for field in ("start_ns", "end_ns")):
                if event["end_ns"] < event["start_ns"]:
                    errors.append("q1_kernel: end_ns раньше start_ns")
        elif kind == "phase_worker":
            for field in ("start_ns", "end_ns"):
                _nonnegative_int(event.get(field), field, errors)
            if all(isinstance(event.get(field), int) for field in ("start_ns", "end_ns")):
                if event["end_ns"] < event["start_ns"]:
                    errors.append("phase_worker: end_ns раньше start_ns")
    return errors


def aggregate_trace(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate validated worker events by token and graph node."""

    errors = validate_trace_events(events)
    if errors:
        raise ValueError("; ".join(errors))

    tokens: dict[int, dict[str, Any]] = {}
    nodes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    q1_events: list[dict[str, Any]] = []
    phase_events: list[dict[str, Any]] = []
    for event in events:
        token_seq = int(event.get("token_seq", 0))
        kind = event["event"]
        if kind == "token_begin":
            tokens.setdefault(token_seq, {})["start_ns"] = event["start_ns"]
        elif kind == "token_end":
            tokens.setdefault(token_seq, {})["end_ns"] = event["end_ns"]
        elif kind == "node_worker":
            nodes[(token_seq, int(event["node_index"]))].append(event)
        elif kind == "q1_kernel":
            q1_events.append(event)
        elif kind == "phase_worker":
            phase_events.append(event)

    token_rows = []
    for token_seq, row in sorted(tokens.items()):
        if "start_ns" not in row or "end_ns" not in row:
            raise ValueError(f"token {token_seq} не имеет begin/end")
        token_rows.append({
            "token_seq": token_seq,
            "start_ns": row["start_ns"],
            "end_ns": row["end_ns"],
            "duration_ns": row["end_ns"] - row["start_ns"],
        })

    node_rows = []
    for (token_seq, node_index), workers in sorted(nodes.items()):
        first = workers[0]
        wall_start = min(event["start_ns"] for event in workers)
        wall_end = max(event["post_barrier_end_ns"] for event in workers)
        node_rows.append({
            "token_seq": token_seq,
            "node_index": node_index,
            "layer_index": first.get("layer_index"),
            "op_name": first.get("op_name"),
            "tensor_id": first.get("tensor_id"),
            "workers": len(workers),
            "wall_start_ns": wall_start,
            "wall_end_ns": wall_end,
            "wall_duration_ns": wall_end - wall_start,
            "worker_compute_duration_ns": max(event["end_ns"] - event["start_ns"] for event in workers),
            "logical_read_bytes": sum(event["logical_read_bytes"] for event in workers),
            "logical_write_bytes": sum(event["logical_write_bytes"] for event in workers),
        })
    return {"tokens": token_rows, "nodes": node_rows, "q1_kernels": q1_events, "phases": phase_events}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("нет значений для percentile")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def classify_trace_ab(
    off_ms: Sequence[float],
    on_ms: Sequence[float],
    *,
    token_stream_equal: bool,
    max_overhead_fraction: float = 0.01,
) -> dict[str, Any]:
    """Compute paired overhead and classify PASS versus TRACE_ONLY/INVALID."""

    if len(off_ms) != len(on_ms) or not off_ms or any(value <= 0 for value in (*off_ms, *on_ms)):
        raise ValueError("A/B samples must be paired positive values")
    off_median = float(median(off_ms))
    on_median = float(median(on_ms))
    off_p95 = _percentile(off_ms, 0.95)
    on_p95 = _percentile(on_ms, 0.95)
    median_overhead = on_median / off_median - 1.0
    p95_overhead = on_p95 / off_p95 - 1.0
    if not token_stream_equal:
        classification = "INVALID"
    elif median_overhead > max_overhead_fraction or p95_overhead > max_overhead_fraction:
        classification = "TRACE_ONLY"
    else:
        classification = "PASS"
    return {
        "off_median_ms": off_median,
        "on_median_ms": on_median,
        "off_p95_ms": off_p95,
        "on_p95_ms": on_p95,
        "median_overhead_fraction": median_overhead,
        "p95_overhead_fraction": p95_overhead,
        "token_stream_equal": token_stream_equal,
        "classification": classification,
    }
