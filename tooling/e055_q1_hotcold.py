"""Contracts and post-processing for E055 Q1 hot/cold microbenchmarks.

E055 is deliberately a bounded ARM microbenchmark, not a model benchmark.  It
keeps the stock packed Q1_0 4x4 traversal and separates three controls:

* ``packed_stream`` — read the same Q1/Q8 carrier bytes and checksum them;
* ``unpack_scale`` — perform the register sign expansion and scale loads, but
  no dot product;
* ``full_dotprod`` — the stock NEON/DOTPROD kernel used by the target build.

The functions in this module are intentionally conservative.  They distinguish
logical/declared bytes from PMU event counts, refuse to compare unlike modes,
and label a cache condition as ``cold_conditioned`` rather than pretending that
a software thrash loop proves an architectural cache miss.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "e055-q1-hot-cold/v1"
Q1_BLOCK_BYTES = 72
Q8_BLOCK_BYTES = 34
Q8_BLOCKS_PER_Q1_BLOCK = 4
NATIVE_CARRIER_BYTES = Q1_BLOCK_BYTES + Q8_BLOCKS_PER_Q1_BLOCK * Q8_BLOCK_BYTES
Q1_VALUES_PER_BLOCK = 128
DOT_PRODUCTS_PER_NATIVE_BLOCK = Q8_BLOCKS_PER_Q1_BLOCK * 8 * 4 * 4

MODES = ("packed_stream", "unpack_scale", "full_dotprod")
CACHE_STATES = ("hot_repeat", "cold_conditioned")
WORKING_SET_BYTES = (
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    12_500 * 1024,
)
CPUS = (0, 6)
MIN_PAIRED_SAMPLES = 5


def actual_working_set(target_bytes: int) -> dict[str, int]:
    """Round a requested set up to complete native Q1/Q8 carrier blocks."""

    target = int(target_bytes)
    if target <= 0:
        raise ValueError("working set must be positive")
    blocks = max(1, (target + NATIVE_CARRIER_BYTES - 1) // NATIVE_CARRIER_BYTES)
    return {
        "target_bytes": target,
        "actual_bytes": blocks * NATIVE_CARRIER_BYTES,
        "blocks": blocks,
        "carrier_bytes_per_block": NATIVE_CARRIER_BYTES,
    }


def logical_bytes_per_call(blocks: int) -> dict[str, int]:
    """Return declared input/operation counts for one native group call.

    These are algorithmic descriptors.  They are not DDR traffic and must not
    be converted into bandwidth without an independent hardware measurement.
    """

    count = int(blocks)
    if count <= 0:
        raise ValueError("blocks must be positive")
    q1 = count * Q1_BLOCK_BYTES
    q8 = count * Q8_BLOCKS_PER_Q1_BLOCK * Q8_BLOCK_BYTES
    return {
        "q1_packed_bytes": q1,
        "q8_bytes": q8,
        "total_input_bytes": q1 + q8,
        "output_bytes": 4 * 4,
        "dot_products": count * DOT_PRODUCTS_PER_NATIVE_BLOCK,
        "q1_values": count * Q1_VALUES_PER_BLOCK,
    }


def summarize_samples(values: Sequence[float]) -> dict[str, Any]:
    """Summarize at least one repeated wall-time or throughput sample."""

    if not values:
        raise ValueError("sample list must not be empty")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise ValueError("samples must be finite and non-negative")
    mean = statistics.fmean(numeric)
    stdev = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
    ordered = sorted(numeric)
    return {
        "count": len(numeric),
        "samples": numeric,
        "mean": mean,
        "median": statistics.median(numeric),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev_population": stdev,
        "cv": stdev / mean if mean else None,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_sample(sample: Mapping[str, Any]) -> list[str]:
    """Return contract violations for one harness/PMU joined sample.

    A list is used instead of raising so a target failure can publish every
    violation in its raw result.  ``observed_ddr_*`` is intentionally forbidden
    here: E055 has PMU event counts, not a direct DDR-byte counter.
    """

    errors: list[str] = []
    if sample.get("schema") != SCHEMA:
        errors.append("schema must be e055-q1-hot-cold/v1")
    if sample.get("mode") not in MODES:
        errors.append(f"mode must be one of {MODES}")
    if sample.get("cache_state") not in CACHE_STATES:
        errors.append(f"cache_state must be one of {CACHE_STATES}")
    if not _is_int(sample.get("cpu")) or int(sample["cpu"]) not in CPUS:
        errors.append("cpu must be A55 CPU0 or A76 CPU6")
    if sample.get("golden_pass") is not True:
        errors.append("golden_pass must be true for a qualified sample")
    cold = sample.get("cold_conditioning")
    if not isinstance(cold, Mapping):
        errors.append("cold_conditioning metadata is required")
    elif sample.get("cache_state") == "cold_conditioned" and cold.get("verified_touched") is not True:
        errors.append("cold conditioning did not verify every thrash line")
    for key in ("actual_working_set_bytes", "blocks", "iterations", "elapsed_ns", "calls"):
        if not _is_int(sample.get(key)) or int(sample[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    if not isinstance(sample.get("checksum"), str) or not sample["checksum"]:
        errors.append("checksum is required to defeat dead-code elimination")
    elif sample["checksum"].lower() in {"0x0", "0x00"}:
        errors.append("checksum must be non-zero")
    pmu = sample.get("pmu")
    if not isinstance(pmu, Mapping):
        errors.append("pmu metadata is required")
    else:
        if pmu.get("values_are_event_counts") is not True:
            errors.append("PMU values must be explicitly labelled event counts")
        events = pmu.get("events")
        if not isinstance(events, list):
            errors.append("pmu.events must be a list")
        else:
            for event in events:
                if not isinstance(event, Mapping):
                    errors.append("pmu event must be an object")
                    continue
                if "value" in event and not _is_int(event["value"]):
                    errors.append("PMU event value must be an integer count")
                if any(key in event for key in ("bytes", "ddr_bytes", "bandwidth_bytes")):
                    errors.append("PMU event counts must never be named bytes")
    for key in ("observed_ddr_read_bytes", "observed_ddr_write_bytes"):
        if key in sample and sample[key] is not None:
            errors.append(f"{key} is forbidden: E055 has no direct DDR-byte counter")
    if _is_int(sample.get("actual_working_set_bytes")) and _is_int(sample.get("blocks")):
        if int(sample["actual_working_set_bytes"]) != int(sample["blocks"]) * NATIVE_CARRIER_BYTES:
            errors.append("actual_working_set_bytes does not match native carrier blocks")
    return errors


def _comparison_key(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        sample.get("mode"),
        sample.get("cache_state"),
        sample.get("cpu"),
        sample.get("actual_working_set_bytes"),
        sample.get("blocks"),
    )


def paired_speed_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Return fractional speed improvement for genuinely comparable samples.

    This function deliberately refuses to subtract ``full_dotprod`` from
    ``packed_stream``.  Cross-mode attribution is performed by
    :func:`infer_bottleneck`, where the relationship is explicit and labelled.
    """

    if _comparison_key(reference) != _comparison_key(candidate):
        raise ValueError("cannot subtract incomparable E055 samples")
    ref_ns = int(reference.get("elapsed_ns", 0))
    cand_ns = int(candidate.get("elapsed_ns", 0))
    if ref_ns <= 0 or cand_ns <= 0:
        raise ValueError("elapsed_ns must be positive")
    return (ref_ns - cand_ns) / ref_ns


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("mode"),
                row.get("cache_state"),
                row.get("cpu"),
                row.get("actual_working_set_bytes"),
            )
        ].append(row)
    return grouped


def infer_bottleneck(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a cautious memory-vs-unpack inference from comparable controls.

    ``cold/hot`` is a cache sensitivity signal.  ``full/unpack`` and
    ``full/stream`` are work decomposition ratios, not speedups.  The function
    only emits a directional hypothesis when each participating cell has five
    samples; otherwise it returns ``insufficient_samples``.
    """

    grouped = _group_rows(rows)
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, samples in grouped.items():
        if len(samples) < MIN_PAIRED_SAMPLES:
            continue
        elapsed_ms = [float(item["elapsed_ns"]) / 1_000_000.0 for item in samples]
        cells[key] = summarize_samples(elapsed_ms)

    records: list[dict[str, Any]] = []
    memory_ratios: list[float] = []
    unpack_ratios: list[float] = []
    for (mode, state, cpu, size), summary in cells.items():
        other_state = "cold_conditioned" if state == "hot_repeat" else "hot_repeat"
        hot_key = (mode, "hot_repeat", cpu, size)
        cold_key = (mode, "cold_conditioned", cpu, size)
        if hot_key in cells and cold_key in cells and state == "hot_repeat":
            ratio = cells[cold_key]["median"] / cells[hot_key]["median"]
            memory_ratios.append(ratio)
            records.append({
                "kind": "cache_sensitivity",
                "mode": mode,
                "cpu": cpu,
                "actual_working_set_bytes": size,
                "cold_over_hot_median": ratio,
            })
        full_hot = ("full_dotprod", "hot_repeat", cpu, size)
        unpack_hot = ("unpack_scale", "hot_repeat", cpu, size)
        stream_hot = ("packed_stream", "hot_repeat", cpu, size)
        if mode == "full_dotprod" and state == "hot_repeat":
            if unpack_hot in cells:
                unpack_ratios.append(summary["median"] / cells[unpack_hot]["median"])
            if stream_hot in cells:
                records.append({
                    "kind": "full_over_stream_work_ratio",
                    "cpu": cpu,
                    "actual_working_set_bytes": size,
                    "full_over_stream_median": summary["median"] / cells[stream_hot]["median"],
                })

    if not cells:
        classification = "insufficient_samples"
        confidence = 0.0
    elif memory_ratios and unpack_ratios:
        # A materially larger cold penalty in the full kernel than in the
        # unpack-only control points to memory/cache sensitivity.  Otherwise
        # the hot full/unpack ratio points to arithmetic/unpack work.  This is
        # intentionally a hypothesis, never a measured DDR bandwidth claim.
        median_memory = statistics.median(memory_ratios)
        median_unpack = statistics.median(unpack_ratios)
        classification = "memory_cache_sensitive" if median_memory >= 1.20 else "unpack_compute_sensitive"
        confidence = min(1.0, 0.5 + 0.1 * min(len(memory_ratios), 5) + 0.1 * min(len(unpack_ratios), 5))
    else:
        classification = "insufficient_control_cells"
        confidence = 0.0
    return {
        "schema": "e055-q1-inference/v1",
        "classification": classification,
        "confidence": confidence,
        "cell_count": len(cells),
        "memory_cache_sensitivity_ratios": memory_ratios,
        "hot_unpack_compute_ratios": unpack_ratios,
        "records": records,
        "interpretation": (
            "Directional microbenchmark inference only; PMU values are event counts, "
            "not DDR bytes. A full-model decision requires the E049 per-token trace."
        ),
    }


def rows_to_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return a compact CSV-like table for raw publication and review."""

    columns = (
        "mode", "cache_state", "cpu", "actual_working_set_bytes", "blocks",
        "elapsed_ns", "calls", "calls_per_second", "checksum", "golden_pass",
    )
    output = [",".join(columns)]
    for row in rows:
        output.append(",".join(str(row.get(column, "")) for column in columns))
    return "\n".join(output) + "\n"
