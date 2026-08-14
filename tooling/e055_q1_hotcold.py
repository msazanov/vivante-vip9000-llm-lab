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

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "e055-q1-hot-cold/v2"
HARNESS_SCHEMA = "e055-q1-hot-cold-harness/v1"
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
    25 * 1024 * 1024 // 2,
)
CPUS = (0, 6)
MIN_PAIRED_SAMPLES = 5
PROMOTION_THRESHOLD = 0.04
MAX_UINT64 = (1 << 64) - 1
UPSTREAM_REPACK_SHA256 = "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d"
UPSTREAM_COMMIT = "38c66ad0241da4f9fcce541cda8edc219086cec5"
UPSTREAM_REF = (
    "https://github.com/ggml-org/llama.cpp/commit/"
    "38c66ad0241da4f9fcce541cda8edc219086cec5"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_RE = re.compile(r"^0x[0-9a-f]+$")
PMU_GROUP_CONFIGS = {
    "core": {"cpu_cycles": "0x11", "instructions": "0x8", "stall_backend": "0x24"},
    "cache": {
        "l1d_cache_refill": "0x3",
        "l2d_cache_refill": "0x17",
        "l3d_cache_refill": "0x2a",
    },
    "memory": {"mem_access": "0x13", "bus_access": "0x19"},
}
PMU_GROUP_EVENTS = {group: set(configs) for group, configs in PMU_GROUP_CONFIGS.items()}
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DATA = ROOT / "experiments/E055-q1-hot-cold/data"
PUBLICATION_MANIFEST = EXPERIMENT_DATA / "manifest.json"
DISASSEMBLY_REVIEW = EXPERIMENT_DATA / "disassembly-review.json"
UPSTREAM_BINDING = EXPERIMENT_DATA / "upstream-source-binding.json"
HARNESS_SOURCE = ROOT / "tooling/e055_q1_hotcold.cpp"
PMU_SOURCE = ROOT / "tooling/a733_pmu_exec.c"


def actual_working_set(target_bytes: int) -> dict[str, int]:
    """Round a requested set up to complete native Q1/Q8 carrier blocks."""

    if not _is_int(target_bytes):
        raise ValueError("working set must be an integer")
    target = target_bytes
    if target <= 0:
        raise ValueError("working set must be positive")
    if target > MAX_UINT64 - (NATIVE_CARRIER_BYTES - 1):
        raise ValueError("working set rounding would overflow uint64")
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

    if not _is_int(blocks):
        raise ValueError("blocks must be an integer")
    count = blocks
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_publication_contract() -> dict[str, Any]:
    """Load the immutable runtime qualifier from committed E055 artifacts.

    Callers do not supply expected hashes. The publication manifest pins the
    disassembly report, harness source, compiler, allowed binaries, upstream
    commit/ref and repack blob. Any cross-artifact disagreement fails closed.
    """

    try:
        manifest = json.loads(PUBLICATION_MANIFEST.read_text(encoding="utf-8"))
        disassembly = json.loads(DISASSEMBLY_REVIEW.read_text(encoding="utf-8"))
        upstream = json.loads(UPSTREAM_BINDING.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load E055 publication contract: {exc}") from exc
    runtime = manifest.get("runtime_qualification")
    if not isinstance(runtime, Mapping) or runtime.get("schema") != "e055-runtime-qualification/v1":
        raise ValueError("manifest has no E055 runtime qualification contract")
    provenance = disassembly.get("provenance")
    builds = disassembly.get("builds")
    if not isinstance(provenance, Mapping) or not isinstance(builds, list):
        raise ValueError("disassembly provenance/builds are malformed")
    if disassembly.get("status") != "PASS" or not builds or any(
        not isinstance(item, Mapping) or item.get("golden_pass") is not True
        or item.get("golden_cases") != 18 for item in builds
    ):
        raise ValueError("allowed harness builds lack the exact 18-case golden")
    expected_builds = {
        item.get("name"): item.get("binary_sha256")
        for item in builds if isinstance(item, Mapping)
    }
    publication_hashes = [
        provenance.get("source_sha256"), provenance.get("compiler_sha256"),
        upstream.get("sha256"), *expected_builds.values(),
    ]
    if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
           or value == "0" * 64 for value in publication_hashes):
        raise ValueError("publication provenance contains an invalid or all-zero SHA-256")
    runtime_builds = runtime.get("allowed_builds")
    if not isinstance(runtime_builds, Mapping) or dict(runtime_builds) != expected_builds:
        raise ValueError("manifest allowed builds do not match disassembly evidence")
    exact_fields = {
        "harness_schema": HARNESS_SCHEMA,
        "golden_cases": 18,
        "source_sha256": provenance.get("source_sha256"),
        "compiler_sha256": provenance.get("compiler_sha256"),
        "compiler_id": provenance.get("compiler_id"),
        "upstream_commit": upstream.get("commit"),
        "upstream_ref": upstream.get("immutable_ref"),
        "upstream_repack_sha256": upstream.get("sha256"),
    }
    for key, value in exact_fields.items():
        if runtime.get(key) != value:
            raise ValueError(f"manifest runtime {key} does not match publication evidence")
    if runtime.get("pmu_group_configs") != PMU_GROUP_CONFIGS:
        raise ValueError("manifest PMU configs do not match E049c definitions")
    if runtime.get("pmu_source_sha256") != _sha256_file(PMU_SOURCE):
        raise ValueError("manifest does not pin the exact E049c PMU source")
    if upstream.get("commit") != UPSTREAM_COMMIT or upstream.get("immutable_ref") != UPSTREAM_REF \
            or upstream.get("sha256") != UPSTREAM_REPACK_SHA256:
        raise ValueError("upstream source binding is not the reviewed immutable blob")
    if _sha256_file(HARNESS_SOURCE) != runtime.get("source_sha256"):
        raise ValueError("published harness source hash does not match the working tree")
    manifested = {
        item.get("path"): item for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    relevant = (HARNESS_SOURCE, PMU_SOURCE, DISASSEMBLY_REVIEW, UPSTREAM_BINDING)
    for path in relevant:
        relative = path.relative_to(ROOT).as_posix()
        item = manifested.get(relative)
        if not isinstance(item, Mapping) or item.get("sha256") != _sha256_file(path) \
                or item.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"manifest does not bind runtime artifact {relative}")
    return dict(runtime)


def ns_per_traversal(sample: Mapping[str, Any]) -> float:
    """Normalize a sample before comparing different call counts."""

    elapsed = sample.get("elapsed_ns")
    calls = sample.get("calls")
    if not _is_int(elapsed) or not _is_int(calls) or elapsed <= 0 or calls <= 0:
        raise ValueError("elapsed_ns and calls must be positive integers")
    return float(elapsed) / float(calls)


def traversals_per_second(sample: Mapping[str, Any]) -> float:
    """Return complete stock-order traversals per second."""

    return 1.0e9 / ns_per_traversal(sample)


def validate_sample(
    sample: Mapping[str, Any],
) -> list[str]:
    """Reject legacy caller-created sample mappings.

    Since Stage 4, measurement qualification starts from a committed raw-bundle
    manifest path.  A mapping can be internally consistent yet fabricated, so
    it can never be promoted as evidence.
    """

    del sample
    return [
        "self-declared sample mappings are not evidence; provide a committed "
        "sealed raw-bundle manifest path"
    ]


def _validate_sample(sample: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    """Validate one sample against an already verified publication contract."""

    errors: list[str] = []
    if sample.get("schema") != SCHEMA:
        errors.append("schema must be e055-q1-hot-cold/v2")
    if sample.get("mode") not in MODES:
        errors.append(f"mode must be one of {MODES}")
    if sample.get("cache_state") not in CACHE_STATES:
        errors.append(f"cache_state must be one of {CACHE_STATES}")
    if not _is_int(sample.get("cpu")) or int(sample["cpu"]) not in CPUS:
        errors.append("cpu must be A55 CPU0 or A76 CPU6")
    cpu = sample.get("cpu")
    for key in ("pinned_cpu", "cpu_start", "cpu_end"):
        if not _is_int(sample.get(key)):
            errors.append(f"{key} must be an integer CPU identifier")
    if sample.get("pinned_cpu") != cpu or sample.get("cpu_start") != cpu or \
            sample.get("cpu_end") != cpu:
        errors.append("sample must remain pinned to the declared CPU")
    affinity = sample.get("affinity_cpus")
    if not isinstance(affinity, list) or len(affinity) != 1 or not _is_int(affinity[0]) \
            or affinity != [cpu]:
        errors.append("affinity must contain one exact integer CPU")
    if not _is_int(sample.get("cpu_migration_count")) or \
            sample.get("cpu_migration_count") != 0:
        errors.append("affinity must contain exactly the declared CPU with no migration")
    if not isinstance(sample.get("run_id"), str) or not sample["run_id"]:
        errors.append("run_id is required")
    if not isinstance(sample.get("pair_id"), str) or not sample["pair_id"]:
        errors.append("pair_id is required")
    if not _is_int(sample.get("pair_index")) or sample.get("pair_index", 0) <= 0:
        errors.append("pair_index must be a positive integer")
    if sample.get("pair_order") not in ("hot_then_cold", "cold_then_hot"):
        errors.append("pair_order is invalid")
    if not _is_int(sample.get("order_index")) or sample.get("order_index") not in (1, 2):
        errors.append("order_index must be 1 or 2")
    expected_order = {
        ("hot_then_cold", "hot_repeat"): 1,
        ("hot_then_cold", "cold_conditioned"): 2,
        ("cold_then_hot", "cold_conditioned"): 1,
        ("cold_then_hot", "hot_repeat"): 2,
    }.get((sample.get("pair_order"), sample.get("cache_state")))
    if expected_order is not None and sample.get("order_index") != expected_order:
        errors.append("order_index does not match pair_order and cache_state")
    if sample.get("golden_pass") is not True:
        errors.append("golden_pass must be true for a qualified sample")
    if sample.get("golden_cases") != 18 or not _is_int(sample.get("golden_cases")):
        errors.append("golden_cases must be exactly 18")
    conditioning = sample.get("cold_conditioning")
    if not isinstance(conditioning, Mapping):
        errors.append("cold_conditioning metadata is required")
    else:
        required = {
            "strategy", "requested_bytes", "actual_bytes", "line_bytes",
            "lines_touched", "checksum", "verified_touched",
        }
        if not required.issubset(conditioning):
            errors.append("cold_conditioning is missing full coverage fields")
        if conditioning.get("verified_touched") is not True:
            errors.append("cache conditioning must be verified for both hot and cold samples")
        for key in ("requested_bytes", "actual_bytes", "line_bytes", "lines_touched"):
            if not _is_int(conditioning.get(key)) or conditioning.get(key, 0) <= 0:
                errors.append(f"cold_conditioning {key} must be a positive integer")
        condition_checksum = conditioning.get("checksum")
        if not isinstance(condition_checksum, str) or \
                CHECKSUM_RE.fullmatch(condition_checksum) is None or \
                int(condition_checksum, 16) == 0:
            errors.append("cold_conditioning checksum must be nonzero lowercase hexadecimal")
        if conditioning.get("line_bytes") != 64:
            errors.append("cold_conditioning line_bytes must be 64")
        requested = conditioning.get("requested_bytes")
        actual = conditioning.get("actual_bytes")
        lines = conditioning.get("lines_touched")
        if all(_is_int(value) for value in (requested, actual, lines)):
            if requested != actual:
                errors.append("cold_conditioning requested and actual bytes must match")
            if sample.get("cache_state") == "cold_conditioned":
                if conditioning.get("strategy") != "verified_write_read_each_64B_line":
                    errors.append("cold sample requires exact verified thrash strategy")
                if actual % 64 != 0 or lines * 64 != actual:
                    errors.append("cold conditioning must cover every requested 64-byte line")
            elif sample.get("cache_state") == "hot_repeat":
                if conditioning.get("strategy") != "verified_kernel_warmup":
                    errors.append("hot sample requires exact verified warmup strategy")
                if not _is_int(conditioning.get("warmup_calls")) or \
                        conditioning.get("warmup_calls", 0) <= 0:
                    errors.append("hot conditioning requires positive integer warmup_calls")
                if actual != sample.get("actual_working_set_bytes") or \
                        lines != (actual + 63) // 64:
                    errors.append("hot conditioning coverage must match the working set")
    for key in ("target_working_set_bytes", "actual_working_set_bytes", "blocks",
                "iterations", "elapsed_ns", "calls"):
        if not _is_int(sample.get(key)) or int(sample[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    target_working_set = sample.get("target_working_set_bytes")
    if _is_int(target_working_set) and target_working_set not in WORKING_SET_BYTES:
        errors.append("target_working_set_bytes is outside the documented seven-size matrix")
    checksum = sample.get("checksum")
    if not isinstance(checksum, str) or CHECKSUM_RE.fullmatch(checksum) is None:
        errors.append("checksum must use exact lowercase hexadecimal 0x schema")
    elif int(checksum, 16) == 0:
        errors.append("checksum must be non-zero")
    if sample.get("cache_state") == "cold_conditioned" and (
        sample.get("iterations") != 1 or sample.get("calls") != 1
    ):
        errors.append("cold_conditioned requires exactly one iteration and one call")
    if _is_int(sample.get("iterations")) and _is_int(sample.get("calls")) \
            and sample["iterations"] != sample["calls"]:
        errors.append("iterations and calls must match")
    sync = sample.get("sync")
    if not isinstance(sync, Mapping) or sync.get("requested") is not True or sync.get("started") is not True \
            or sync.get("acknowledged") is not True or sync.get("ended") is not True \
            or sync.get("sequence") != "S/A/E":
        errors.append("qualified sample requires exact S/ACK/E synchronization")
    pmu = sample.get("pmu")
    if not isinstance(pmu, Mapping):
        errors.append("pmu metadata is required")
    else:
        if pmu.get("schema_version") != "e049c-arm-pmu/v2" or pmu.get("sample_valid") is not True \
                or pmu.get("status") != "ok":
            errors.append("PMU must be a valid E049c v2 sample")
        group = pmu.get("event_group")
        if group not in PMU_GROUP_EVENTS:
            errors.append("PMU event_group must be core, cache, or memory")
        if pmu.get("event_source") != "armv8_pmuv3_raw_config":
            errors.append("PMU event_source must be armv8_pmuv3_raw_config")
        if pmu.get("values_are_event_counts") is not True:
            errors.append("PMU values must be explicitly labelled event counts")
        events = pmu.get("events")
        if not isinstance(events, list) or not events:
            errors.append("pmu.events must be a non-empty list")
        else:
            names = {event.get("name") for event in events if isinstance(event, Mapping)}
            if group in PMU_GROUP_EVENTS and (
                names != PMU_GROUP_EVENTS[group] or len(events) != len(PMU_GROUP_EVENTS[group])
            ):
                errors.append("PMU events must exactly match the selected group")
            if (
                not _is_int(pmu.get("event_group_size"))
                or pmu.get("event_group_size") != len(events)
            ):
                errors.append("PMU event_group_size must match the exact event list")
            for event in events:
                if not isinstance(event, Mapping):
                    errors.append("pmu event must be an object")
                    continue
                if event.get("support") != "supported" or event.get("sample_valid") is not True:
                    errors.append("every PMU event must be supported and sample_valid")
                if group in PMU_GROUP_CONFIGS and \
                        event.get("config") != PMU_GROUP_CONFIGS[group].get(event.get("name")):
                    errors.append("PMU event config must exactly match E049c a733_pmu_exec.c")
                if not _is_int(event.get("value")) or event.get("value", -1) < 0:
                    errors.append("PMU event value must be a non-negative integer count")
                ratio = event.get("running_ratio")
                if type(ratio) is not float or not math.isfinite(ratio) or ratio != 1.0:
                    errors.append("every PMU event must have finite float running_ratio exactly 1.0")
                if any(key in event for key in ("bytes", "ddr_bytes", "bandwidth_bytes")):
                    errors.append("PMU event counts must never be named bytes")
    for key in ("observed_ddr_read_bytes", "observed_ddr_write_bytes"):
        if key in sample and sample[key] is not None:
            errors.append(f"{key} is forbidden: E055 has no direct DDR-byte counter")
    if _is_int(sample.get("actual_working_set_bytes")) and _is_int(sample.get("blocks")):
        if int(sample["actual_working_set_bytes"]) != int(sample["blocks"]) * NATIVE_CARRIER_BYTES:
            errors.append("actual_working_set_bytes does not match native carrier blocks")
    if _is_int(target_working_set) and target_working_set > 0:
        expected_layout = actual_working_set(target_working_set)
        if sample.get("actual_working_set_bytes") != expected_layout["actual_bytes"] or \
                sample.get("blocks") != expected_layout["blocks"]:
            errors.append("working-set layout does not match the documented target size")
    thermal = sample.get("thermal")
    if not isinstance(thermal, Mapping) or thermal.get("readable") is not True \
            or thermal.get("tripped") is not False:
        errors.append("thermal gate must be readable and pass")
    elif not _is_number(thermal.get("max_temp_c")) \
            or not _is_number(thermal.get("limit_c")) \
            or not math.isfinite(float(thermal["max_temp_c"])) \
            or not math.isfinite(float(thermal["limit_c"])) \
            or thermal["max_temp_c"] > thermal["limit_c"]:
        errors.append("thermal maximum must not exceed its limit")
    provenance = sample.get("provenance")
    provenance_keys = (
        "build_name", "source_sha256", "binary_sha256", "compiler_sha256",
        "compiler_id", "upstream_commit", "upstream_ref", "upstream_repack_sha256",
    )
    if not isinstance(provenance, Mapping):
        errors.append("provenance is required")
    else:
        if set(provenance) != set(provenance_keys):
            errors.append("provenance must contain the exact publication-bound fields")
        for key in ("source_sha256", "binary_sha256", "compiler_sha256", "upstream_repack_sha256"):
            value = provenance.get(key)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
                errors.append(f"{key} must be a nonzero lowercase SHA-256")
        if not isinstance(provenance.get("compiler_id"), str) or not provenance["compiler_id"]:
            errors.append("compiler_id is required")
        build_name = provenance.get("build_name")
        allowed_builds = contract.get("allowed_builds")
        expected_provenance = {
            "build_name": build_name,
            "source_sha256": contract.get("source_sha256"),
            "binary_sha256": (
                allowed_builds.get(build_name) if isinstance(allowed_builds, Mapping) else None
            ),
            "compiler_sha256": contract.get("compiler_sha256"),
            "compiler_id": contract.get("compiler_id"),
            "upstream_commit": contract.get("upstream_commit"),
            "upstream_ref": contract.get("upstream_ref"),
            "upstream_repack_sha256": contract.get("upstream_repack_sha256"),
        }
        if not isinstance(build_name, str) or not isinstance(allowed_builds, Mapping) or \
                build_name not in allowed_builds:
            errors.append("build_name is not allowed by the publication manifest")
        for key in provenance_keys:
            if provenance.get(key) != expected_provenance.get(key):
                errors.append(f"{key} does not match immutable publication provenance")
    harness_result = sample.get("harness_result")
    harness_result_sha256 = sample.get("harness_result_sha256")
    if not isinstance(harness_result, Mapping):
        errors.append("exact harness_result is required")
    else:
        expected_result_keys = {
            "schema", "mode", "cache_state", "cpu", "target_working_set_bytes",
            "actual_working_set_bytes", "blocks", "iterations", "calls", "elapsed_ns",
            "checksum", "golden_pass", "golden_cases", "conditioning", "provenance",
        }
        if set(harness_result) != expected_result_keys:
            errors.append("harness_result has unexpected or missing fields")
        result_matches = {
            "schema": HARNESS_SCHEMA,
            "mode": sample.get("mode"),
            "cache_state": sample.get("cache_state"),
            "cpu": sample.get("cpu"),
            "target_working_set_bytes": sample.get("target_working_set_bytes"),
            "actual_working_set_bytes": sample.get("actual_working_set_bytes"),
            "blocks": sample.get("blocks"),
            "iterations": sample.get("iterations"),
            "calls": sample.get("calls"),
            "elapsed_ns": sample.get("elapsed_ns"),
            "checksum": sample.get("checksum"),
            "golden_pass": sample.get("golden_pass"),
            "golden_cases": sample.get("golden_cases"),
            "conditioning": sample.get("cold_conditioning"),
            "provenance": sample.get("provenance"),
        }
        if dict(harness_result) != result_matches:
            errors.append("sample fields do not match the exact harness_result")
        calculated_result_hash = _canonical_mapping_sha256(harness_result)
        if not isinstance(harness_result_sha256, str) or \
                SHA256_RE.fullmatch(harness_result_sha256) is None or \
                harness_result_sha256 != calculated_result_hash:
            errors.append("harness_result_sha256 does not bind the exact harness result")
    if not _is_int(sample.get("exit_code")) or sample.get("exit_code") != 0:
        errors.append("exit_code must be zero")
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
    ref_ns = ns_per_traversal(reference)
    cand_ns = ns_per_traversal(candidate)
    return (ref_ns - cand_ns) / ref_ns


def cache_penalty_ratio(hot: Mapping[str, Any], cold: Mapping[str, Any]) -> float:
    """Compare cold and hot traversals only for one identical control mode."""

    if hot.get("cache_state") != "hot_repeat" or cold.get("cache_state") != "cold_conditioned":
        raise ValueError("expected hot_repeat followed by cold_conditioned")
    keys = ("mode", "cpu", "target_working_set_bytes",
            "actual_working_set_bytes", "blocks")
    if any(hot.get(key) != cold.get(key) for key in keys):
        raise ValueError("cold penalty requires like-for-like samples")
    return ns_per_traversal(cold) / ns_per_traversal(hot)


def infer_bottleneck(manifest_path: str | Path) -> dict[str, Any]:
    """Infer a bottleneck only from one committed, Git-sealed raw bundle."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError(
            "E055 promotion requires one committed raw-bundle manifest path"
        )
    from tooling.e055_raw_bundle import load_sealed_bundle

    bundle = load_sealed_bundle(manifest_path)
    samples = [
        {
            "run_id": sample.run_id,
            "pair_id": sample.pair_id,
            "pair_index": sample.pair_index,
            "pair_order": sample.pair_order,
            "order_index": sample.order_index,
            "build_name": sample.build_name,
            "mode": sample.mode,
            "cache_state": sample.cache_state,
            "cpu": sample.cpu,
            "target_working_set_bytes": sample.target_working_set_bytes,
            "actual_working_set_bytes": sample.actual_working_set_bytes,
            "blocks": sample.blocks,
            "calls": sample.calls,
            "elapsed_ns": sample.elapsed_ns,
            "checksum": sample.checksum,
            "pmu_group": sample.pmu_group,
            "pmu_events": {event.name: event.value for event in sample.pmu_events},
            "measured_elapsed_ns": sample.measured_elapsed_ns,
            "thermal_limit_c": sample.thermal_limit_c,
            "max_temp_c": sample.max_temp_c,
            "publication_identity": sample.publication_identity,
            "commit": sample.commit,
            "tree": sample.tree,
            "manifest_path": sample.manifest_path,
            "manifest_blob_oid": sample.manifest_blob_oid,
        }
        for sample in bundle.samples
    ]
    if not samples:
        raise ValueError("E055 inference requires qualified raw samples")
    phase_identity = {
        (
            sample["build_name"], sample["publication_identity"],
            sample["commit"], sample["tree"], sample["manifest_path"],
            sample["manifest_blob_oid"],
        )
        for sample in samples
    }
    if len(phase_identity) != 1:
        raise ValueError("one E055 phase must use one exact sealed build and Git identity")

    # Promotion is a phase-level decision. It is forbidden until the complete
    # documented 7-size x 2-CPU x 3-mode x 3-PMU-group paired matrix exists.
    expected_matrix_cells = {
        (target, cpu, mode, pmu_group)
        for target in WORKING_SET_BYTES for cpu in CPUS
        for mode in MODES for pmu_group in PMU_GROUP_CONFIGS
    }
    observed_matrix_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for sample in samples:
        observed_matrix_counts[
            (sample["target_working_set_bytes"], sample["cpu"], sample["mode"],
             sample["pmu_group"])
        ] += 1
    if set(observed_matrix_counts) != expected_matrix_cells or \
            len(samples) != len(expected_matrix_cells) * 2 * MIN_PAIRED_SAMPLES or \
            any(observed_matrix_counts[cell] != 2 * MIN_PAIRED_SAMPLES
                for cell in expected_matrix_cells):
        raise ValueError(
            "promotion requires the complete documented 7x2x3x2x3 matrix "
            "with at least five exact hot/cold pairs per cell"
        )

    run_ids = [str(sample["run_id"]) for sample in samples]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id must be globally unique")

    # A cell is one mode/CPU/shape/PMU/provenance combination. Pair IDs may
    # appear exactly twice inside one cell: one qualified hot and one cold row.
    cell_pairs: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_locations: dict[str, tuple[Any, ...]] = {}
    for sample in samples:
        cell = (
            sample["mode"], sample["cpu"], sample["target_working_set_bytes"],
            sample["actual_working_set_bytes"],
            sample["blocks"], sample["pmu_group"], sample["publication_identity"],
        )
        pair_id = str(sample["pair_id"])
        previous = pair_locations.setdefault(pair_id, cell)
        if previous != cell:
            raise ValueError("pair_id must be unique to one mode/shape/PMU cell")
        cell_pairs[cell][pair_id].append(sample)

    cell_summaries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell, pairs in cell_pairs.items():
        pair_indexes: set[int] = set()
        penalties: list[float] = []
        pair_records: list[dict[str, Any]] = []
        event_ratios: dict[str, list[float]] = defaultdict(list)
        for pair_id, pair_rows in pairs.items():
            if len(pair_rows) != 2:
                raise ValueError(f"{pair_id} must contain exactly one hot and one cold row")
            by_state = {str(row["cache_state"]): row for row in pair_rows}
            if set(by_state) != set(CACHE_STATES):
                raise ValueError(f"{pair_id} must contain exactly one hot and one cold row")
            hot = by_state["hot_repeat"]
            cold = by_state["cold_conditioned"]
            if len({row["cache_state"] for row in pair_rows}) != 2:
                raise ValueError(f"{pair_id} contains a duplicate pair half")
            match_keys = (
                "pair_index", "pair_order", "mode", "cpu", "build_name",
                "target_working_set_bytes", "actual_working_set_bytes", "blocks",
                "pmu_group", "publication_identity", "commit", "tree",
                "manifest_path", "manifest_blob_oid",
            )
            if any(hot[key] != cold[key] for key in match_keys):
                raise ValueError(f"{pair_id} hot/cold metadata mismatch")
            pair_index = int(hot["pair_index"])
            if pair_index in pair_indexes:
                raise ValueError("pair_index must be unique inside each cell")
            pair_indexes.add(pair_index)
            expected_order = "hot_then_cold" if pair_index % 2 else "cold_then_hot"
            if hot["pair_order"] != expected_order:
                raise ValueError("pair_order must alternate by pair_index")
            penalty = cache_penalty_ratio(hot, cold)
            penalties.append(penalty)
            if set(hot["pmu_events"]) != set(cold["pmu_events"]):
                raise ValueError(f"{pair_id} PMU event sets differ")
            event_record: dict[str, dict[str, float | None]] = {}
            for event_name in sorted(hot["pmu_events"]):
                hot_count = float(hot["pmu_events"][event_name]) / float(hot["calls"])
                cold_count = float(cold["pmu_events"][event_name]) / float(cold["calls"])
                ratio = cold_count / hot_count if hot_count > 0.0 else None
                event_record[event_name] = {
                    "hot_count_per_traversal": hot_count,
                    "cold_count_per_traversal": cold_count,
                    "cold_over_hot": ratio,
                }
                if ratio is not None:
                    event_ratios[event_name].append(ratio)
            pair_records.append({
                "pair_id": pair_id,
                "pair_index": pair_index,
                "cold_over_hot": penalty,
                "pmu_event_counts": event_record,
            })
        if pair_indexes != set(range(1, MIN_PAIRED_SAMPLES + 1)):
            raise ValueError("pair_index sequence must be exactly one through five")
        cell_summaries[cell] = {
            "paired_samples": len(penalties),
            "median_pair_penalty": statistics.median(penalties),
            "pair_penalties": sorted(pair_records, key=lambda item: item["pair_index"]),
            "pmu_event_median_cold_over_hot": {
                name: statistics.median(ratios)
                for name, ratios in sorted(event_ratios.items()) if ratios
            },
        }

    records: list[dict[str, Any]] = []
    memory_ratios: list[float] = []
    control_tracking_ratios: list[float] = []
    coordinates: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell, summary in cell_summaries.items():
        mode, cpu, target, size, blocks, pmu_group, identity = cell
        coordinates[(cpu, target, size, blocks, pmu_group, identity)][mode] = summary
    for (cpu, target, size, blocks, pmu_group, _), mode_summaries in sorted(
        coordinates.items(), key=lambda item: str(item[0])
    ):
        if set(mode_summaries) != set(MODES):
            continue
        penalties = {mode: mode_summaries[mode]["median_pair_penalty"] for mode in MODES}
        full_penalty = penalties["full_dotprod"]
        control_penalty = statistics.median(
            [penalties["packed_stream"], penalties["unpack_scale"]]
        )
        tracking = full_penalty / control_penalty
        memory_ratios.append(full_penalty)
        control_tracking_ratios.append(tracking)
        records.append({
            "kind": "median_of_per_pair_cold_penalties",
            "cpu": cpu,
            "target_working_set_bytes": target,
            "actual_working_set_bytes": size,
            "blocks": blocks,
            "pmu_group": pmu_group,
            "paired_samples_per_mode": {
                mode: mode_summaries[mode]["paired_samples"] for mode in MODES
            },
            "full_cold_over_hot": full_penalty,
            "packed_stream_cold_over_hot": penalties["packed_stream"],
            "unpack_scale_cold_over_hot": penalties["unpack_scale"],
            "full_over_median_control_penalty": tracking,
            "pair_penalties": {
                mode: mode_summaries[mode]["pair_penalties"] for mode in MODES
            },
            "pmu_event_median_cold_over_hot": {
                mode: mode_summaries[mode]["pmu_event_median_cold_over_hot"]
                for mode in MODES
            },
        })

    if not cell_summaries:
        classification = "insufficient_samples"
        confidence = 0.0
    elif memory_ratios and control_tracking_ratios:
        median_memory = statistics.median(memory_ratios)
        median_tracking = statistics.median(control_tracking_ratios)
        classification = (
            "memory_cache_sensitive"
            if median_memory > 1.0 and 0.80 <= median_tracking <= 1.25
            else "unpack_compute_sensitive"
        )
        confidence = min(1.0, 0.5 + 0.1 * min(len(memory_ratios), 5) +
                         0.1 * min(len(control_tracking_ratios), 5))
    else:
        classification = "insufficient_control_cells"
        confidence = 0.0
    return {
        "schema": "e055-q1-inference/v1",
        "classification": classification,
        "confidence": confidence,
        "matrix_complete": True,
        "matrix_expected_cells": len(expected_matrix_cells),
        "matrix_qualified_rows": len(samples),
        "cell_count": len(cell_summaries),
        "memory_cache_sensitivity_ratios": memory_ratios,
        "full_to_control_cold_penalty_ratio_of_ratios": control_tracking_ratios,
        "records": records,
        "promotion": {
            "threshold_fraction": PROMOTION_THRESHOLD,
            "projected_q1_gain_fraction": (
                max(0.0, 1.0 - 1.0 / statistics.median(memory_ratios))
                if memory_ratios else 0.0
            ),
            "eligible": bool(
                classification == "memory_cache_sensitive" and memory_ratios and
                max(0.0, 1.0 - 1.0 / statistics.median(memory_ratios)) >=
                PROMOTION_THRESHOLD
            ),
            "semantics": (
                "Optimistic component bound from making full cold traversal equal hot; "
                "not an end-to-end model speedup claim."
            ),
        },
        "interpretation": (
            "Directional microbenchmark inference only; PMU values are event counts, "
            "not DDR bytes. A full-model decision requires the E049 per-token trace."
        ),
    }


def rows_to_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return a compact CSV-like table for raw publication and review."""

    columns = (
        "mode", "cache_state", "cpu", "actual_working_set_bytes", "blocks",
        "elapsed_ns", "calls", "ns_per_traversal", "traversals_per_second",
        "checksum", "golden_pass",
    )
    output = [",".join(columns)]
    for row in rows:
        normalized = dict(row)
        normalized["ns_per_traversal"] = ns_per_traversal(row)
        normalized["traversals_per_second"] = traversals_per_second(row)
        output.append(",".join(str(normalized.get(column, "")) for column in columns))
    return "\n".join(output) + "\n"
