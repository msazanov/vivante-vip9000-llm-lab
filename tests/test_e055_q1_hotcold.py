"""Adversarial host-side contract tests for E055 qualification."""

from __future__ import annotations

import copy
import unittest

import tooling.e055_q1_hotcold as e055
from tooling.e055_q1_hotcold import (
    CACHE_STATES,
    MODES,
    WORKING_SET_BYTES,
    actual_working_set,
    cache_penalty_ratio,
    infer_bottleneck,
    logical_bytes_per_call,
    ns_per_traversal,
    paired_speed_delta,
    summarize_samples,
    traversals_per_second,
    validate_sample,
)


EXPECTED_PROVENANCE = {
    "source_sha256": "11" * 32,
    "binary_sha256": "22" * 32,
    "compiler_sha256": "33" * 32,
    "compiler_id": "aarch64-linux-gnu-g++ test fixture",
    "upstream_repack_sha256": "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d",
}


def qualified_sample(
    cache_state: str = "hot_repeat",
    calls: int = 250,
    *,
    mode: str = "full_dotprod",
    pair_index: int = 1,
    cpu: int = 6,
    event_group: str = "core",
) -> dict:
    blocks = 316
    pair_order = "hot_then_cold" if pair_index % 2 else "cold_then_hot"
    order_index = 1 if (
        (pair_order == "hot_then_cold" and cache_state == "hot_repeat") or
        (pair_order == "cold_then_hot" and cache_state == "cold_conditioned")
    ) else 2
    names = {
        "core": ["cpu_cycles", "instructions", "stall_backend"],
        "cache": ["l1d_cache_refill", "l2d_cache_refill", "l3d_cache_refill"],
        "memory": ["mem_access", "bus_access"],
    }[event_group]
    return {
        "schema": "e055-q1-hot-cold/v2",
        "run_id": f"run-{mode}-{event_group}-{pair_index}-{cache_state}",
        "pair_id": f"pair-{mode}-{event_group}-{pair_index}",
        "pair_index": pair_index,
        "pair_order": pair_order,
        "order_index": order_index,
        "mode": mode,
        "cache_state": cache_state,
        "cpu": cpu,
        "pinned_cpu": cpu,
        "affinity_cpus": [cpu],
        "cpu_start": cpu,
        "cpu_end": cpu,
        "cpu_migration_count": 0,
        "golden_pass": True,
        "cold_conditioning": {"verified_touched": True},
        "actual_working_set_bytes": blocks * 208,
        "blocks": blocks,
        "iterations": calls,
        "elapsed_ns": calls * 1_000_000,
        "calls": calls,
        "checksum": "0x1",
        "sync": {"requested": True, "started": True, "acknowledged": True,
                 "ended": True, "sequence": "S/A/E"},
        "pmu": {
            "schema_version": "e049c-arm-pmu/v2",
            "status": "ok",
            "sample_valid": True,
            "event_group": event_group,
            "event_group_size": len(names),
            "values_are_event_counts": True,
            "events": [{"name": name, "support": "supported", "sample_valid": True,
                        "value": 10, "running_ratio": 1.0} for name in names],
        },
        "thermal": {"readable": True, "tripped": False, "max_temp_c": 61.0,
                    "limit_c": 80.0},
        "provenance": dict(EXPECTED_PROVENANCE),
        "exit_code": 0,
    }


def qualified_matrix(penalties: dict[str, list[float]] | None = None) -> list[dict]:
    penalties = penalties or {mode: [2.0] * 5 for mode in MODES}
    rows = []
    for mode in MODES:
        for pair_index in range(1, 6):
            hot = qualified_sample("hot_repeat", 250, mode=mode, pair_index=pair_index)
            cold = qualified_sample("cold_conditioned", 1, mode=mode, pair_index=pair_index)
            hot["elapsed_ns"] = 250 * 1_000_000
            cold["elapsed_ns"] = round(penalties[mode][pair_index - 1] * 1_000_000)
            rows.extend((hot, cold))
    return rows


class E055Q1HotColdContractTest(unittest.TestCase):
    def test_working_set_and_declared_bytes_match_stock_carrier(self) -> None:
        result = actual_working_set(64 * 1024)
        self.assertEqual(result["actual_bytes"] % 208, 0)
        counts = logical_bytes_per_call(40)
        self.assertEqual(counts["total_input_bytes"], 40 * 208)
        self.assertEqual(counts["dot_products"], 40 * 512)

    def test_working_set_rejects_uint64_rounding_overflow(self) -> None:
        with self.assertRaises(ValueError):
            actual_working_set((1 << 64) - 207)

    def test_public_matrix_contains_exact_binary_12_5_mib(self) -> None:
        self.assertEqual(WORKING_SET_BYTES[-1], 13_107_200)

    def test_upstream_repack_is_pinned_to_reviewed_hash(self) -> None:
        self.assertEqual(getattr(e055, "UPSTREAM_REPACK_SHA256", None),
                         EXPECTED_PROVENANCE["upstream_repack_sha256"])

    def test_normalization_exactly_handles_250_hot_calls_vs_one_cold_call(self) -> None:
        hot = qualified_sample("hot_repeat", 250)
        cold = qualified_sample("cold_conditioned", 1)
        self.assertEqual(ns_per_traversal(hot), 1_000_000.0)
        self.assertEqual(ns_per_traversal(cold), 1_000_000.0)
        self.assertEqual(traversals_per_second(hot), 1000.0)
        self.assertEqual(cache_penalty_ratio(hot, cold), 1.0)

    def test_paired_delta_normalizes_calls_and_rejects_unlike_modes(self) -> None:
        reference = qualified_sample(calls=250)
        candidate = qualified_sample(calls=1)
        candidate["elapsed_ns"] = 900_000
        self.assertAlmostEqual(paired_speed_delta(reference, candidate), 0.1)
        candidate["mode"] = "unpack_scale"
        with self.assertRaises(ValueError):
            paired_speed_delta(reference, candidate)

    def test_fully_qualified_sample_requires_declared_provenance_binding(self) -> None:
        self.assertTrue(validate_sample(qualified_sample()))
        self.assertEqual(validate_sample(qualified_sample(), EXPECTED_PROVENANCE), [])
        self.assertEqual(
            validate_sample(qualified_sample("cold_conditioned", 1), EXPECTED_PROVENANCE), []
        )

    def test_strict_validator_rejects_reviewed_failures(self) -> None:
        mutations = {
            "sync=false": lambda s: s["sync"].update(requested=False),
            "empty events": lambda s: s["pmu"].update(events=[]),
            "wrong PMU group": lambda s: s["pmu"].update(event_group="memory"),
            "duplicate PMU event": lambda s: s["pmu"]["events"].append(
                dict(s["pmu"]["events"][0])),
            "negative PMU count": lambda s: s["pmu"]["events"][0].update(value=-1),
            "boolean PMU count": lambda s: s["pmu"]["events"][0].update(value=True),
            "integer ratio": lambda s: s["pmu"]["events"][0].update(running_ratio=1),
            "boolean ratio": lambda s: s["pmu"]["events"][0].update(running_ratio=True),
            "multiplexed PMU": lambda s: s["pmu"]["events"][0].update(running_ratio=0.999),
            "invalid PMU sample": lambda s: s["pmu"].update(sample_valid=False),
            "thermal trip": lambda s: s["thermal"].update(tripped=True),
            "nonfinite thermal": lambda s: s["thermal"].update(max_temp_c=float("nan")),
            "CPU migration": lambda s: s.update(cpu_end=0, cpu_migration_count=1),
            "missing pair": lambda s: s.pop("pair_id"),
            "missing pair index": lambda s: s.pop("pair_index"),
            "wrong pair order": lambda s: s.update(order_index=2),
            "zero source hash": lambda s: s["provenance"].update(source_sha256="0" * 64),
            "unbound source hash": lambda s: s["provenance"].update(source_sha256="4" * 64),
            "wrong repack hash": lambda s: s["provenance"].update(upstream_repack_sha256="5" * 64),
            "uppercase checksum": lambda s: s.update(checksum="0xAB"),
            "malformed checksum": lambda s: s.update(checksum="not-a-checksum"),
            "unverified hot conditioning": lambda s: s["cold_conditioning"].update(
                verified_touched=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                sample = qualified_sample()
                mutate(sample)
                self.assertTrue(validate_sample(sample, EXPECTED_PROVENANCE), name)

    def test_cold_qualification_rejects_more_than_one_traversal(self) -> None:
        sample = qualified_sample("cold_conditioned", 2)
        self.assertTrue(any("exactly one" in error for error in
                            validate_sample(sample, EXPECTED_PROVENANCE)))

    def test_analyzer_uses_median_of_actual_per_pair_penalties(self) -> None:
        # For each mode, median(cold)/median(hot) is 100, but the five actual
        # pair penalties are [1, 100, 100, 1, 0.01], whose median is exactly 1.
        rows = []
        hot_ns = [1, 1, 1, 100, 100]
        cold_ns = [1, 100, 100, 100, 1]
        for mode in MODES:
            for index, (hot_value, cold_value) in enumerate(zip(hot_ns, cold_ns), 1):
                hot = qualified_sample("hot_repeat", 1, mode=mode, pair_index=index)
                cold = qualified_sample("cold_conditioned", 1, mode=mode, pair_index=index)
                hot["elapsed_ns"] = hot_value
                cold["elapsed_ns"] = cold_value
                rows.extend((hot, cold))
        result = infer_bottleneck(rows, EXPECTED_PROVENANCE)
        self.assertEqual(result["memory_cache_sensitivity_ratios"], [1.0])
        self.assertEqual(result["records"][0]["full_cold_over_hot"], 1.0)

    def test_analyzer_rejects_invalid_or_unpaired_rows(self) -> None:
        cases = {}
        invalid = qualified_matrix()
        invalid[0]["sync"]["ended"] = False
        cases["invalid strict sample"] = invalid
        duplicate_run = qualified_matrix()
        duplicate_run[1]["run_id"] = duplicate_run[0]["run_id"]
        cases["duplicate run ID"] = duplicate_run
        missing_half = qualified_matrix()[:-1]
        cases["missing pair half"] = missing_half
        duplicate_half = qualified_matrix()
        duplicate_half.append(copy.deepcopy(duplicate_half[0]))
        duplicate_half[-1]["run_id"] = "extra-run"
        cases["duplicate pair half"] = duplicate_half
        mismatched = qualified_matrix()
        mismatched[1]["blocks"] += 1
        mismatched[1]["actual_working_set_bytes"] += 208
        cases["mismatched pair shape"] = mismatched
        for name, rows in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                infer_bottleneck(rows, EXPECTED_PROVENANCE)

    def test_analyzer_enforces_four_percent_promotion_threshold(self) -> None:
        below = infer_bottleneck(qualified_matrix({mode: [1.04] * 5 for mode in MODES}),
                                 EXPECTED_PROVENANCE)
        above = infer_bottleneck(qualified_matrix({mode: [1.05] * 5 for mode in MODES}),
                                 EXPECTED_PROVENANCE)
        self.assertEqual(below["promotion"]["threshold_fraction"], 0.04)
        self.assertFalse(below["promotion"]["eligible"])
        self.assertTrue(above["promotion"]["eligible"])

    def test_summary_and_public_matrix(self) -> None:
        summary = summarize_samples([100.0, 110.0, 90.0, 100.0, 100.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 100.0)
        self.assertEqual(MODES, ("packed_stream", "unpack_scale", "full_dotprod"))
        self.assertEqual(CACHE_STATES, ("hot_repeat", "cold_conditioned"))
        self.assertEqual(len(WORKING_SET_BYTES), 7)


if __name__ == "__main__":
    unittest.main()
