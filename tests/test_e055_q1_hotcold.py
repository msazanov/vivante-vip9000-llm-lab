"""Adversarial host-side contract tests for E055 qualification."""

from __future__ import annotations

import unittest

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


def qualified_sample(cache_state: str = "hot_repeat", calls: int = 250) -> dict:
    blocks = 316
    event_names = ["cpu_cycles", "instructions", "stall_backend"]
    return {
        "schema": "e055-q1-hot-cold/v2",
        "run_id": "run-001",
        "pair_id": "pair-001",
        "pair_order": "hot_then_cold",
        "order_index": 1 if cache_state == "hot_repeat" else 2,
        "mode": "full_dotprod",
        "cache_state": cache_state,
        "cpu": 6,
        "pinned_cpu": 6,
        "affinity_cpus": [6],
        "cpu_start": 6,
        "cpu_end": 6,
        "cpu_migration_count": 0,
        "golden_pass": True,
        "cold_conditioning": {"verified_touched": cache_state == "cold_conditioned"},
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
            "event_group": "core",
            "event_group_size": 3,
            "values_are_event_counts": True,
            "events": [{"name": name, "support": "supported", "sample_valid": True,
                        "value": 10, "running_ratio": 1.0} for name in event_names],
        },
        "thermal": {"readable": True, "tripped": False, "max_temp_c": 61.0,
                    "limit_c": 80.0},
        "provenance": {"source_sha256": "a" * 64, "binary_sha256": "b" * 64,
                       "compiler_sha256": "c" * 64, "compiler_id": "gcc-test"},
        "exit_code": 0,
    }


class E055Q1HotColdContractTest(unittest.TestCase):
    def test_working_set_and_declared_bytes_match_stock_carrier(self) -> None:
        result = actual_working_set(64 * 1024)
        self.assertEqual(result["actual_bytes"] % 208, 0)
        counts = logical_bytes_per_call(40)
        self.assertEqual(counts["total_input_bytes"], 40 * 208)
        self.assertEqual(counts["dot_products"], 40 * 512)

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

    def test_cold_penalty_rejects_unlike_controls(self) -> None:
        hot = qualified_sample("hot_repeat", 250)
        cold = qualified_sample("cold_conditioned", 1)
        cold["mode"] = "packed_stream"
        with self.assertRaises(ValueError):
            cache_penalty_ratio(hot, cold)

    def test_fully_qualified_sample_passes(self) -> None:
        self.assertEqual(validate_sample(qualified_sample()), [])
        self.assertEqual(validate_sample(qualified_sample("cold_conditioned", 1)), [])

    def test_strict_validator_rejects_every_reviewed_failure(self) -> None:
        mutations = {
            "sync=false": lambda s: s["sync"].update(requested=False),
            "empty events": lambda s: s["pmu"].update(events=[]),
            "wrong PMU group": lambda s: s["pmu"].update(event_group="memory"),
            "duplicate PMU event": lambda s: s["pmu"]["events"].append(
                dict(s["pmu"]["events"][0])),
            "multiplexed PMU": lambda s: s["pmu"]["events"][0].update(running_ratio=0.999),
            "invalid PMU sample": lambda s: s["pmu"].update(sample_valid=False),
            "thermal trip": lambda s: s["thermal"].update(tripped=True),
            "nonfinite thermal": lambda s: s["thermal"].update(max_temp_c=float("nan")),
            "CPU migration": lambda s: s.update(cpu_end=0, cpu_migration_count=1),
            "missing pair": lambda s: s.pop("pair_id"),
            "wrong pair order": lambda s: s.update(order_index=2),
            "bad source hash": lambda s: s["provenance"].update(source_sha256="bad"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                sample = qualified_sample()
                mutate(sample)
                self.assertTrue(validate_sample(sample), name)

    def test_cold_qualification_rejects_more_than_one_traversal(self) -> None:
        sample = qualified_sample("cold_conditioned", 2)
        self.assertTrue(any("exactly one" in error for error in validate_sample(sample)))

    def test_ddr_bytes_are_never_inferred_from_event_counts(self) -> None:
        sample = qualified_sample()
        sample["observed_ddr_read_bytes"] = 123
        self.assertTrue(any("DDR" in error for error in validate_sample(sample)))

    def test_summary_and_public_matrix(self) -> None:
        summary = summarize_samples([100.0, 110.0, 90.0, 100.0, 100.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 100.0)
        self.assertEqual(MODES, ("packed_stream", "unpack_scale", "full_dotprod"))
        self.assertEqual(CACHE_STATES, ("hot_repeat", "cold_conditioned"))
        self.assertEqual(len(WORKING_SET_BYTES), 7)

    def test_inference_uses_normalized_like_for_like_penalties(self) -> None:
        rows = []
        for mode in MODES:
            for state in CACHE_STATES:
                for repeat in range(5):
                    calls = 250 if state == "hot_repeat" else 1
                    ns_each = 1_000_000 if state == "hot_repeat" else 2_000_000
                    rows.append({"mode": mode, "cache_state": state, "cpu": 6,
                                 "actual_working_set_bytes": 316 * 208,
                                 "blocks": 316, "calls": calls,
                                 "elapsed_ns": calls * ns_each + repeat,
                                 "pair_id": f"pair-{repeat}",
                                 "pmu": {"event_group": "core"}})
        result = infer_bottleneck(rows)
        self.assertEqual(result["classification"], "memory_cache_sensitive")
        self.assertAlmostEqual(
            result["full_to_control_cold_penalty_ratio_of_ratios"][0], 1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
