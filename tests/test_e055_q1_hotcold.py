"""Host-side contract tests for E055 Q1 hot/cold inference."""

from __future__ import annotations

import unittest

from tooling.e055_q1_hotcold import (
    CACHE_STATES,
    MODES,
    WORKING_SET_BYTES,
    actual_working_set,
    logical_bytes_per_call,
    paired_speed_delta,
    summarize_samples,
    validate_sample,
)


class E055Q1HotColdContractTest(unittest.TestCase):
    def test_working_set_rounds_to_complete_native_block_carriers(self) -> None:
        result = actual_working_set(64 * 1024)
        self.assertEqual(result["target_bytes"], 64 * 1024)
        self.assertGreaterEqual(result["actual_bytes"], result["target_bytes"])
        self.assertEqual(result["actual_bytes"] % 208, 0)
        self.assertGreater(result["blocks"], 0)

    def test_declared_bytes_match_native_q1_q8_traversal(self) -> None:
        result = logical_bytes_per_call(40)
        self.assertEqual(result["q1_packed_bytes"], 40 * 72)
        self.assertEqual(result["q8_bytes"], 40 * 4 * 34)
        self.assertEqual(result["total_input_bytes"], 40 * 208)
        self.assertEqual(result["dot_products"], 40 * 512)

    def test_contract_exposes_only_declared_event_counts_not_bytes(self) -> None:
        sample = {
            "schema": "e055-q1-hot-cold/v1",
            "mode": "full_dotprod",
            "cache_state": "hot_repeat",
            "cpu": 6,
            "golden_pass": True,
            "cold_conditioning": {"verified_touched": True},
            "actual_working_set_bytes": 65728,
            "blocks": 316,
            "iterations": 10,
            "elapsed_ns": 1000000,
            "calls": 10,
            "checksum": "0x1",
            "pmu": {
                "events": [{"name": "mem_access", "value": 10}],
                "values_are_event_counts": True,
            },
        }
        self.assertEqual(validate_sample(sample), [])
        bad = dict(sample)
        bad["observed_ddr_read_bytes"] = 123
        self.assertTrue(any("DDR" in error for error in validate_sample(bad)))

    def test_validation_rejects_unknown_modes_or_unverified_cold(self) -> None:
        sample = {
            "schema": "e055-q1-hot-cold/v1",
            "mode": "not-a-mode",
            "cache_state": "cold_conditioned",
            "cpu": 0,
            "golden_pass": True,
            "cold_conditioning": {"verified_touched": False},
            "actual_working_set_bytes": 65728,
            "blocks": 316,
            "iterations": 1,
            "elapsed_ns": 100,
            "calls": 1,
            "checksum": "0x1",
            "pmu": {"events": [], "values_are_event_counts": True},
        }
        errors = validate_sample(sample)
        self.assertTrue(any("mode" in error for error in errors))
        self.assertTrue(any("cold" in error.lower() for error in errors))

    def test_validation_rejects_zero_checksum(self) -> None:
        sample = {
            "schema": "e055-q1-hot-cold/v1",
            "mode": "packed_stream",
            "cache_state": "hot_repeat",
            "cpu": 0,
            "golden_pass": True,
            "cold_conditioning": {"verified_touched": False},
            "actual_working_set_bytes": 65728,
            "blocks": 316,
            "iterations": 1,
            "elapsed_ns": 100,
            "calls": 1,
            "checksum": "0x0",
            "pmu": {"events": [], "values_are_event_counts": True},
        }
        self.assertTrue(any("checksum" in error for error in validate_sample(sample)))

    def test_paired_delta_never_subtracts_incomparable_modes(self) -> None:
        reference = {"mode": "full_dotprod", "cache_state": "hot_repeat",
                     "cpu": 6, "actual_working_set_bytes": 65728,
                     "elapsed_ns": 1000}
        candidate = dict(reference)
        candidate["elapsed_ns"] = 900
        self.assertAlmostEqual(paired_speed_delta(reference, candidate), 0.1)
        incomparable = dict(candidate)
        incomparable["mode"] = "unpack_scale"
        with self.assertRaises(ValueError):
            paired_speed_delta(reference, incomparable)

    def test_summary_reports_repeat_count_and_variability(self) -> None:
        summary = summarize_samples([100.0, 110.0, 90.0, 100.0, 100.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 100.0)
        self.assertGreaterEqual(summary["cv"], 0.0)

    def test_public_matrix_is_bounded_and_named(self) -> None:
        self.assertEqual(
            WORKING_SET_BYTES,
            (64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
             1 * 1024 * 1024, 4 * 1024 * 1024, 12_500 * 1024),
        )
        self.assertEqual(MODES, ("packed_stream", "unpack_scale", "full_dotprod"))
        self.assertEqual(CACHE_STATES, ("hot_repeat", "cold_conditioned"))


if __name__ == "__main__":
    unittest.main()
