import json
import math
import unittest
from pathlib import Path

from tooling.e048_trace import (
    TRACE_SCHEMA,
    aggregate_trace,
    classify_trace_ab,
    q1_gemv_byte_ledger,
    unique_byte_ranges,
    validate_trace_events,
)


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "patches/llama.cpp/38c66ad/e048-per-op-trace.patch"


class E048TraceTest(unittest.TestCase):
    def test_q1_4x8_ledger_separates_logical_reuse_from_unique_activation(self):
        ledger = q1_gemv_byte_ledger(n=5120, nc=17408)
        blocks = 5120 // 128
        groups = 17408 // 4
        self.assertEqual(ledger["q1_packed_weight_read_bytes"], blocks * groups * 72)
        self.assertEqual(ledger["q8_activation_logical_read_bytes"], blocks * groups * 4 * 34)
        self.assertEqual(ledger["q8_activation_unique_read_bytes"], blocks * 4 * 34)
        self.assertEqual(ledger["output_write_bytes"], 17408 * 4)

    def test_unique_ranges_merge_aliases_without_counting_overlap_twice(self):
        self.assertEqual(
            unique_byte_ranges([(100, 140), (120, 180), (200, 208)]),
            88,
        )

    def test_trace_validation_rejects_observed_ddr_without_direct_counter(self):
        events = [
            {
                "schema": TRACE_SCHEMA,
                "event": "token_begin",
                "token_seq": 0,
                "start_ns": 100,
                "observed_ddr_read_bytes": 12,
                "observed_ddr_write_bytes": None,
                "counter_method": "none",
            }
        ]
        errors = validate_trace_events(events)
        self.assertTrue(any("observed_ddr_read_bytes" in error for error in errors))

    def test_aggregate_uses_post_barrier_end_for_node_wall_span(self):
        events = [
            {"schema": TRACE_SCHEMA, "event": "token_begin", "token_seq": 0, "start_ns": 10},
            {
                "schema": TRACE_SCHEMA,
                "event": "node_worker",
                "token_seq": 0,
                "node_index": 7,
                "layer_index": 3,
                "op_name": "MUL_MAT",
                "worker_ith": 0,
                "start_ns": 100,
                "end_ns": 300,
                "post_barrier_end_ns": 500,
                "logical_read_bytes": 72,
                "logical_write_bytes": 16,
                "observed_ddr_read_bytes": None,
                "observed_ddr_write_bytes": None,
                "counter_method": "none",
            },
            {
                "schema": TRACE_SCHEMA,
                "event": "node_worker",
                "token_seq": 0,
                "node_index": 7,
                "layer_index": 3,
                "op_name": "MUL_MAT",
                "worker_ith": 1,
                "start_ns": 120,
                "end_ns": 260,
                "post_barrier_end_ns": 620,
                "logical_read_bytes": 72,
                "logical_write_bytes": 16,
                "observed_ddr_read_bytes": None,
                "observed_ddr_write_bytes": None,
                "counter_method": "none",
            },
            {"schema": TRACE_SCHEMA, "event": "token_end", "token_seq": 0, "end_ns": 700},
        ]
        self.assertEqual(validate_trace_events(events), [])
        aggregate = aggregate_trace(events)["nodes"][0]
        self.assertEqual(aggregate["node_index"], 7)
        self.assertEqual(aggregate["wall_end_ns"], 620)
        self.assertEqual(aggregate["wall_duration_ns"], 520)

    def test_q1_and_f32_to_q8_events_are_valid_trace_kinds(self):
        events = [
            {"schema": TRACE_SCHEMA, "event": "token_begin", "token_seq": 1, "start_ns": 10,
             "observed_ddr_read_bytes": None, "observed_ddr_write_bytes": None, "counter_method": "none"},
            {"schema": TRACE_SCHEMA, "event": "q1_kernel", "token_seq": 1, "node_index": 3,
             "start_ns": 20, "end_ns": 40, "q1_packed_weight_read_bytes": 72,
             "q8_activation_logical_read_bytes": 136, "q8_activation_unique_read_bytes": 136,
             "output_write_bytes": 16, "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None, "counter_method": "none"},
            {"schema": TRACE_SCHEMA, "event": "phase_worker", "token_seq": 1, "node_index": 3,
             "start_ns": 41, "end_ns": 50, "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None, "counter_method": "none"},
            {"schema": TRACE_SCHEMA, "event": "token_end", "token_seq": 1, "end_ns": 60,
             "observed_ddr_read_bytes": None, "observed_ddr_write_bytes": None, "counter_method": "none"},
        ]
        self.assertEqual(validate_trace_events(events), [])
        aggregate = aggregate_trace(events)
        self.assertEqual(len(aggregate["q1_kernels"]), 1)
        self.assertEqual(len(aggregate["phases"]), 1)

    def test_trace_ab_above_one_percent_is_trace_only(self):
        result = classify_trace_ab([1000, 1002, 998, 1001, 999], [1015, 1016, 1014, 1017, 1013], token_stream_equal=True)
        self.assertEqual(result["classification"], "TRACE_ONLY")
        self.assertGreater(result["median_overhead_fraction"], 0.01)

    def test_patch_targets_current_38c66ad_seams(self):
        patch = PATCH_PATH.read_text(encoding="utf-8")
        required = (
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu/ggml-cpu-impl.h",
            "ggml/src/ggml-cpu/ggml-cpu.cpp",
            "ggml/include/ggml-cpu.h",
            "ggml/src/ggml-cpu/arch/arm/repack.cpp",
            "src/llama-context.cpp",
            "ggml_graph_compute_thread",
            "ggml_gemv_q1_0_4x4_q8_0",
            "ggml_gemv_q1_0_4x8_q8_0",
            "GGML_CPU_DISABLE_FUSION",
        )
        for needle in required:
            self.assertIn(needle, patch)


if __name__ == "__main__":
    unittest.main()
