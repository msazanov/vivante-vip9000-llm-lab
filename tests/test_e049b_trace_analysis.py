import unittest

from tooling.e049b_trace_analysis import (
    aggregate_run_events,
    family_for_q1_variant,
    interval_duration,
    merge_intervals,
    percentile,
    summarize_samples,
)


class E049bTraceAnalysisTest(unittest.TestCase):
    def test_merge_intervals_and_duration_do_not_sum_parallel_workers(self):
        intervals = merge_intervals([(100, 300), (120, 260), (400, 450)])
        self.assertEqual(intervals, [(100, 300), (400, 450)])
        self.assertEqual(interval_duration(intervals), 250)

    def test_q1_variant_family_preserves_actual_4x4_and_4x8(self):
        self.assertEqual(family_for_q1_variant("q1_0_4x4_q8_0"), "q1_gemv_4x4")
        self.assertEqual(family_for_q1_variant("q1_0_4x8_q8_0"), "q1_gemv_4x8")
        self.assertEqual(family_for_q1_variant("unknown"), "q1_gemv_unknown")

    def test_token_categories_use_wall_unions_and_partition_token(self):
        events = [
            {"schema": "e048-trace/v1", "event": "token_begin", "token_seq": 3,
             "start_ns": 0, "steady_decode": True,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "node_worker", "token_seq": 3,
             "node_index": 7, "layer_index": 2, "worker_ith": 0, "fused_nodes": 1,
             "op_name": "MUL_MAT", "tensor_id": "x", "start_ns": 100,
             "end_ns": 300, "post_barrier_end_ns": 350, "cpu_start": 0,
             "cpu_end": 0, "logical_read_bytes": 100, "logical_write_bytes": 4,
             "unique_read_bytes": 100, "unique_write_bytes": 4,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "node_worker", "token_seq": 3,
             "node_index": 7, "layer_index": 2, "worker_ith": 1, "fused_nodes": 1,
             "op_name": "MUL_MAT", "tensor_id": "x", "start_ns": 120,
             "end_ns": 280, "post_barrier_end_ns": 380, "cpu_start": 1,
             "cpu_end": 1, "logical_read_bytes": 0, "logical_write_bytes": 0,
             "unique_read_bytes": 0, "unique_write_bytes": 0,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "q1_kernel", "token_seq": 3,
             "node_index": 7, "worker_ith": 0, "variant": "q1_0_4x8_q8_0",
             "n": 128, "nr": 1, "nc": 8, "start_ns": 150, "end_ns": 250,
             "cpu_start": 0, "cpu_end": 0, "q1_packed_weight_read_bytes": 72,
             "q8_activation_logical_read_bytes": 136,
             "q8_activation_unique_read_bytes": 136, "output_write_bytes": 32,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "token_end", "token_seq": 3,
             "end_ns": 500, "status": 0, "overflow_count": 0,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
        ]
        result = aggregate_run_events(events)
        token = result["tokens"][0]
        self.assertEqual(token["duration_ns"], 500)
        self.assertEqual(token["categories_ns"]["q1_gemv"], 100)
        self.assertEqual(token["categories_ns"]["barrier_sync"], 80)
        self.assertEqual(sum(token["categories_ns"].values()), 500)
        self.assertEqual(result["nodes"][0]["wall_duration_ns"], 280)
        self.assertEqual(result["nodes"][0]["worker_sum_duration_ns"], 360)

    def test_fused_node_is_retained_in_group_identity(self):
        events = [
            {"schema": "e048-trace/v1", "event": "token_begin", "token_seq": 3,
             "start_ns": 0, "steady_decode": True,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "node_worker", "token_seq": 3,
             "node_index": 1, "layer_index": 4, "worker_ith": 0, "fused_nodes": 2,
             "op_name": "RMS_NORM", "tensor_id": "x", "start_ns": 10,
             "end_ns": 20, "post_barrier_end_ns": 22, "cpu_start": 0,
             "cpu_end": 0, "logical_read_bytes": 8, "logical_write_bytes": 4,
             "unique_read_bytes": 8, "unique_write_bytes": 4,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
            {"schema": "e048-trace/v1", "event": "token_end", "token_seq": 3,
             "end_ns": 30, "status": 0, "overflow_count": 0,
             "counter_method": "none", "observed_ddr_read_bytes": None,
             "observed_ddr_write_bytes": None},
        ]
        row = aggregate_run_events(events)["nodes"][0]
        self.assertTrue(row["fused"])
        self.assertEqual(row["fused_nodes"], 2)

    def test_statistics_are_five_run_samples_with_percentiles_and_cv(self):
        values = [1, 2, 3, 4, 5]
        result = summarize_samples(values)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["median"], 3.0)
        self.assertEqual(result["p10"], 1.4)
        self.assertEqual(result["p90"], 4.6)
        self.assertAlmostEqual(result["cv"], 0.4714045207, places=9)
        self.assertEqual(percentile(values, 0.1), 1.4)


if __name__ == "__main__":
    unittest.main()
