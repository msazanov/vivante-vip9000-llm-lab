import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "benchmarks"
    / "results"
    / "vip9000-q1-q8-row-reuse-20260811-001"
    / "summary.json"
)
CHART = ROOT / "benchmarks" / "charts" / "vip9000-q1-q8-row-reuse-20260811-001.svg"
SCRIPT = ROOT / "tooling" / "generate_q1_reuse_chart.py"

SPEC = importlib.util.spec_from_file_location("generate_q1_reuse_chart_evidence", SCRIPT)
chart = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = chart
SPEC.loader.exec_module(chart)


class Q1ReuseEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_recorded_speedups_and_projection_recompute_from_measurements(self):
        tile = self.payload["bonsai_real_tile_1024x5120"]
        old = tile["baseline_e003"]
        e011 = tile["e011_r4_shared_q8"]
        e014 = tile["e014_r4_vector_accum"]

        metric_keys = {
            "device_cycles": "device_cycles_median",
            "device_time": "device_ms_median",
            "run": "run_ms_median",
            "end_to_end": "end_to_end_ms_median",
        }
        for speedup_name, numerator, denominator in (
            ("speedup_e003_over_e011", old, e011),
            ("speedup_e003_over_e014", old, e014),
            ("speedup_e011_over_e014", e011, e014),
        ):
            recorded = tile[speedup_name]
            for label, key in metric_keys.items():
                if label in recorded:
                    self.assertTrue(
                        math.isclose(
                            recorded[label], numerator[key] / denominator[key],
                            rel_tol=0.0, abs_tol=1e-12,
                        ),
                        f"stale {speedup_name}.{label}",
                    )

        full = self.payload["full_layer_context"]
        projection = 17 * e014["end_to_end_ms_median"]
        self.assertTrue(math.isclose(full["npu_e014_linear_projection_ms"], projection))
        self.assertTrue(
            math.isclose(
                full["projected_npu_to_measured_cpu_slowdown"],
                projection / full["measured_cpu_full_layer_ms_median"],
            )
        )
        self.assertIs(full["projection_is_measurement"], False)
        self.assertIs(full["tokens_per_second_improvement_demonstrated"], False)

    def test_all_variants_pass_golden_and_new_best_output_is_byte_exact(self):
        variants = self.payload["synthetic_r64_k128"]["variants"]
        self.assertTrue(all(item["golden_passed"] is True for item in variants))
        accepted = {item["id"]: item for item in variants if item["status"].startswith("accepted")}
        self.assertEqual(set(accepted), {"e011-r4-shared-q8", "e014-r4-vector-accum"})

        tile = self.payload["bonsai_real_tile_1024x5120"]
        self.assertEqual(
            tile["e014_r4_vector_accum"]["output_sha256"],
            tile["e011_r4_shared_q8"]["output_sha256"],
        )
        self.assertEqual(tile["e014_r4_vector_accum"]["max_abs_error_vs_e011"], 0.0)

    def test_committed_chart_is_deterministically_generated_from_summary(self):
        expected = chart.render(self.payload)
        self.assertEqual(CHART.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
