import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_q1_reuse_chart.py"


def fixture():
    return {
        "schema_version": "vip9000-q1-q8-row-reuse/v1",
        "synthetic_r64_k128": {
            "variants": [
                {
                    "id": "e003-row-local",
                    "status": "baseline",
                    "device_cycles_median": 28514,
                    "golden_passed": True,
                },
                {
                    "id": "e011-r4-shared-q8",
                    "status": "accepted-local-kernel-win",
                    "device_cycles_median": 25586,
                    "golden_passed": True,
                },
                {
                    "id": "e012-r4-direct-signed-read",
                    "status": "rejected-no-material-win",
                    "device_cycles_median": 25705,
                    "golden_passed": True,
                },
                {
                    "id": "e013-r8-shared-q8",
                    "status": "rejected-r8-slower",
                    "device_cycles_range": [42310, 43057],
                    "golden_passed": True,
                },
                {
                    "id": "e014-r4-vector-accum",
                    "status": "accepted-new-best-evis",
                    "device_cycles_median": 24570,
                    "golden_passed": True,
                },
            ]
        },
        "bonsai_real_tile_1024x5120": {
            "baseline_e003": {"end_to_end_ms_median": 24.137459, "golden_passed": True},
            "e011_r4_shared_q8": {"end_to_end_ms_median": 14.150126, "golden_passed": True},
            "e014_r4_vector_accum": {"end_to_end_ms_median": 13.9695, "golden_passed": True},
        },
        "full_layer_context": {
            "rows": 17408,
            "measured_cpu_full_layer_ms_median": 3.8503125,
            "npu_e011_linear_projection_ms": 240.552142,
            "npu_e014_linear_projection_ms": 237.4815,
            "projection_is_measurement": False,
            "tokens_per_second_improvement_demonstrated": False,
        },
    }


class GenerateQ1ReuseChartTests(unittest.TestCase):
    def run_chart(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.json"
            output = Path(directory) / "chart.svg"
            source.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            svg = output.read_text(encoding="utf-8") if output.exists() else ""
            return completed, svg

    def test_chart_separates_kernel_tile_and_full_layer_claims(self):
        completed, svg = self.run_chart(fixture())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Микротест 64×128", svg)
        self.assertIn("25 586", svg)
        self.assertIn("Реальный tile Bonsai 1024×5120", svg)
        self.assertIn("E014 · vector FP32", svg)
        self.assertIn("1,73× быстрее", svg)
        self.assertIn("Полный слой 17408×5120", svg)
        self.assertIn("измерено", svg)
        self.assertIn("прогноз, не измерение", svg)
        self.assertIn("Ускорение tokens/s пока не доказано", svg)

    def test_chart_rejects_projection_marked_as_measurement(self):
        payload = fixture()
        payload["full_layer_context"]["projection_is_measurement"] = True
        completed, svg = self.run_chart(payload)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(svg)
        self.assertIn("projection_is_measurement must be false", completed.stderr)

    def test_chart_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.json"
            output = Path(directory) / "chart.svg"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            output.write_text("keep", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
