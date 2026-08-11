import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_q1_evis_chart.py"


def fixture():
    return {
        "schema_version": "q1-vip9000-evis-study/v1",
        "run_id": "q1-test",
        "microkernel_same_shape": {
            "shape": {"rows": 16, "columns": 128},
            "scalar_npu": {
                "label": "Scalar NPU",
                "cycles_median": 133472,
                "device_us_median": 137,
                "run_us_median": 176.459,
                "e2e_us_median": 178.668,
                "golden_passed": True,
            },
            "evis_packed_q1_q8": {
                "label": "EVIS packed Q1×Q8",
                "cycles_median": 13331,
                "device_us_median": 18,
                "run_us_median": 67.834,
                "e2e_us_median": 73.209,
                "golden_passed": True,
            },
        },
        "bonsai_ffn_gate_scaling": {
            "columns": 5120,
            "full_rows": 17408,
            "cpu_full_layer_ms_median": 3.85,
            "npu_measurements": [
                {"rows": 16, "e2e_ms_median": 0.598791, "device_ms_median": 0.535,
                 "golden_passed": True},
                {"rows": 1024, "e2e_ms_median": 24.137459, "device_ms_median": 23.585,
                 "golden_passed": True},
            ],
            "npu_full_layer_linear_projection_ms": 410.336803,
            "projection_is_measurement": False,
        },
    }


class GenerateQ1EvisChartTests(unittest.TestCase):
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

    def test_chart_separates_same_shape_speedup_from_real_scaling(self):
        completed, svg = self.run_chart(fixture())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Одинаковая задача: 16×128", svg)
        self.assertIn("10,01×", svg)
        self.assertIn("Golden: PASS", svg)
        self.assertIn("Реальный Bonsai blk.0.ffn_gate", svg)
        self.assertIn("16 строк · измерено", svg)
        self.assertIn("1024 строк · измерено", svg)
        self.assertIn("линейная оценка, не измерение", svg)
        self.assertIn("CPU · 17408 строк · измерено", svg)

    def test_chart_rejects_projection_marked_as_measurement(self):
        payload = fixture()
        payload["bonsai_ffn_gate_scaling"]["projection_is_measurement"] = True
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
