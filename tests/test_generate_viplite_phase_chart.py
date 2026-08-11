import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_viplite_phase_chart.py"


class GenerateVIPLitePhaseChartTests(unittest.TestCase):
    def test_generates_deterministic_russian_svg_with_semantic_layers(self) -> None:
        summary = {
            "schema_version": "vip9000-viplite-phase-profile/v1",
            "run_id": "profile-001",
            "iterations": {"first": 1, "steady": 999, "total": 1000},
            "stats_us": {
                "first": {
                    "h2d": {"median": 120.0}, "device": {"median": 2800.0},
                    "run_minus_device": {"median": 220.0}, "d2h": {"median": 6.0},
                    "end_to_end": {"median": 3146.0},
                },
                "steady": {
                    "h2d": {"median": 66.0, "p95": 120.0},
                    "device": {"median": 2815.0, "p95": 2907.0},
                    "run_minus_device": {"median": 100.0, "p95": 211.0},
                    "d2h": {"median": 8.0, "p95": 15.0},
                    "end_to_end": {"median": 2989.0, "p95": 3135.0},
                },
            },
            "correctness": {
                "golden_checked": False, "repeat_equal_failures": 0,
                "repeat_equal_is_not_golden": True,
            },
            "thermal": {"npu_start_c": 35.278, "npu_peak_c": 42.16, "npu_end_c": 41.23,
                        "npu_clock_min_hz": 1008000000, "npu_clock_max_hz": 1008000000,
                        "throttling_evidence": False},
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "summary.json"
            first = Path(temporary) / "first.svg"
            second = Path(temporary) / "second.svg"
            source.write_text(json.dumps(summary), encoding="utf-8")

            for output in (first, second):
                completed = subprocess.run(
                    [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            root = ET.fromstring(first.read_bytes())
            text = " ".join(element.text or "" for element in root.iter() if element.tag.endswith("text"))
            self.assertIn("Профиль VIPLite", text)
            self.assertIn("H2D", text)
            self.assertIn("D2H", text)
            self.assertIn("golden", text)
            self.assertIn("42,16", text)
            self.assertGreaterEqual(len(root.findall(".//{http://www.w3.org/2000/svg}rect")), 8)

    def test_rejects_wrong_schema_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "summary.json"
            output = Path(temporary) / "chart.svg"
            source.write_text("{}", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("schema", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
