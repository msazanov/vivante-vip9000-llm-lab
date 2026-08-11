import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_experiment_xy_chart.py"


class GenerateExperimentXYChartTests(unittest.TestCase):
    def test_renders_separate_token_operator_and_golden_xy_panels(self) -> None:
        model_rows = [
            {
                "run_id": "model-a55", "status": "unqualified",
                "configuration": {"partition": "A55 CPUs 0-5"},
                "workload": {"prompt_tokens": 512, "generated_tokens": 128},
                "performance": {"decode_tps": 0.72, "statistics": {
                    "decode_tps": {"median": 0.72, "p10": 0.71, "p90": 0.73}}},
                "quality": {"passed": None},
            },
            {
                "run_id": "model-failed", "status": "failed",
                "configuration": {"partition": "A76 CPUs 6-7"},
                "workload": {"prompt_tokens": 512, "generated_tokens": 128},
                "performance": {"decode_tps": None, "statistics": {
                    "decode_tps": {"median": None, "p10": None, "p90": None}}},
                "quality": {"passed": None},
            },
        ]
        capability = {
            "schema_version": "vip9000-capability-run/v1", "run_id": "resident-100",
            "configuration": {"measured_loops": 99},
            "host_run_us": {"median": 2846, "p10": 2830, "p90": 2889},
            "quality": {"golden_present": False, "correctness_verified": False},
        }
        phase = {
            "schema_version": "vip9000-viplite-phase-profile/v1", "run_id": "phase-1000",
            "stats_us": {"steady": {"end_to_end": {"median": 3025.668,
                                                        "p95": 3182.788}}},
            "correctness": {"golden_checked": False, "repeat_equal_failures": 0},
        }
        golden = {
            "schema_version": "q1-vip-cpu-golden-tests/v1", "run_id": "golden-001",
            "backend": "host-cpu-python", "tests_run": 3, "failures": 1, "errors": 0,
            "tests": [
                {"name": "scale-planes", "kind": "numeric", "passed": True},
                {"name": "lsb-order", "kind": "numeric", "passed": True},
                {"name": "invalid-length", "kind": "guard", "passed": False},
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "ledger.jsonl"
            results = root / "results"
            results.mkdir()
            ledger.write_text("".join(json.dumps(row) + "\n" for row in model_rows))
            for name, value in (("resident", capability), ("phase", phase)):
                directory = results / name
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps(value))
            nested = results / "nested"
            nested.mkdir()
            (nested / "summary.json").write_text(json.dumps({
                "schema_version": "vip9000-capability-run/v1", "run_id": "single-001",
                "status": "execution-compatible-unqualified",
                "network": {"host_run_us": 3212},
                "quality": {"correctness_verified": False},
            }))
            failed_npu = results / "failed"
            failed_npu.mkdir()
            (failed_npu / "summary.json").write_text(json.dumps({
                "schema_version": "vip9000-capability-run/v1", "run_id": "npu-failed",
                "status": "failed-before-launch",
            }))
            golden_path = root / "golden.json"
            golden_path.write_text(json.dumps(golden))
            first = root / "first.svg"
            second = root / "second.svg"

            for output in (first, second):
                completed = subprocess.run(
                    [sys.executable, str(SCRIPT), "--ledger", str(ledger),
                     "--results-dir", str(results), "--golden-summary", str(golden_path),
                     "--output", str(output)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            svg = ET.fromstring(first.read_bytes())
            namespace = {"s": "http://www.w3.org/2000/svg"}
            text = " ".join(node.text or "" for node in svg.findall(".//s:text", namespace))
            self.assertIn("Bonsai 27B", text)
            self.assertIn("мс/токен", text)
            self.assertIn("токен/с", text)
            self.assertIn("VIPLite operator", text)
            self.assertIn("мс/inference", text)
            self.assertIn("CPU golden", text)
            self.assertIn("FAIL", text)
            self.assertIn("model-failed", text)
            self.assertIn("npu-failed", text)
            self.assertEqual(len(svg.findall(".//*[@data-role='model-point']")), 1)
            self.assertEqual(len(svg.findall(".//*[@data-role='operator-point']")), 3)
            self.assertEqual(len(svg.findall(".//*[@data-role='golden-point']")), 3)

    def test_rejects_malformed_sources_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "ledger.jsonl"
            results = root / "results"
            golden = root / "golden.json"
            output = root / "chart.svg"
            results.mkdir()
            ledger.write_text("not-json\n")
            golden.write_text("{}")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--ledger", str(ledger),
                 "--results-dir", str(results), "--golden-summary", str(golden),
                 "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("malformed", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
