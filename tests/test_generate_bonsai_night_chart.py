import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_bonsai_night_chart.py"
MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
RUNTIME_COMMIT = "38c66ad0241da4f9fcce541cda8edc219086cec5"
GOLDEN_SHA256 = "a9d86b7d298be35cf27a156be87f9aeedbae3602b877e74ced0fdc75b41c4a5f"


def summary_fixture():
    return {
        "schema_version": 1,
        "series": "Bonsai-27B Q1_0 / A733 heterogeneous night",
        "model_sha256": MODEL_SHA256,
        "runtime_commit": RUNTIME_COMMIT,
        "quality_evidence": {
            "method": "deterministic greedy exact stdout comparison",
            "reference_run_id": "golden-reference",
            "candidate_run_id": "golden-candidate",
            "reference_sha256": GOLDEN_SHA256,
            "candidate_sha256": GOLDEN_SHA256,
            "exact_match": True,
        },
        "rows": [
            {
                "run_id": "cpu-reference",
                "label": "CPU A55×6 fresh reference",
                "backend": "cpu",
                "metric_scope": "full_model_decode",
                "unit": "tokens_per_second",
                "samples": [0.70, 0.72, 0.74],
                "median": 0.72,
                "quality": {"exact_match": True},
                "reference": True,
            },
            {
                "run_id": "cpu-candidate",
                "label": "CPU A55×6 poll 25",
                "backend": "cpu",
                "metric_scope": "full_model_decode",
                "unit": "tokens_per_second",
                "samples": [0.78, 0.80, 0.82],
                "median": 0.80,
                "quality": {"exact_match": True},
            },
            {
                "run_id": "vulkan-candidate",
                "label": "PowerVR Vulkan ngl99",
                "backend": "vulkan",
                "metric_scope": "full_model_decode",
                "unit": "tokens_per_second",
                "samples": [0.66, 0.68, 0.70],
                "median": 0.68,
                "quality": {"exact_match": True},
            },
            {
                "run_id": "npu-operator",
                "label": "VIP9000 Q1 GEMV",
                "backend": "npu",
                "metric_scope": "operator",
                "unit": "milliseconds",
                "samples": [1.0, 1.2, 1.4],
                "median": 1.2,
                "quality": {"exact_match": True},
                "rejection_reason": "operator timing, not full model decode",
            },
            {
                "run_id": "npu-fallback",
                "label": "VIP9000 NBG fallback",
                "backend": "npu",
                "metric_scope": "full_model_decode",
                "unit": "milliseconds",
                "samples": [2.0, 2.1, 2.2],
                "median": 2.1,
                "quality": {"exact_match": False},
                "rejection_reason": "NPU timing/fallback is not a model tok/s result",
            },
        ],
    }


class BonsaiNightChartTests(unittest.TestCase):
    def run_generator(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "summary.json"
            output = directory / "chart.svg"
            source.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            return completed, output.read_bytes() if output.exists() else None

    def test_full_decode_chart_has_reference_error_bars_percentages_and_rejections(self):
        completed, output = self.run_generator(summary_fixture())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIsNotNone(output)
        svg = output.decode("utf-8")
        self.assertIn('data-panel="tokens_per_second"', svg)
        self.assertIn('data-run-id="cpu-candidate"', svg)
        self.assertIn('data-backend="cpu"', svg)
        self.assertIn("cpu · CPU A55×6 poll 25", svg)
        self.assertIn("0,800 ток/с", svg)
        self.assertIn("±", svg)
        self.assertIn("свежий эталон", svg)
        self.assertIn("0,720 ток/с", svg)
        self.assertIn("+11,1%", svg)
        self.assertIn("Отклонены от сравнения", svg)
        self.assertIn("VIP9000 Q1 GEMV", svg)
        self.assertIn("operator timing", svg)
        self.assertNotIn('data-panel="milliseconds"', svg)
        self.assertNotIn('data-metric="npu-operator"', svg)

    def test_malformed_schema_fails_closed_before_writing_output(self):
        payload = summary_fixture()
        payload["schema_version"] = 2
        completed, output = self.run_generator(payload)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("schema", completed.stderr.lower())
        self.assertIsNone(output)

    def test_rejects_wrong_model_runtime_or_golden_provenance(self):
        mutations = {
            "model": lambda payload: payload.__setitem__("model_sha256", "0" * 64),
            "runtime": lambda payload: payload.__setitem__("runtime_commit", "1" * 40),
            "missing golden": lambda payload: payload.pop("quality_evidence"),
            "mismatched golden": lambda payload: payload["quality_evidence"].__setitem__(
                "candidate_sha256", "2" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = summary_fixture()
                mutate(payload)

                completed, output = self.run_generator(payload)

                self.assertEqual(completed.returncode, 2)
                self.assertRegex(completed.stderr.lower(), r"model|runtime|quality|golden")
                self.assertIsNone(output)

    def test_non_exact_full_model_row_is_visible_only_in_unqualified_screen_panel(self):
        payload = summary_fixture()
        payload["rows"][1]["quality"]["exact_match"] = False
        completed, output = self.run_generator(payload)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIsNotNone(output)
        svg = output.decode("utf-8")
        self.assertNotIn('data-panel="tokens_per_second" data-run-id="cpu-candidate"', svg)
        self.assertIn('data-panel="screen_tokens_per_second" data-run-id="cpu-candidate"', svg)
        self.assertIn('data-status="unqualified"', svg)
        self.assertIn("Исследовательские screens", svg)
        self.assertIn("0,800 ток/с", svg)
        self.assertIn("1 250,0 мс/токен", svg)
        self.assertIn("CPU A55×6 poll 25", svg)
        self.assertIn("golden", svg.lower())

    def test_identical_input_is_byte_for_byte_deterministic(self):
        first, first_output = self.run_generator(summary_fixture())
        second, second_output = self.run_generator(summary_fixture())

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            first_output, second_output,
        )

    def test_rejects_stale_median_and_unknown_explicit_reference(self):
        payload = summary_fixture()
        payload["rows"][1]["median"] = 9.9
        completed, output = self.run_generator(payload)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("median", completed.stderr.lower())
        self.assertIsNone(output)

        payload = summary_fixture()
        payload["reference_run_id"] = "does-not-exist"
        completed, output = self.run_generator(payload)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("reference_run_id", completed.stderr)
        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
