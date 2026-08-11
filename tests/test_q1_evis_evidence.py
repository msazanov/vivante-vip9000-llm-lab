import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks" / "results" / "q1-vip9000-evis-bonsai-20260811"
SUMMARY = RESULT / "summary.json"
SUMMARIZER = ROOT / "tooling" / "summarize_viplite_profile.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Q1EvisEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_synthetic_fixture_and_both_outputs_are_independently_auditable(self):
        micro = self.summary["microkernel_same_shape"]
        fixture = micro["fixture"]
        directory = RESULT / fixture["directory"]
        expected = directory / "expected_f32.bin"
        scalar_output = directory / "scalar-output.f32.bin"
        evis_output = directory / "evis-output.f32.bin"

        self.assertEqual((directory / "q1_canonical.bin").stat().st_size, 16 * 18)
        self.assertEqual((directory / "q8.bin").stat().st_size, 136)
        self.assertEqual((directory / "activation.f32.bin").stat().st_size, 128 * 4)
        self.assertEqual(expected.stat().st_size, 16 * 4)
        self.assertEqual(sha256(directory / "q1_canonical.bin"), fixture["q1_canonical_sha256"])
        self.assertEqual(sha256(directory / "q8.bin"), fixture["q8_sha256"])
        self.assertEqual(sha256(directory / "activation.f32.bin"), fixture["activation_f32_sha256"])
        self.assertEqual(sha256(expected), fixture["golden_f32_sha256"])
        self.assertEqual(scalar_output.read_bytes(), expected.read_bytes())
        self.assertEqual(evis_output.read_bytes(), expected.read_bytes())

        for record_name in ("scalar_npu", "evis_packed_q1_q8"):
            record = micro[record_name]
            self.assertTrue(record["golden_passed"])
            self.assertEqual(record["output_sha256"], sha256(expected))
            self.assertEqual(record["golden_sha256"], sha256(expected))
            raw_log = RESULT / record["raw_log"]
            self.assertEqual(sha256(raw_log), record["raw_log_sha256"])

    def test_retained_logs_reproduce_medians_speedups_and_projection(self):
        micro = self.summary["microkernel_same_shape"]
        parsed = {}
        for record_name in ("scalar_npu", "evis_packed_q1_q8"):
            record = micro[record_name]
            completed = subprocess.run(
                [sys.executable, str(SUMMARIZER), str(RESULT / record["raw_log"]),
                 "--run-id", f"check-{record_name}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            parsed[record_name] = json.loads(completed.stdout)["stats_us"]["steady"]
            for source, target in (
                ("cycles", "cycles_median"),
                ("h2d", "h2d_us_median"),
                ("device", "device_us_median"),
                ("run", "run_us_median"),
                ("d2h", "d2h_us_median"),
                ("end_to_end", "e2e_us_median"),
            ):
                self.assertEqual(parsed[record_name][source]["median"], record[target])

        scalar = micro["scalar_npu"]
        evis = micro["evis_packed_q1_q8"]
        for result_key, scalar_key, evis_key in (
            ("cycles", "cycles_median", "cycles_median"),
            ("device", "device_us_median", "device_us_median"),
            ("run", "run_us_median", "run_us_median"),
            ("end_to_end", "e2e_us_median", "e2e_us_median"),
        ):
            self.assertTrue(math.isclose(
                micro["speedup"][result_key], scalar[scalar_key] / evis[evis_key],
                rel_tol=0.0, abs_tol=1e-12,
            ))

        scaling = self.summary["bonsai_ffn_gate_scaling"]
        largest = max(scaling["npu_measurements"], key=lambda item: item["rows"])
        tiles = math.ceil(scaling["full_rows"] / largest["rows"])
        self.assertFalse(scaling["projection_is_measurement"])
        self.assertTrue(math.isclose(
            scaling["npu_full_layer_linear_projection_ms"],
            tiles * largest["e2e_ms_median"], rel_tol=0.0, abs_tol=1e-12,
        ))
        self.assertTrue(math.isclose(
            scaling["projected_npu_to_measured_cpu_slowdown"],
            scaling["npu_full_layer_linear_projection_ms"] / scaling["cpu_full_layer_ms_median"],
            rel_tol=0.0, abs_tol=1e-12,
        ))

        forbidden = {".nb", ".vxgcsl", ".gcpgm", ".gguf"}
        self.assertFalse(any(path.suffix.lower() in forbidden for path in RESULT.rglob("*")))


if __name__ == "__main__":
    unittest.main()
