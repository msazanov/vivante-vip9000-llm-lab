import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tooling" / "generate_q1_i16_evis_gemm_fixture.py"


def unpack_i16(path: Path) -> list[int]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw) // 2}h", raw))


class GenerateQ1I16EvisGemmFixtureTest(unittest.TestCase):
    def test_adversarial_m1_k32_has_hand_checked_exact_dots(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture"
            result = subprocess.run(
                ["python3", str(GENERATOR), "--output-dir", str(output),
                 "--m", "1", "--k", "32", "--n", "1"],
                text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "vip9000-q1-i16-evis-gemm-fixture/v1")
            self.assertEqual(manifest["shape"], {"m": 1, "k": 32, "n": 1})
            self.assertEqual(manifest["dfp_fixed_point_pos"], 8)
            self.assertEqual(
                [case["name"] for case in manifest["cases"]],
                ["all-negative", "all-positive", "alternating-plus", "alternating-minus"],
            )

            activation = [-128, -127, -1, 0, 1, 126, 127, 64] * 4
            expected = {
                "all-negative": -248,
                "all-positive": 248,
                "alternating-plus": -256,
                "alternating-minus": 256,
            }
            for case in manifest["cases"]:
                case_dir = output / case["name"]
                self.assertEqual(unpack_i16(case_dir / "input_b.i16.bin"), activation)
                self.assertEqual(unpack_i16(case_dir / "expected_c.i16.bin"),
                                 [expected[case["name"]]])
                weights = unpack_i16(case_dir / "input_a.i16.bin")
                self.assertEqual(len(weights), 32)
                self.assertTrue(all(value in (-256, 256) for value in weights))
                self.assertEqual(sum((weight // 256) * value
                                     for weight, value in zip(weights, activation)),
                                 expected[case["name"]])
                self.assertEqual(case["expected_physical_i16"], [expected[case["name"]]])
                self.assertEqual(set(case["sha256"]), {"input_a", "input_b", "expected_c"})

    def test_refuses_to_overwrite_existing_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture"
            first = subprocess.run(
                ["python3", str(GENERATOR), "--output-dir", str(output)],
                text=True, capture_output=True, check=False)
            second = subprocess.run(
                ["python3", str(GENERATOR), "--output-dir", str(output)],
                text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 2)
            self.assertIn("already exists", second.stderr)


if __name__ == "__main__":
    unittest.main()
