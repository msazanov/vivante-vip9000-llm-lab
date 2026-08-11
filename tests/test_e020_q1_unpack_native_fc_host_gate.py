import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tooling" / "run_e020_q1_unpack_native_fc_host_gate.sh"


@unittest.skipUnless(os.environ.get("E020_RUN_Q1_FC_HOST_GATE") == "1",
                     "set E020_RUN_Q1_FC_HOST_GATE=1 to run Vivante Docker export")
class E020Q1UnpackNativeFCHostGateTest(unittest.TestCase):
    def test_packed_q1_feeds_native_fc_through_virtual_int8_weights(self):
        self.assertTrue(RUNNER.is_file(), f"missing production runner: {RUNNER}")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "e020-q1-fc-host-gate"
            result = subprocess.run(
                ["bash", str(RUNNER), "--output-dir", str(output_dir),
                 "--image", "ubuntu-npu:v2.0.10.2"],
                cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=180)
            self.assertEqual(result.returncode, 0,
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            summary = json.loads((output_dir / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual(summary["schema"],
                             "vip9000-e020-q1-unpack-native-fc-host/v1")
            self.assertEqual(summary["shape"], {"m": 16, "k": 32, "n": 1})
            self.assertEqual(summary["graph_inputs"], 2)
            self.assertEqual(summary["graph_outputs"], 1)
            self.assertEqual(summary["custom_kernel_count"], 1)
            self.assertEqual(summary["native_op"], "vxFullyConnectedLayer")
            self.assertEqual(summary["intermediate"], "virtual_int8_weights")
            self.assertEqual(summary["external_packed_weight_bytes"], 64)
            self.assertEqual(summary["external_expanded_weight_bytes"], 0)
            self.assertGreater(summary["nbg_bytes"], 0)
            self.assertTrue((output_dir / summary["nbg_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
