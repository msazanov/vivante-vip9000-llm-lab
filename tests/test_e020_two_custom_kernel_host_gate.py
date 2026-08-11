import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tooling" / "run_e020_two_custom_kernel_host_gate.sh"


@unittest.skipUnless(os.environ.get("E020_RUN_HOST_GATE") == "1",
                     "set E020_RUN_HOST_GATE=1 to run Vivante Docker export")
class E020TwoCustomKernelHostGateTest(unittest.TestCase):
    def test_two_custom_nodes_export_one_nbg_with_virtual_intermediate(self):
        self.assertTrue(RUNNER.is_file(), f"missing production runner: {RUNNER}")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "e020-host-gate"
            result = subprocess.run(
                ["bash", str(RUNNER), "--output-dir", str(output_dir),
                 "--image", "ubuntu-npu:v2.0.10.2", "--elements", "16"],
                cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=180)
            self.assertEqual(result.returncode, 0,
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            summary = json.loads((output_dir / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual(summary["schema"],
                             "vip9000-e020-two-custom-kernel-host/v1")
            self.assertEqual(summary["custom_kernel_count"], 2)
            self.assertEqual(summary["graph_node_count"], 2)
            self.assertEqual(summary["graph_inputs"], 1)
            self.assertEqual(summary["graph_outputs"], 1)
            self.assertEqual(summary["intermediate"], "virtual_uint8_tensor")
            self.assertEqual(summary["elements"], 16)
            self.assertGreater(summary["nbg_bytes"], 0)
            self.assertTrue((output_dir / summary["nbg_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
