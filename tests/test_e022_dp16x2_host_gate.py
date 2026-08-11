import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tooling" / "run_e022_dp16x2_host_gate.sh"


@unittest.skipUnless(os.environ.get("E022_RUN_DP16X2_HOST_GATE") == "1",
                     "set E022_RUN_DP16X2_HOST_GATE=1 to run Vivante Docker export")
class E022DP16x2HostGateTest(unittest.TestCase):
    def test_raw_probe_exports_one_custom_kernel(self):
        self.assertTrue(RUNNER.is_file(), f"missing production runner: {RUNNER}")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dp16x2"
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
                             "vip9000-e022-dp16x2-probe-host/v1")
            self.assertEqual(summary["custom_kernel_count"], 1)
            self.assertEqual(summary["graph_inputs"], 3)
            self.assertEqual(summary["graph_outputs"], 1)
            self.assertEqual(summary["output_values"],
                             ["pair_00_x", "pair_00_y", "pair_01_x", "pair_01_y",
                              "pair_11_x", "pair_11_y", "pair_15_x", "pair_15_y"])
            self.assertGreater(summary["nbg_bytes"], 0)
            self.assertTrue((output_dir / summary["nbg_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
