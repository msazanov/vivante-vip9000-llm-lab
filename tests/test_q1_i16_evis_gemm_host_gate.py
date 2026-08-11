import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tooling" / "run_q1_i16_evis_gemm_host_gate.sh"


@unittest.skipUnless(os.environ.get("E019_RUN_HOST_GATE") == "1",
                     "set E019_RUN_HOST_GATE=1 to run Vivante Docker export")
class Q1I16EvisGemmHostGateTest(unittest.TestCase):
    def test_m1_k32_exports_standalone_custom_evis_nbg(self):
        self.assertTrue(RUNNER.is_file(), f"missing production runner: {RUNNER}")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "e019-host-gate"
            result = subprocess.run(
                [str(RUNNER), "--output-dir", str(output_dir), "--image",
                 "ubuntu-npu:v2.0.10.2", "--m", "1", "--k", "32", "--n", "1"],
                cwd=ROOT, text=True, capture_output=True, check=False, timeout=180)
            self.assertEqual(result.returncode, 0,
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.is_file(), result.stdout + result.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "vip9000-q1-i16-evis-gemm-host/v1")
            self.assertEqual(summary["gate"], {"m": 1, "k": 32, "n": 1})
            self.assertEqual(summary["kernel"],
                             "com.vivantecorp.extension.evis.gemm_I16I16toI16")
            self.assertEqual(summary["graph_inputs"], 2)
            self.assertEqual(summary["graph_outputs"], 1)
            self.assertEqual(summary["host_export_exit_code"], 0)
            self.assertGreater(summary["shader_binary_bytes"], 0)
            self.assertGreater(summary["nbg_bytes"], 0)
            self.assertEqual(summary["forbidden_markers_found"], [])
            self.assertTrue((output_dir / summary["nbg_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
