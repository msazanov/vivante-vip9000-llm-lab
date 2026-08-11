import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("E022_RUN_FUSED_HOST_GATES") == "1",
                     "set E022_RUN_FUSED_HOST_GATES=1 to run Vivante Docker exports")
class E022FusedHostGatesTest(unittest.TestCase):
    def _run(self, script: str, rows: int, schema: str, k: int) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / script
            result = subprocess.run(
                ["bash", str(ROOT / "tooling" / script),
                 "--output-dir", str(output_dir), "--rows", str(rows),
                 "--image", "ubuntu-npu:v2.0.10.2"],
                cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=180)
            self.assertEqual(result.returncode, 0,
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            summary = json.loads((output_dir / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual(summary["schema"], schema)
            self.assertEqual(summary["custom_kernel_count"], 1)
            self.assertEqual(summary["graph_inputs"], 2)
            self.assertEqual(summary["graph_outputs"], 1)
            self.assertEqual(summary["rows"], rows)
            self.assertEqual(summary["k"], k)
            self.assertTrue((output_dir / summary["nbg_file"]).is_file())

    def test_rows4_k128_host_export(self):
        self._run("run_e022_fused_rows4_host_gate.sh", 4,
                  "vip9000-e022-fused-q1-rows4-host/v1", 128)

    def test_rows4_k5120_host_export(self):
        self._run("run_e022_fused_rows4_k5120_host_gate.sh", 4,
                  "vip9000-e022-fused-q1-rows4-k5120-host/v1", 5120)

    def test_rows4_k5120_scales_host_export(self):
        self._run("run_e022_fused_rows4_k5120_scales_host_gate.sh", 4,
                  "vip9000-e022-fused-q1-rows4-k5120-scales-host/v1", 5120)


if __name__ == "__main__":
    unittest.main()
