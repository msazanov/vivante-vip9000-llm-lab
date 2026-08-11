import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E019-q1-i16-evis-gemm"


class Q1I16EvisGemmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls._tmp.name) / "e019-contract"
        cls.compile_result = subprocess.run(
            ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
             str(EXPERIMENT / "q1_i16_evis_gemm_contract.c"),
             str(EXPERIMENT / "q1_i16_evis_gemm_contract_cli.c"), "-o", str(cls.binary)],
            text=True, capture_output=True, check=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_contract(self, m, k, n=1):
        self.assertEqual(self.compile_result.returncode, 0, self.compile_result.stderr)
        return subprocess.run([str(self.binary), str(m), str(k), str(n)],
                              text=True, capture_output=True, check=False)

    def test_m1_k32_exact_dfp8_contract(self):
        result = self.run_contract(1, 32)
        self.assertEqual(result.returncode, 0, result.stderr)
        c = json.loads(result.stdout)
        self.assertEqual(c["kernel"], "com.vivantecorp.extension.evis.gemm_I16I16toI16")
        self.assertEqual(c["a_dims"], [32, 1])
        self.assertEqual(c["b_dims"], [1, 32])
        self.assertEqual(c["c_dims"], [1, 1])
        self.assertEqual(c["dfp_fixed_point_pos"], 8)
        self.assertEqual(c["parameter_count"], 10)
        self.assertEqual(c["scalar_count"], 7)
        self.assertEqual(c["uniform_count"], 9)
        self.assertEqual(c["global_scale"], [4, 4, 1])
        self.assertEqual(c["global_size"], [4, 4, 1])
        self.assertEqual(c["physical_max_abs_dot"], 4096)
        self.assertIs(c["int16_safe"], True)

    def test_m1024_k128_safe_and_geometry(self):
        result = self.run_contract(1024, 128)
        self.assertEqual(result.returncode, 0, result.stderr)
        c = json.loads(result.stdout)
        self.assertEqual(c["global_size"], [4, 256, 1])
        self.assertEqual(c["physical_max_abs_dot"], 16384)
        self.assertIs(c["int16_safe"], True)

    def test_rejects_k256(self):
        result = self.run_contract(1, 256)
        self.assertEqual(result.returncode, 2)
        self.assertIn("K must be 32 or 128", result.stderr)

    def test_rejects_prefill_n(self):
        result = self.run_contract(1, 32, 2)
        self.assertEqual(result.returncode, 2)
        self.assertIn("N must be 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
