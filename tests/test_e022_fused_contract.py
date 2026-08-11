import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "experiments/E022-fused-q1/q1_fused_contract.h"


class E022FusedContractTest(unittest.TestCase):
    def test_c_header_matches_python_identity_for_extreme_values(self):
        source = r'''
        #include <stdint.h>
        #include "q1_fused_contract.h"
        int32_t run(const int8_t *q, const uint8_t *p, size_t k) {
            return q1_fused_dot_from_bits(q, p, k);
        }
        '''
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            c_file = tmp_path / "contract.c"
            so_file = tmp_path / "contract.so"
            c_file.write_text(source, encoding="utf-8")
            result = subprocess.run(
                ["cc", "-shared", "-fPIC", "-O2", "-I", str(ROOT / "experiments/E022-fused-q1"),
                 str(c_file), "-o", str(so_file)],
                text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            lib = ctypes.CDLL(str(so_file))
            lib.run.argtypes = [ctypes.POINTER(ctypes.c_int8),
                                ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            lib.run.restype = ctypes.c_int32
            q_values = [-128, -127, -1, 0, 1, 126, 127] * 4
            packed = bytes([0x00, 0xFF, 0x55, 0xAA] * 8)
            q = (ctypes.c_int8 * len(q_values))(*q_values)
            p = (ctypes.c_uint8 * len(packed))(*packed)
            expected = 0
            for index, value in enumerate(q_values):
                bit = (packed[index // 8] >> (index % 8)) & 1
                expected += value if bit else -value
            self.assertEqual(lib.run(q, p, len(q_values)), expected)


if __name__ == "__main__":
    unittest.main()
