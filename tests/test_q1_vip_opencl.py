import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "experiments" / "E003-q1-packed-carrier" / "q1_vip_16x128_c0.cl"
EXPERIMENT_README = KERNEL.with_name("README.md")


class Q1VipOpenCLTests(unittest.TestCase):
    def _require_kernel(self) -> str:
        self.assertTrue(KERNEL.is_file(), f"missing kernel: {KERNEL}")
        return KERNEL.read_text(encoding="utf-8")

    def test_kernel_compiles_as_opencl_c_12_with_clang(self):
        clang = shutil.which("clang")
        if clang is None:
            self.skipTest("clang is not installed")
        self.assertTrue(KERNEL.is_file(), f"missing kernel: {KERNEL}")
        completed = subprocess.run(
            [
                clang,
                "-x",
                "cl",
                "-cl-std=CL1.2",
                "-Werror",
                "-fsyntax-only",
                str(KERNEL),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_kernel_has_exact_five_buffer_abi(self):
        source = self._require_kernel()
        match = re.search(
            r"__kernel\s+void\s+q1_vip_16x128_c0\s*\((.*?)\)\s*\{",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "kernel symbol or signature is missing")
        arguments = [re.sub(r"\s+", " ", item.strip()) for item in match.group(1).split(",")]
        self.assertEqual(
            arguments,
            [
                "__global const uchar *packed_signs",
                "__global const float *q1_scales",
                "__global const uchar *q8_values",
                "__global const float *q8_scales",
                "__global float *output",
            ],
        )

    def test_preprocessed_kernel_uses_no_forbidden_opencl_features(self):
        clang = shutil.which("clang")
        if clang is None:
            self.skipTest("clang is not installed")
        self.assertTrue(KERNEL.is_file(), f"missing kernel: {KERNEL}")
        completed = subprocess.run(
            [clang, "-E", "-x", "cl", "-cl-std=CL1.2", str(KERNEL)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        preprocessed = completed.stdout
        # Clang's implicit OpenCL header has optional-extension pragmas and
        # half/image declarations. Use its line marker for this source file to
        # isolate the user translation unit without relying on header spelling.
        markers = list(
            re.finditer(
                r'^#\s+\d+\s+"[^"\n]*q1_vip_16x128_c0\.cl"(?:\s+\d+)*\s*$',
                preprocessed,
                re.MULTILINE,
            )
        )
        self.assertTrue(markers, "clang -E did not emit a source line marker")
        preprocessed = preprocessed[markers[-1].end():]
        self.assertNotRegex(preprocessed, r"(?i)#\s*pragma\s+opencl\s+extension")
        for token in (
            r"\bhalf\b",
            r"\bcl_khr_fp16\b",
            r"\bsubgroups?\b",
            r"\bsub_group\w*\b",
            r"\bimage\w*\b",
            r"\bread_image\w*\b",
            r"\bwrite_image\w*\b",
            r"\bfast(?:_relaxed)?_math\b",
            r"\bcl_(?:amd|arm|intel|qcom|vivante)\w*\b",
        ):
            self.assertNotRegex(preprocessed, token, f"forbidden token matched: {token}")

    def test_kernel_carries_the_independent_signed_byte_and_bit_order_contract(self):
        source = self._require_kernel()
        normalized = re.sub(r"\s+", " ", source)
        self.assertIn("get_global_id(0)", normalized)
        self.assertIn("row >= 16", normalized)
        self.assertIn("block < 4", normalized)
        self.assertIn("j < 32", normalized)
        self.assertRegex(normalized, r"raw\s*<=\s*127")
        self.assertRegex(normalized, r"raw\s*-\s*256")
        self.assertRegex(normalized, r"k\s*&\s*7")
        self.assertIn("q1_scales[row]", normalized)
        self.assertIn("q8_scales[block]", normalized)
        self.assertIn("output[row]", normalized)

    def test_kernel_and_contract_leave_local_size_unfixed(self):
        source = self._require_kernel()
        normalized = re.sub(r"\s+", " ", source)
        self.assertNotRegex(normalized, r"\bget_local_(?:id|size)\b")
        self.assertNotRegex(normalized, r"reqd_work_group_size")
        document = EXPERIMENT_README.read_text(encoding="utf-8")
        self.assertIn("local size", document)
        self.assertIn("не фиксируется", document)

    def test_contract_states_exact_little_endian_plane_conversion(self):
        self.assertTrue(EXPERIMENT_README.is_file(), f"missing experiment README: {EXPERIMENT_README}")
        document = EXPERIMENT_README.read_text(encoding="utf-8")
        for phrase in (
            "q1_vip[0:256]",
            "packed_signs",
            "q1_vip[256:288]",
            "16 FP16",
            "FP32 q1_scales",
            "4 Q8-блоках по 34 байта",
            "первые 2 bytes little-endian FP16",
            "следующие 32 bytes",
            "q8_values",
            "FP32 q8_scales",
            "q1_canonical",
            "pack_tensor",
            "288 bytes",
            "136 bytes",
            "64 bytes",
        ):
            self.assertIn(phrase, document)

    def test_contract_marks_each_q8_scale_header_as_little_endian_fp16(self):
        self.assertTrue(EXPERIMENT_README.is_file(), f"missing experiment README: {EXPERIMENT_README}")
        document = EXPERIMENT_README.read_text(encoding="utf-8")
        self.assertIn("первые 2 bytes little-endian FP16", document)

    def test_experiment_readme_states_target_only_validation_contract(self):
        self.assertTrue(EXPERIMENT_README.is_file(), f"missing experiment README: {EXPERIMENT_README}")
        document = EXPERIMENT_README.read_text(encoding="utf-8")
        for phrase in (
            "Acuity",
            "TIM-VX",
            "VIPLite",
            "expected_f32.bin",
            "prepare",
            "first",
            "steady",
            "h2d",
            "run",
            "d2h",
            "1/10/100/1000",
            "50",
            "Что реально прошло на VIP9000",
            "100/100",
            "внутренний fallback",
        ):
            self.assertIn(phrase, document)


if __name__ == "__main__":
    unittest.main()
