"""Fresh cross/QEMU and optimized-disassembly gates for E055."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tooling.check_e055_disassembly import inspect


ROOT = pathlib.Path(__file__).resolve().parents[1]
DISASSEMBLY_REVIEW = (
    ROOT / "experiments/E055-q1-hot-cold/data/disassembly-review.json"
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(all(shutil.which(x) for x in (
    "aarch64-linux-gnu-g++", "aarch64-linux-gnu-objdump", "qemu-aarch64")),
    "AArch64 cross/QEMU tools are required")
class E055AArch64Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="e055-gate-")
        cls.binary = pathlib.Path(cls.temp.name) / "e055"
        cls.cpp_object = pathlib.Path(cls.temp.name) / "e055_cpp.o"
        cls.asm_object = pathlib.Path(cls.temp.name) / "e055_asm.o"
        subprocess.run([
            "aarch64-linux-gnu-g++", "-std=c++17", "-O3", "-Wall", "-Wextra",
            "-Werror", "-march=armv8.2-a+dotprod",
            "-c", "tooling/e055_q1_hotcold.cpp", "-o", str(cls.cpp_object),
        ], cwd=ROOT, check=True)
        subprocess.run([
            "aarch64-linux-gnu-g++", "-march=armv8.2-a+dotprod", "-c",
            "experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S",
            "-o", str(cls.asm_object),
        ], cwd=ROOT, check=True)
        subprocess.run([
            "aarch64-linux-gnu-g++", "-O3", str(cls.cpp_object),
            str(cls.asm_object), "-o", str(cls.binary),
        ], cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def run_harness(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu",
                               str(self.binary), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_exact_stock_golden(self) -> None:
        result = self.run_harness("--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data["golden_pass"])
        self.assertEqual(data["golden_cases"], 18)

    def test_cold_defaults_to_exactly_one_call(self) -> None:
        result = self.run_harness("--mode", "packed_stream", "--cache-state",
                                  "cold_conditioned", "--working-set-bytes", "65536")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["schema"], "e055-q1-hot-cold-harness/v1")
        self.assertEqual((data["iterations"], data["calls"]), (1, 1))
        conditioning = data["cold_conditioning"]
        self.assertEqual(conditioning["requested_bytes"], 67_108_864)
        self.assertEqual(conditioning["actual_bytes"], 67_108_864)
        self.assertEqual(conditioning["lines_touched"], 1_048_576)
        self.assertEqual(conditioning["checksum"], "0x8d3ea13d15850279")

    def test_hot_default_reports_verified_warmup_conditioning(self) -> None:
        result = self.run_harness("--mode", "packed_stream", "--cache-state",
                                  "hot_repeat", "--working-set-bytes", "65536",
                                  "--iterations", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        conditioning = json.loads(result.stdout)["cold_conditioning"]
        self.assertEqual(conditioning["strategy"], "verified_kernel_warmup")
        self.assertTrue(conditioning["verified_touched"])
        self.assertEqual(conditioning["requested_bytes"], 65728)
        self.assertEqual(conditioning["actual_bytes"], 65728)
        self.assertEqual(conditioning["lines_touched"], (65728 + 63) // 64)
        self.assertNotEqual(conditioning["checksum"], "0x0")
        self.assertEqual(conditioning["warmup_calls"], 16)

    def test_cold_rejects_iteration_and_budget_overrides(self) -> None:
        for args in (("--iterations", "2"), ("--budget-ms", "10")):
            with self.subTest(args=args):
                result = self.run_harness("--cache-state", "cold_conditioned", *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --iterations 1", result.stderr)

    def test_thrash_size_must_be_cache_line_aligned(self) -> None:
        result = self.run_harness("--cache-state", "cold_conditioned",
                                  "--thrash-bytes", "65")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("64-byte aligned", result.stderr)

    def test_working_set_rounding_overflow_fails_closed(self) -> None:
        result = self.run_harness("--working-set-bytes", str((1 << 64) - 207),
                                  "--iterations", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rounding overflow", result.stderr)

    def test_sync_without_fds_fails_closed(self) -> None:
        result = self.run_harness("--iterations", "1", "--sync")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fd9/fd8 are unavailable", result.stderr)

    def test_optimized_disassembly_shape(self) -> None:
        report = inspect(str(self.binary), "aarch64-linux-gnu-objdump")
        self.assertEqual(report["status"], "PASS", report["failures"])
        published = json.loads(DISASSEMBLY_REVIEW.read_text(encoding="utf-8"))
        expected = next(item for item in published["builds"] if item["name"] == "O3")
        self.assertEqual(sha256(self.binary), expected["binary_sha256"])

    def test_exact_cross_compiler_is_publication_bound(self) -> None:
        published = json.loads(DISASSEMBLY_REVIEW.read_text(encoding="utf-8"))
        compiler = pathlib.Path(shutil.which("aarch64-linux-gnu-g++") or "")
        compiler_id = subprocess.run(
            [str(compiler), "--version"], check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()[0]
        self.assertEqual(sha256(compiler), published["provenance"]["compiler_sha256"])
        self.assertEqual(compiler_id, published["provenance"]["compiler_id"])

    def test_lto_optimized_disassembly_shape(self) -> None:
        binary = pathlib.Path(self.temp.name) / "e055-lto"
        cpp_object = pathlib.Path(self.temp.name) / "e055_lto_cpp.o"
        subprocess.run([
            "aarch64-linux-gnu-g++", "-std=c++17", "-O3", "-flto", "-Wall",
            "-Wextra", "-Werror", "-march=armv8.2-a+dotprod",
            "-c", "tooling/e055_q1_hotcold.cpp", "-o", str(cpp_object),
        ], cwd=ROOT, check=True)
        subprocess.run([
            "aarch64-linux-gnu-g++", "-O3", "-flto", str(cpp_object),
            str(self.asm_object), "-o", str(binary),
        ], cwd=ROOT, check=True)
        report = inspect(str(binary), "aarch64-linux-gnu-objdump")
        self.assertEqual(report["status"], "PASS", report["failures"])
        published = json.loads(DISASSEMBLY_REVIEW.read_text(encoding="utf-8"))
        expected = next(
            item for item in published["builds"] if item["name"] == "O3-flto"
        )
        self.assertEqual(sha256(binary), expected["binary_sha256"])


if __name__ == "__main__":
    unittest.main()
