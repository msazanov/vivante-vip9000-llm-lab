"""Fresh cross/QEMU and optimized-disassembly gates for E055."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tooling.check_e055_disassembly import inspect


ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless(all(shutil.which(x) for x in (
    "aarch64-linux-gnu-g++", "aarch64-linux-gnu-objdump", "qemu-aarch64")),
    "AArch64 cross/QEMU tools are required")
class E055AArch64Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="e055-gate-")
        cls.binary = pathlib.Path(cls.temp.name) / "e055"
        subprocess.run([
            "aarch64-linux-gnu-g++", "-std=c++17", "-O3", "-Wall", "-Wextra",
            "-Werror", "-march=armv8.2-a+dotprod",
            str(ROOT / "tooling/e055_q1_hotcold.cpp"),
            str(ROOT / "experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S"),
            "-o", str(cls.binary),
        ], check=True)

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
                                  "cold_conditioned", "--working-set-bytes", "65536",
                                  "--thrash-bytes", "65536")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["schema"], "e055-q1-hot-cold-harness/v1")
        self.assertEqual((data["iterations"], data["calls"]), (1, 1))

    def test_cold_rejects_iteration_and_budget_overrides(self) -> None:
        for args in (("--iterations", "2"), ("--budget-ms", "10")):
            with self.subTest(args=args):
                result = self.run_harness("--cache-state", "cold_conditioned", *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --iterations 1", result.stderr)

    def test_sync_without_fds_fails_closed(self) -> None:
        result = self.run_harness("--iterations", "1", "--sync")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fd9/fd8 are unavailable", result.stderr)

    def test_optimized_disassembly_shape(self) -> None:
        report = inspect(str(self.binary), "aarch64-linux-gnu-objdump")
        self.assertEqual(report["status"], "PASS", report["failures"])


if __name__ == "__main__":
    unittest.main()
