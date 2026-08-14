#!/usr/bin/env python3
"""Host-side contract tests for the A733 PMU launcher.

The host is not expected to expose the A733 raw events.  These tests therefore
check the safe envelope and the explicit ``unavailable`` result rather than
requiring a particular PMU implementation.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLING = ROOT / "tooling"


class A733PmuExecContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="e049c-pmu-test-"))
        cls.launcher = cls.tmp / "a733-pmu-exec"
        cls.control = cls.tmp / "a733-pmu-control"
        common = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
        subprocess.run(
            [*common, str(TOOLING / "a733_pmu_exec.c"), "-o", str(cls.launcher)],
            check=True,
        )
        subprocess.run(
            [*common, str(TOOLING / "a733_pmu_control.c"), "-o", str(cls.control)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls.tmp.glob("*"):
            path.unlink()
        cls.tmp.rmdir()

    def run_launcher(self, *args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = self.tmp / "result.json"
        command = [
            str(self.launcher),
            "--no-drop",
            "--no-thermal-guard",
            "--output",
            str(output),
            *args,
        ]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertTrue(output.exists(), proc.stderr)
        return proc, json.loads(output.read_text(encoding="utf-8"))

    def assert_common_shape(self, result: dict) -> None:
        self.assertEqual(result["schema_version"], "e049c-arm-pmu/v1")
        self.assertIn(result["status"], {"ok", "partial", "counter_unavailable", "failed"})
        self.assertIsInstance(result["events"], list)
        names = {item["name"] for item in result["events"]}
        self.assertEqual(
            names,
            {
                "cpu_cycles",
                "instructions",
                "l1d_cache_refill",
                "l2d_cache_refill",
                "l3d_cache_refill",
                "mem_access",
                "bus_access",
                "stall_backend",
            },
        )
        for event in result["events"]:
            self.assertIn(
                event["support"],
                {"supported", "unavailable", "open_error", "read_error"},
            )
            if event["support"] != "supported":
                self.assertIn("errno", event)
                self.assertIsInstance(event["error"], str)

    def test_start_immediately_preserves_child_status_and_raw_schema(self) -> None:
        proc, result = self.run_launcher(
            "--start-immediately",
            "--",
            "/bin/sh",
            "-c",
            "exit 0",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_common_shape(result)
        self.assertEqual(result["exit"]["code"], 0)
        self.assertGreaterEqual(result["measured_elapsed_ns"], 0)

    def test_ready_protocol_counts_only_between_markers(self) -> None:
        proc, result = self.run_launcher(
            "--start-on-ready",
            "--sync-timeout-ms",
            "5000",
            "--",
            str(self.control),
            "--mode",
            "cpu",
            "--iterations",
            "1000000",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_common_shape(result)
        self.assertTrue(result["sync"]["started"])
        self.assertTrue(result["sync"]["ended"])
        self.assertGreater(result["measured_elapsed_ns"], 0)
        self.assertEqual(result["exit"]["code"], 0)

    def test_memory_control_is_deterministic_and_records_control_metadata(self) -> None:
        proc, result = self.run_launcher(
            "--start-on-ready",
            "--",
            str(self.control),
            "--mode",
            "memory",
            "--bytes",
            "1048576",
            "--iterations",
            "4",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_common_shape(result)
        self.assertEqual(result["exit"]["code"], 0)
        self.assertEqual(result["command"][1], "--mode")


if __name__ == "__main__":
    unittest.main()
