#!/usr/bin/env python3
"""Host-side safety and protocol tests for the A733 PMU v2 launcher."""

from __future__ import annotations

import errno
import json
import math
import os
import pathlib
import subprocess
import tempfile
import time
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLING = ROOT / "tooling"


class A733PmuExecContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="e049c-pmu-v2-test-"))
        cls.launcher = cls.tmp / "a733-pmu-exec"
        cls.control = cls.tmp / "a733-pmu-control"
        cls.result_index = 0
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
        for path in cls.tmp.iterdir():
            if path.is_dir():
                for child in path.iterdir():
                    child.unlink()
                path.rmdir()
            else:
                path.unlink()
        cls.tmp.rmdir()

    def next_output(self) -> pathlib.Path:
        type(self).result_index += 1
        return self.tmp / f"result-{self.result_index}.json"

    def run_launcher(
        self, *args: str, thermal_guard: bool = False
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = self.next_output()
        command = [str(self.launcher), "--no-drop", "--output", str(output)]
        if not thermal_guard:
            command.append("--no-thermal-guard")
        command.extend(args)
        proc = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertTrue(output.exists(), proc.stderr)
        return proc, json.loads(output.read_text(encoding="utf-8"))

    def assert_common_shape(self, result: dict, group: str) -> None:
        self.assertEqual(result["schema_version"], "e049c-arm-pmu/v2")
        self.assertEqual(result["event_group"], group)
        self.assertIn(result["status"], {"ok", "counter_unavailable", "failed"})
        self.assertIsInstance(result["sample_valid"], bool)
        self.assertIsInstance(result["events"], list)
        self.assertLessEqual(len(result["events"]), 4)
        expected = {
            "core": {"cpu_cycles", "instructions", "stall_backend"},
            "cache": {"l1d_cache_refill", "l2d_cache_refill", "l3d_cache_refill"},
            "memory": {"mem_access", "bus_access"},
        }[group]
        self.assertEqual({item["name"] for item in result["events"]}, expected)
        for event in result["events"]:
            self.assertIn(event["support"], {"supported", "unavailable", "read_error"})
            self.assertIsInstance(event["sample_valid"], bool)
            if event["support"] != "supported":
                self.assertIsInstance(event["errno"], int)
                self.assertIsInstance(event["error"], str)

    def test_explicit_groups_fit_conservative_software_limit(self) -> None:
        expected_sizes = {"core": 3, "cache": 3, "memory": 2}
        for group, expected_size in expected_sizes.items():
            proc, result = self.run_launcher(
                "--event-group",
                group,
                "--start-on-ready",
                "--sync-timeout-ms",
                "5000",
                "--",
                str(self.control),
                "--mode",
                "cpu",
                "--iterations",
                "10000",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assert_common_shape(result, group)
            self.assertNotIn("event_group_capacity", result)
            self.assertEqual(result["software_group_size_limit"], 4)
            self.assertEqual(result["event_group_size"], expected_size)
            self.assertEqual(
                result["software_group_size_limit_semantics"],
                "conservative launcher policy; not measured hardware PMU capacity",
            )

    def test_nonfinite_double_options_are_rejected_before_workload(self) -> None:
        for option in ("--min-running-ratio", "--max-temp-c"):
            for spelling in ("nan", "NaN", "inf", "-inf"):
                with self.subTest(option=option, spelling=spelling):
                    output = self.next_output()
                    marker = output.with_suffix(".ran")
                    proc = subprocess.run(
                        [
                            str(self.launcher),
                            "--no-drop",
                            "--no-thermal-guard",
                            "--output",
                            str(output),
                            option,
                            spelling,
                            "--event-group",
                            "core",
                            "--start-immediately",
                            "--",
                            "/bin/sh",
                            "-c",
                            f"printf ran > '{marker}'",
                        ],
                        text=True,
                        capture_output=True,
                        timeout=15,
                    )
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertFalse(output.exists())
                    self.assertFalse(marker.exists())
                    self.assertIn(f"invalid {option}", proc.stderr)

    def test_json_contains_only_finite_numbers(self) -> None:
        proc, result = self.run_launcher(
            "--event-group",
            "core",
            "--start-on-ready",
            "--",
            str(self.control),
            "--mode",
            "cpu",
            "--iterations",
            "10000",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

        def assert_finite(value: object) -> None:
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), value)
            elif isinstance(value, list):
                for item in value:
                    assert_finite(item)
            elif isinstance(value, dict):
                for item in value.values():
                    assert_finite(item)

        assert_finite(result)

    def test_ready_protocol_requires_parent_ack_before_work(self) -> None:
        proc, result = self.run_launcher(
            "--event-group",
            "core",
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
        self.assert_common_shape(result, "core")
        self.assertEqual(
            result["sync"],
            {
                "mode": "start_ack_end",
                "started": True,
                "acknowledged": True,
                "ended": True,
            },
        )
        self.assertGreater(result["measured_elapsed_ns"], 0)
        if result["status"] != "ok":
            self.assertFalse(result["sample_valid"])

    def test_memory_control_is_deterministic_and_records_group(self) -> None:
        proc, result = self.run_launcher(
            "--event-group",
            "memory",
            "--start-on-ready",
            "--",
            str(self.control),
            "--mode",
            "memory",
            "--bytes",
            "1048576",
            "--iterations",
            "2",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_common_shape(result, "memory")
        self.assertEqual(result["exit"]["code"], 0)
        self.assertEqual(result["command"][1], "--mode")

    def test_fixed_fd9_shell_shim_has_no_eval_or_proc_fd(self) -> None:
        shim = (TOOLING / "a733_pmu_sync_exec.sh").read_text(encoding="utf-8")
        self.assertNotIn("eval", shim)
        self.assertNotIn("/proc/self/fd", shim)
        self.assertIn(">&9", shim)
        self.assertIn("<&8", shim)
        proc, result = self.run_launcher(
            "--event-group",
            "core",
            "--start-on-ready",
            "--",
            str(TOOLING / "a733_pmu_sync_exec.sh"),
            "/bin/true",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(result["sync"]["acknowledged"])
        self.assertTrue(result["sync"]["ended"])

    def test_fixed_sync_fds_survive_pipe_source_target_collision(self) -> None:
        """Two inherited fds make marker=8 and ACK=9 before remapping."""

        output = self.next_output()
        inherited = [os.open(os.devnull, os.O_RDONLY) for _ in range(2)]
        try:
            proc = subprocess.run(
                [
                    str(self.launcher),
                    "--no-drop",
                    "--no-thermal-guard",
                    "--output",
                    str(output),
                    "--event-group",
                    "core",
                    "--start-on-ready",
                    "--sync-timeout-ms",
                    "1000",
                    "--",
                    str(self.control),
                    "--mode",
                    "cpu",
                    "--iterations",
                    "10000",
                ],
                pass_fds=tuple(inherited),
                text=True,
                capture_output=True,
                timeout=15,
            )
        finally:
            for descriptor in inherited:
                os.close(descriptor)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            result["sync"],
            {
                "mode": "start_ack_end",
                "started": True,
                "acknowledged": True,
                "ended": True,
            },
        )

    def test_thermal_unreadable_fails_closed(self) -> None:
        proc, result = self.run_launcher(
            "--thermal-root",
            str(self.tmp / "missing-thermal-root"),
            "--event-group",
            "core",
            "--start-immediately",
            "--",
            "/bin/sleep",
            "5",
            thermal_guard=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "thermal_unreadable")
        self.assertFalse(result["thermal"]["readable"])
        self.assertFalse(result["sample_valid"])

    def test_output_symlink_is_rejected_without_clobber(self) -> None:
        sentinel = self.tmp / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        output = self.tmp / "unsafe-result.json"
        marker = self.tmp / "unsafe-result.ran"
        output.symlink_to(sentinel)
        proc = subprocess.run(
            [
                str(self.launcher),
                "--no-drop",
                "--no-thermal-guard",
                "--output",
                str(output),
                "--event-group",
                "core",
                "--start-immediately",
                "--",
                "/bin/sh",
                "-c",
                f"printf ran > '{marker}'",
            ],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertTrue(output.is_symlink())
        self.assertFalse(marker.exists())

    def test_existing_output_is_rejected_before_workload(self) -> None:
        output = self.tmp / "existing-result.json"
        marker = self.tmp / "existing-result.ran"
        output.write_text("keep", encoding="utf-8")
        proc = subprocess.run(
            [
                str(self.launcher),
                "--no-drop",
                "--no-thermal-guard",
                "--output",
                str(output),
                "--event-group",
                "core",
                "--start-immediately",
                "--",
                "/bin/sh",
                "-c",
                f"printf ran > '{marker}'",
            ],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")
        self.assertFalse(marker.exists())

    def test_capacity_provenance_does_not_claim_hardware_capacity(self) -> None:
        main_readme = (
            ROOT / "experiments" / "E049c-arm-pmu" / "README.md"
        ).read_text(encoding="utf-8")
        v2_readme = (
            ROOT
            / "experiments"
            / "E049c-arm-pmu"
            / "results"
            / "v2-20260814"
            / "README.md"
        ).read_text(encoding="utf-8")
        for document in (main_readme, v2_readme):
            normalized = " ".join(document.split())
            self.assertIn("software", normalized.lower())
            self.assertIn("3/3/2", normalized)
            self.assertIn("не доказывает аппаратную ёмкость", normalized)

    def test_marker_timeout_kills_and_reaps_process_group(self) -> None:
        pid_file = self.tmp / "grandchild.pid"
        proc, result = self.run_launcher(
            "--event-group",
            "core",
            "--start-on-ready",
            "--sync-timeout-ms",
            "300",
            "--",
            "/bin/sh",
            "-c",
            f"sleep 30 & printf '%s' $! > '{pid_file}'; wait",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "start_marker_timeout")
        self.assertFalse(result["sample_valid"])
        self.assertTrue(pid_file.exists(), proc.stderr)
        grandchild = int(pid_file.read_text(encoding="ascii"))
        for _ in range(50):
            try:
                os.kill(grandchild, 0)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    break
            time.sleep(0.01)
        else:
            self.fail(f"grandchild {grandchild} survived/remaind unreaped")


if __name__ == "__main__":
    unittest.main()
