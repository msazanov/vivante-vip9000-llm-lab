#!/usr/bin/env python3
"""Host-side safety and protocol tests for the A733 PMU v2 launcher."""

from __future__ import annotations

import errno
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
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
        cls.proc_stat_override = cls.tmp / "e049c-proc-stat-override.so"
        cls.result_index = 0
        common = [
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DE049C_TESTING",
        ]
        subprocess.run(
            [*common, str(TOOLING / "a733_pmu_exec.c"), "-o", str(cls.launcher)],
            check=True,
        )
        subprocess.run(
            [*common, str(TOOLING / "a733_pmu_control.c"), "-o", str(cls.control)],
            check=True,
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-shared", "-fPIC",
                str(ROOT / "tests" / "e049c_proc_stat_override.c"),
                "-o", str(cls.proc_stat_override),
            ],
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

    def wait_for_pid_file(self, path: pathlib.Path, timeout: float = 5.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists() and path.read_text(encoding="ascii").strip():
                return int(path.read_text(encoding="ascii").strip())
            time.sleep(0.01)
        self.fail(f"timed out waiting for PID file: {path}")

    def wait_for_pid_pair(
        self, path: pathlib.Path, timeout: float = 5.0,
    ) -> tuple[int, int]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                fields = path.read_text(encoding="ascii").split()
                if len(fields) == 2:
                    return int(fields[0]), int(fields[1])
            time.sleep(0.01)
        self.fail(f"timed out waiting for PID pair: {path}")

    def assert_process_gone(self, pid: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            status_path = pathlib.Path(f"/proc/{pid}/stat")
            if status_path.exists():
                fields = status_path.read_text(encoding="ascii").split()
                if len(fields) > 2 and fields[2] == "Z":
                    time.sleep(0.01)
                    continue
            time.sleep(0.01)
        self.fail(f"process {pid} survived launcher cleanup")

    def write_helper(self, name: str, source: str) -> pathlib.Path:
        path = self.tmp / name
        path.write_text(source, encoding="utf-8")
        return path

    def isolated_child_stream_args(self, label: str) -> tuple[str, ...]:
        return (
            "--child-stdout", str(self.tmp / f"{label}.stdout"),
            "--child-stderr", str(self.tmp / f"{label}.stderr"),
        )

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

    def test_post_open_validation_failure_unlinks_only_new_output(self) -> None:
        output = self.tmp / "validation-failure.json"
        marker = self.tmp / "validation-failure.ran"
        sentinel = self.tmp / "validation-failure.sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        environment = dict(os.environ)
        environment["E049C_TEST_FAIL_OUTPUT_VALIDATION"] = str(output)
        proc = subprocess.run(
            [
                str(self.launcher), "--no-drop", "--no-thermal-guard",
                "--output", str(output), "--event-group", "core",
                "--start-immediately", "--", "/bin/sh", "-c",
                f"printf ran > '{marker}'",
            ],
            text=True, capture_output=True, timeout=15, env=environment,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(output.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_child_stdout_and_stderr_are_separate_from_wrapper_streams(self) -> None:
        child_stdout = self.tmp / "child-separated.stdout"
        child_stderr = self.tmp / "child-separated.stderr"
        proc, result = self.run_launcher(
            "--child-stdout",
            str(child_stdout),
            "--child-stderr",
            str(child_stderr),
            "--event-group",
            "core",
            "--start-immediately",
            "--",
            "/bin/sh",
            "-c",
            "printf child-out; printf child-err >&2",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["exit"]["code"], 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, "")
        self.assertEqual(child_stdout.read_text(encoding="utf-8"), "child-out")
        self.assertEqual(child_stderr.read_text(encoding="utf-8"), "child-err")

    def test_child_output_collision_fails_before_workload_and_cleans_new_files(self) -> None:
        for collision in ("existing", "symlink"):
            with self.subTest(collision=collision):
                output = self.tmp / f"child-collision-{collision}.json"
                child_stdout = self.tmp / f"child-collision-{collision}.stdout"
                child_stderr = self.tmp / f"child-collision-{collision}.stderr"
                marker = self.tmp / f"child-collision-{collision}.ran"
                if collision == "existing":
                    child_stderr.write_text("keep", encoding="utf-8")
                else:
                    sentinel = self.tmp / f"child-collision-{collision}.sentinel"
                    sentinel.write_text("keep", encoding="utf-8")
                    child_stderr.symlink_to(sentinel)
                proc = subprocess.run(
                    [
                        str(self.launcher), "--no-drop", "--no-thermal-guard",
                        "--output", str(output),
                        "--child-stdout", str(child_stdout),
                        "--child-stderr", str(child_stderr),
                        "--event-group", "core", "--start-immediately", "--",
                        "/bin/sh", "-c", f"printf ran > '{marker}'",
                    ],
                    text=True, capture_output=True, timeout=15,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse(marker.exists())
                self.assertFalse(output.exists())
                self.assertFalse(child_stdout.exists())
                if collision == "existing":
                    self.assertEqual(child_stderr.read_text(encoding="utf-8"), "keep")
                else:
                    self.assertTrue(child_stderr.is_symlink())

    def test_child_capture_fds_survive_fixed_marker_fd_collision(self) -> None:
        output = self.next_output()
        child_stdout = self.tmp / "child-fd-collision.stdout"
        child_stderr = self.tmp / "child-fd-collision.stderr"
        inherited = [os.open(os.devnull, os.O_RDONLY) for _ in range(2)]
        try:
            proc = subprocess.run(
                [
                    str(self.launcher), "--no-drop", "--no-thermal-guard",
                    "--output", str(output),
                    "--child-stdout", str(child_stdout),
                    "--child-stderr", str(child_stderr),
                    "--event-group", "core", "--start-on-ready",
                    "--sync-timeout-ms", "1000", "--",
                    str(self.control), "--mode", "cpu", "--iterations", "10000",
                ],
                pass_fds=tuple(inherited), text=True, capture_output=True, timeout=15,
            )
        finally:
            for descriptor in inherited:
                os.close(descriptor)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["sync"]["acknowledged"])
        self.assertIn(b"control=cpu iterations=10000", child_stdout.read_bytes())
        self.assertEqual(child_stderr.read_bytes(), b"")

    def test_child_exec_failure_is_captured_without_polluting_wrapper_stderr(self) -> None:
        child_stdout = self.tmp / "child-exec-failure.stdout"
        child_stderr = self.tmp / "child-exec-failure.stderr"
        proc, result = self.run_launcher(
            "--child-stdout", str(child_stdout),
            "--child-stderr", str(child_stderr),
            "--event-group", "core", "--start-immediately", "--",
            "/definitely/missing/e055-child",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(child_stdout.read_bytes(), b"")
        self.assertIn("execvp", child_stderr.read_text(encoding="utf-8"))
        self.assertEqual(result["exit"]["code"], 127)

    def test_external_sigterm_terminates_and_reaps_child_process_group(self) -> None:
        output = self.next_output()
        child_stdout = self.tmp / "signal-child.stdout"
        child_stderr = self.tmp / "signal-child.stderr"
        pid_file = self.tmp / "signal-child.pids"
        command = [
            str(self.launcher), "--no-drop", "--no-thermal-guard",
            "--output", str(output), "--child-stdout", str(child_stdout),
            "--child-stderr", str(child_stderr), "--event-group", "core",
            "--start-immediately", "--", "/bin/sh", "-c",
            f"sleep 30 & printf '%s %s' $$ $! > '{pid_file}'; wait",
        ]
        launcher = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        child_pid = descendant_pid = 0
        try:
            child_pid, descendant_pid = self.wait_for_pid_pair(pid_file)
            launcher.send_signal(signal.SIGTERM)
            _, wrapper_stderr = launcher.communicate(timeout=10)
            self.assertEqual(launcher.returncode, 128 + signal.SIGTERM, wrapper_stderr)
            self.assertGreater(output.stat().st_size, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_reason"], "parent_cancelled")
            self.assert_process_gone(descendant_pid)
            self.assert_process_gone(child_pid)
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(timeout=5)
            if child_pid > 0:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_external_sigterm_reaps_descendant_that_escaped_with_setsid(self) -> None:
        output = self.next_output()
        pid_file = self.tmp / "signal-setsid-descendant.pid"
        leader = self.write_helper(
            "signal-setsid-leader.py",
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n"
            "time.sleep(30)\n",
        )
        command = [
            str(self.launcher), "--no-drop", "--no-thermal-guard",
            "--output", str(output),
            *self.isolated_child_stream_args("signal-setsid"),
            "--event-group", "core", "--start-immediately", "--",
            sys.executable, str(leader), str(pid_file),
        ]
        launcher = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        descendant_pid = 0
        try:
            descendant_pid = self.wait_for_pid_file(pid_file)
            launcher.send_signal(signal.SIGTERM)
            _, wrapper_stderr = launcher.communicate(timeout=10)
            self.assertEqual(launcher.returncode, 128 + signal.SIGTERM, wrapper_stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["failure_reason"], "parent_cancelled")
            self.assert_process_gone(descendant_pid)
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(timeout=5)
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_post_end_nonzero_exit_kills_remaining_descendant(self) -> None:
        pid_file = self.tmp / "post-end-nonzero.pids"
        child_stdout = self.tmp / "post-end-nonzero.stdout"
        child_stderr = self.tmp / "post-end-nonzero.stderr"
        proc = None
        child_pid = descendant_pid = 0
        try:
            proc, result = self.run_launcher(
                "--child-stdout", str(child_stdout), "--child-stderr",
                str(child_stderr), "--event-group", "core", "--start-on-ready",
                "--sync-timeout-ms", "1000", "--", "/bin/sh", "-c",
                f"printf S >&9; read ack <&8; sleep 30 & "
                f"printf '%s %s' $$ $! > '{pid_file}'; printf E >&9; exit 7",
            )
            child_pid, descendant_pid = self.wait_for_pid_pair(pid_file)
            self.assertEqual(proc.returncode, 7)
            self.assertEqual(result["exit"]["code"], 7)
            self.assertEqual(result["failure_reason"], "child_exit_nonzero")
            self.assert_process_gone(descendant_pid)
        finally:
            if child_pid > 0:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_immediate_exit_zero_cleans_remaining_descendant_before_success(self) -> None:
        pid_file = self.tmp / "immediate-exit-zero.pids"
        child_stdout = self.tmp / "immediate-exit-zero.stdout"
        child_stderr = self.tmp / "immediate-exit-zero.stderr"
        child_pid = descendant_pid = 0
        try:
            proc, result = self.run_launcher(
                "--child-stdout", str(child_stdout), "--child-stderr",
                str(child_stderr), "--event-group", "core", "--start-immediately",
                "--", "/bin/sh", "-c",
                f"sleep 30 & printf '%s %s' $$ $! > '{pid_file}'; exit 0",
            )
            child_pid, descendant_pid = self.wait_for_pid_pair(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(result["exit"]["code"], 0)
            self.assert_process_gone(descendant_pid)
            self.assert_process_gone(child_pid)
        finally:
            if child_pid > 0:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_start_ack_end_exit_zero_cleans_descendant_before_success(self) -> None:
        pid_file = self.tmp / "sae-exit-zero.pids"
        child_stdout = self.tmp / "sae-exit-zero.stdout"
        child_stderr = self.tmp / "sae-exit-zero.stderr"
        child_pid = descendant_pid = 0
        try:
            proc, result = self.run_launcher(
                "--child-stdout", str(child_stdout), "--child-stderr",
                str(child_stderr), "--event-group", "core", "--start-on-ready",
                "--sync-timeout-ms", "1000", "--", "/bin/sh", "-c",
                f"printf S >&9; read ack <&8; sleep 30 & "
                f"printf '%s %s' $$ $! > '{pid_file}'; printf E >&9; exit 0",
            )
            child_pid, descendant_pid = self.wait_for_pid_pair(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(result["exit"]["code"], 0)
            self.assertTrue(result["sync"]["ended"])
            self.assert_process_gone(descendant_pid)
            self.assert_process_gone(child_pid)
        finally:
            if child_pid > 0:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_immediate_exit_zero_cleans_descendant_that_calls_setsid(self) -> None:
        pid_file = self.tmp / "immediate-setsid.pid"
        leader = self.write_helper(
            "immediate-setsid-leader.py",
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n",
        )
        descendant_pid = 0
        try:
            proc, result = self.run_launcher(
                *self.isolated_child_stream_args("immediate-setsid"),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(pid_file),
            )
            descendant_pid = self.wait_for_pid_file(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(result["exit"]["code"], 0)
            self.assert_process_gone(descendant_pid)
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_start_ack_end_exit_zero_cleans_descendant_in_new_process_group(self) -> None:
        pid_file = self.tmp / "sae-new-pgid.pid"
        leader = self.write_helper(
            "sae-new-pgid-leader.py",
            "import os, pathlib, subprocess, sys\n"
            "os.write(9, b'S')\n"
            "os.read(8, 2)\n"
            "child = subprocess.Popen(\n"
            "    ['/bin/sleep', '30'], preexec_fn=lambda: os.setpgid(0, 0)\n"
            ")\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n"
            "os.write(9, b'E')\n",
        )
        descendant_pid = 0
        try:
            proc, result = self.run_launcher(
                *self.isolated_child_stream_args("sae-new-pgid"),
                "--event-group", "core", "--start-on-ready", "--sync-timeout-ms",
                "1000", "--", sys.executable, str(leader), str(pid_file),
            )
            descendant_pid = self.wait_for_pid_file(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(result["sync"]["ended"])
            self.assert_process_gone(descendant_pid)
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_nested_escaped_descendants_are_adopted_and_reaped(self) -> None:
        pid_file = self.tmp / "nested-escaped.pids"
        nested = self.write_helper(
            "nested-escaped-child.py",
            "import os, pathlib, subprocess, sys, time\n"
            "grandchild = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)\n"
            "pathlib.Path(sys.argv[1]).write_text(\n"
            "    f'{os.getpid()} {grandchild.pid}', encoding='ascii'\n"
            ")\n"
            "time.sleep(30)\n",
        )
        leader = self.write_helper(
            "nested-escaped-leader.py",
            "import pathlib, subprocess, sys, time\n"
            "subprocess.Popen(\n"
            "    [sys.executable, sys.argv[1], sys.argv[2]], start_new_session=True\n"
            ")\n"
            "path = pathlib.Path(sys.argv[2])\n"
            "deadline = time.monotonic() + 2\n"
            "while not path.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.005)\n"
            "if not path.exists():\n"
            "    raise RuntimeError('nested helper did not publish PIDs')\n",
        )
        nested_pid = grandchild_pid = 0
        try:
            proc, result = self.run_launcher(
                *self.isolated_child_stream_args("nested-escaped"),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(nested), str(pid_file),
            )
            nested_pid, grandchild_pid = self.wait_for_pid_pair(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(result["exit"]["code"], 0)
            self.assert_process_gone(nested_pid)
            self.assert_process_gone(grandchild_pid)
        finally:
            for pid in (nested_pid, grandchild_pid):
                if pid > 0:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_cleanup_does_not_signal_unrelated_control_process(self) -> None:
        pid_file = self.tmp / "unrelated-control-descendant.pid"
        leader = self.write_helper(
            "unrelated-control-leader.py",
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n",
        )
        control = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        descendant_pid = 0
        try:
            proc, _ = self.run_launcher(
                *self.isolated_child_stream_args("unrelated-control"),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(pid_file),
            )
            descendant_pid = self.wait_for_pid_file(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assert_process_gone(descendant_pid)
            self.assertIsNone(control.poll(), "wrapper signaled an unrelated sibling")
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if control.poll() is None:
                os.killpg(control.pid, signal.SIGKILL)
            control.wait(timeout=5)

    def test_unrelated_proc_record_with_zero_start_time_is_not_malformed(self) -> None:
        original = pathlib.Path("/proc/1/stat").read_text(encoding="ascii")
        closing = original.rfind(")")
        self.assertGreater(closing, 0)
        fields = original[closing + 2 :].split()
        self.assertGreaterEqual(len(fields), 20)
        fields[19] = "0"  # /proc stat field 22: unsigned process start time.
        fixture = self.tmp / "unrelated-proc1-zero-start-time.stat"
        fixture.write_text(
            original[: closing + 1] + " " + " ".join(fields) + "\n",
            encoding="ascii",
        )
        output = self.next_output()
        environment = os.environ.copy()
        environment["LD_PRELOAD"] = str(self.proc_stat_override)
        environment["E049C_TEST_PROC1_STAT"] = str(fixture)
        proc = subprocess.run(
            [
                str(self.launcher), "--no-drop", "--no-thermal-guard",
                "--output", str(output),
                *self.isolated_child_stream_args("zero-start-time"),
                "--event-group", "core", "--start-immediately", "--",
                "/bin/true",
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotEqual(result["failure_reason"], "process_tree_cleanup_failed")

    def run_owned_identity_adversary(
        self, mode: str, label: str, *, expect_cleanup_failure: bool
    ) -> tuple[dict, int]:
        pid_file = self.tmp / f"{label}.pid"
        leader = self.write_helper(
            f"{label}-leader.py",
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n",
        )
        output = self.next_output()
        environment = os.environ.copy()
        environment["LD_PRELOAD"] = str(self.proc_stat_override)
        environment[mode] = "1"
        proc = subprocess.run(
            [
                str(self.launcher), "--no-drop", "--no-thermal-guard",
                "--output", str(output),
                *self.isolated_child_stream_args(label),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(pid_file),
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
        )
        descendant_pid = self.wait_for_pid_file(pid_file)
        result = json.loads(output.read_text(encoding="utf-8"))
        if expect_cleanup_failure:
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(result["failure_reason"], "process_tree_cleanup_failed")
        else:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        os.kill(descendant_pid, 0)
        return result, descendant_pid

    def test_owned_process_requires_nonzero_recorded_start_time(self) -> None:
        descendant_pid = 0
        try:
            _, descendant_pid = self.run_owned_identity_adversary(
                "E049C_TEST_ZERO_OWNED_START", "zero-owned-start-time",
                expect_cleanup_failure=True,
            )
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_replaced_owned_identity_is_never_signaled(self) -> None:
        descendant_pid = 0
        try:
            _, descendant_pid = self.run_owned_identity_adversary(
                "E049C_TEST_REPLACE_OWNED_START", "replaced-owned-start-time",
                expect_cleanup_failure=False,
            )
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_proc_exit_races_do_not_create_false_cleanup_failure(self) -> None:
        pid_file = self.tmp / "proc-race.pids"
        leader = self.write_helper(
            "proc-race-leader.py",
            "import pathlib, subprocess, sys\n"
            "children = [subprocess.Popen(\n"
            "    ['/bin/sleep', '0.005' if index % 2 == 0 else '30'],\n"
            "    start_new_session=True\n"
            ") for index in range(32)]\n"
            "pathlib.Path(sys.argv[1]).write_text(\n"
            "    ' '.join(str(child.pid) for child in children), encoding='ascii'\n"
            ")\n",
        )
        child_pids: list[int] = []
        try:
            proc, result = self.run_launcher(
                *self.isolated_child_stream_args("proc-race"),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(pid_file),
            )
            child_pids = [
                int(value) for value in pid_file.read_text(encoding="ascii").split()
            ]
            self.assertEqual(len(child_pids), 32)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotEqual(result["failure_reason"], "process_tree_cleanup_failed")
            for pid in child_pids:
                self.assert_process_gone(pid)
        finally:
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_term_resistant_escaped_descendant_is_killed_and_reaped(self) -> None:
        pid_file = self.tmp / "term-resistant.pid"
        resistant = self.write_helper(
            "term-resistant-child.py",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n",
        )
        leader = self.write_helper(
            "term-resistant-leader.py",
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen(\n"
            "    [sys.executable, sys.argv[1]], start_new_session=True\n"
            ")\n"
            "pathlib.Path(sys.argv[2]).write_text(str(child.pid), encoding='ascii')\n",
        )
        descendant_pid = 0
        started = time.monotonic()
        try:
            proc, result = self.run_launcher(
                *self.isolated_child_stream_args("term-resistant"),
                "--event-group", "core", "--start-immediately", "--",
                sys.executable, str(leader), str(resistant), str(pid_file),
            )
            descendant_pid = self.wait_for_pid_file(pid_file)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(result["exit"]["code"], 0)
            self.assertLess(time.monotonic() - started, 5.0)
            self.assert_process_gone(descendant_pid)
        finally:
            if descendant_pid > 0:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_closed_ack_pipe_is_failure_not_sigpipe_and_cleans_descendants(self) -> None:
        output = self.next_output()
        child_stdout = self.tmp / "closed-ack.stdout"
        child_stderr = self.tmp / "closed-ack.stderr"
        pid_file = self.tmp / "closed-ack.pids"
        command = [
            str(self.launcher), "--no-drop", "--no-thermal-guard",
            "--output", str(output), "--child-stdout", str(child_stdout),
            "--child-stderr", str(child_stderr), "--event-group", "core",
            "--start-on-ready", "--sync-timeout-ms", "1000", "--",
            "/bin/sh", "-c", f"exec 8<&-; sleep 30 & "
            f"printf '%s %s' $$ $! > '{pid_file}'; printf S >&9; wait",
        ]
        child_pid = descendant_pid = 0
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=10)
            child_pid, descendant_pid = self.wait_for_pid_pair(pid_file)
            self.assertNotEqual(proc.returncode, -signal.SIGPIPE)
            self.assertGreater(output.stat().st_size, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_reason"], "ack_write_failed")
            self.assert_process_gone(descendant_pid)
        finally:
            if child_pid > 0:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

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
        child_stdout = self.tmp / "timeout-child.stdout"
        child_stderr = self.tmp / "timeout-child.stderr"
        proc, result = self.run_launcher(
            "--child-stdout",
            str(child_stdout),
            "--child-stderr",
            str(child_stderr),
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
        self.assertEqual(child_stdout.read_bytes(), b"")
        self.assertEqual(child_stderr.read_bytes(), b"")
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
