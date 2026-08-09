import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tooling" / "thermal_exec_guard.py"
sys.path.insert(0, str(MODULE.parent))
import thermal_exec_guard  # noqa: E402


class ThermalExecGuardTests(unittest.TestCase):
    @staticmethod
    def process_is_running(pid: int) -> bool:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return False
        state = next((line for line in status.splitlines() if line.startswith("State:")), "")
        return "Z (zombie)" not in state

    def make_sysfs(self, root: Path) -> None:
        thermal = root / "class" / "thermal"
        (thermal / "thermal_zone0").mkdir(parents=True)
        (thermal / "thermal_zone0" / "type").write_text("cpu-thermal\n")
        (thermal / "thermal_zone0" / "temp").write_text("42000\n")
        (thermal / "thermal_zone1").mkdir()
        (thermal / "thermal_zone1" / "type").write_text("gpu-thermal\n")
        (thermal / "thermal_zone1" / "temp").write_text("41000\n")
        cdev = thermal / "cooling_device0"
        cdev.mkdir()
        (cdev / "type").write_text("thermal-cpufreq-0\n")
        (cdev / "cur_state").write_text("0\n")
        (cdev / "max_state").write_text("10\n")
        policy = root / "devices" / "system" / "cpu" / "cpufreq" / "policy0"
        policy.mkdir(parents=True)
        (policy / "affected_cpus").write_text("0-3\n")
        (policy / "scaling_cur_freq").write_text("1200000\n")
        (policy / "scaling_max_freq").write_text("1800000\n")
        (policy / "scaling_governor").write_text("ondemand\n")

    def read_trace(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def run_guard(self, sys_root: Path, trace: Path, command: list[str], **kwargs: object) -> int:
        guard = thermal_exec_guard.Guard(
            trace=trace,
            sys_root=sys_root,
            **kwargs,
        )
        return guard.run(command)

    def test_success_records_inventory_samples_and_child_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            result = self.run_guard(
                root,
                trace,
                [sys.executable, "-c", "print('child-out'); import sys; print('child-err', file=sys.stderr)"],
                interval_ms=10,
            )
            self.assertEqual(result, 0)
            rows = self.read_trace(trace)
            self.assertEqual(rows[0]["event"], "start")
            inventory = next(row for row in rows if row.get("event") == "inventory")
            self.assertEqual(len(inventory["thermal_zones"]), 2)
            samples = [row for row in rows if row.get("kind") == "sample"]
            self.assertTrue(samples)
            self.assertEqual({"path", "type", "millidegrees_c"}, set(samples[0]["thermal_zones"][0]))
            self.assertIn("cpu_policies", samples[0])
            self.assertIn("cooling_devices", samples[0])
            child_started = next(row for row in rows if row.get("event") == "child_started")
            self.assertEqual(samples[0]["process"]["pid"], child_started["pid"])
            self.assertIn("available", samples[0]["process"])
            child_exits = [row for row in rows if row.get("event") == "child_exit"]
            self.assertEqual(len(child_exits), 1)
            self.assertEqual(child_exits[0]["returncode"], 0)

    def test_temperature_ceiling_terminates_process_group_with_distinct_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            temp = root / "class" / "thermal" / "thermal_zone0" / "temp"
            trace = Path(tmp) / "trace.jsonl"

            def raise_temperature() -> None:
                time.sleep(0.06)
                temp.write_text("85000\n")

            changer = threading.Thread(target=raise_temperature)
            changer.start()
            result = self.run_guard(
                root,
                trace,
                [sys.executable, "-c", "import time; time.sleep(5)"],
                interval_ms=10,
            )
            changer.join()
            self.assertEqual(result, thermal_exec_guard.EXIT_ABORT)
            rows = self.read_trace(trace)
            abort = next(row for row in rows if row.get("event") == "abort")
            self.assertIn("temperature", abort["reason"])
            self.assertTrue(any(row.get("event") == "terminated" for row in rows))

    def test_temperature_abort_kills_sigterm_ignoring_grandchild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            temp = root / "class" / "thermal" / "thermal_zone0" / "temp"
            trace = Path(tmp) / "trace.jsonl"
            pid_file = Path(tmp) / "grandchild.pid"
            ready_file = Path(tmp) / "grandchild.ready"
            grandchild_code = (
                "import signal,time,pathlib; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"pathlib.Path({str(ready_file)!r}).write_text('ready'); "
                "time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"p=subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]); "
                f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
                "time.sleep(30)"
            )

            def raise_temperature() -> None:
                deadline = time.monotonic() + 2
                while not ready_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                temp.write_text("85000\n")

            changer = threading.Thread(target=raise_temperature)
            changer.start()
            grandchild_pid = -1
            try:
                result = self.run_guard(
                    root,
                    trace,
                    [sys.executable, "-c", parent_code],
                    interval_ms=10,
                )
                changer.join()
                self.assertEqual(result, thermal_exec_guard.EXIT_ABORT)
                grandchild_pid = int(pid_file.read_text())
                deadline = time.monotonic() + 2
                while self.process_is_running(grandchild_pid) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(self.process_is_running(grandchild_pid))
            finally:
                if grandchild_pid > 0 and self.process_is_running(grandchild_pid):
                    os.kill(grandchild_pid, signal.SIGKILL)

    def test_sensor_loss_terminates_child_and_marks_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            temp = root / "class" / "thermal" / "thermal_zone0" / "temp"
            trace = Path(tmp) / "trace.jsonl"

            def remove_sensor() -> None:
                time.sleep(0.06)
                temp.unlink()

            changer = threading.Thread(target=remove_sensor)
            changer.start()
            result = self.run_guard(
                root,
                trace,
                [sys.executable, "-c", "import time; time.sleep(5)"],
                interval_ms=10,
            )
            changer.join()
            self.assertEqual(result, thermal_exec_guard.EXIT_ABORT)
            self.assertIn("sensor", next(row for row in self.read_trace(trace) if row.get("event") == "abort")["reason"])

    def test_cooling_device_identity_change_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            cooling_type = root / "class" / "thermal" / "cooling_device0" / "type"
            trace = Path(tmp) / "trace.jsonl"

            def replace_device_identity() -> None:
                time.sleep(0.06)
                cooling_type.write_text("replacement-device\n")

            changer = threading.Thread(target=replace_device_identity)
            changer.start()
            result = self.run_guard(
                root,
                trace,
                [sys.executable, "-c", "import time; time.sleep(5)"],
                interval_ms=10,
            )
            changer.join()
            self.assertEqual(result, thermal_exec_guard.EXIT_ABORT)
            abort = next(row for row in self.read_trace(trace) if row.get("event") == "abort")
            self.assertIn("cooling", abort["reason"])

    def test_same_type_cooling_device_replacement_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            thermal = root / "class" / "thermal"
            original = thermal / "cooling_device0"
            replacement = thermal / "replacement"
            replacement.mkdir()
            (replacement / "type").write_text("thermal-cpufreq-0\n")
            (replacement / "cur_state").write_text("0\n")
            (replacement / "max_state").write_text("10\n")
            backup = thermal / "old-cooling-device"
            guard = thermal_exec_guard.Guard(trace=trace, sys_root=root, interval_ms=10)
            original_sample = guard.reader.sample
            replaced = False

            def replace_then_sample(inventory: thermal_exec_guard.Inventory) -> dict[str, object]:
                nonlocal replaced
                if not replaced:
                    original.rename(backup)
                    replacement.rename(original)
                    replaced = True
                return original_sample(inventory)

            guard.reader.sample = replace_then_sample  # type: ignore[method-assign]
            result = guard.run([sys.executable, "-c", "import time; time.sleep(5)"])
            self.assertEqual(result, thermal_exec_guard.EXIT_ABORT)
            abort = next(row for row in self.read_trace(trace) if row.get("event") == "abort")
            self.assertIn("cooling", abort["reason"])

    def test_signal_wins_over_simultaneous_sampling_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            guard = thermal_exec_guard.Guard(trace=trace, sys_root=root, interval_ms=10)

            def interrupted_sample(_inventory: thermal_exec_guard.Inventory) -> dict[str, object]:
                guard._signal_handler(signal.SIGTERM, None)
                raise thermal_exec_guard.GuardFailure("sensor_loss:simulated-race")

            guard.reader.sample = interrupted_sample  # type: ignore[method-assign]
            result = guard.run([sys.executable, "-c", "import time; time.sleep(5)"])
            self.assertEqual(result, 128 + signal.SIGTERM)
            rows = self.read_trace(trace)
            self.assertTrue(any(row.get("event") == "signal" for row in rows))

    def test_existing_trace_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text("do-not-append\n")
            result = self.run_guard(root, trace, [sys.executable, "-c", "pass"], interval_ms=10)
            self.assertEqual(result, thermal_exec_guard.EXIT_PREFLIGHT)
            self.assertEqual(trace.read_text(), "do-not-append\n")

    def test_trace_failure_before_launch_returns_preflight_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            with mock.patch.object(
                thermal_exec_guard.Trace,
                "write",
                side_effect=thermal_exec_guard.GuardFailure("trace_write:test"),
            ):
                result = self.run_guard(root, trace, [sys.executable, "-c", "pass"], interval_ms=10)
            self.assertEqual(result, thermal_exec_guard.EXIT_PREFLIGHT)

    def test_child_nonzero_exit_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            result = self.run_guard(
                root,
                trace,
                [sys.executable, "-c", "raise SystemExit(7)"],
                interval_ms=10,
            )
            self.assertEqual(result, 7)
            self.assertEqual(next(row for row in self.read_trace(trace) if row.get("event") == "child_exit")["returncode"], 7)

    def test_sigterm_terminates_child_group_and_returns_signal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            root.mkdir()
            self.make_sysfs(root)
            trace = Path(tmp) / "trace.jsonl"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(MODULE),
                    "--sys-root",
                    str(root),
                    "--interval-ms",
                    "10",
                    "--trace",
                    str(trace),
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if trace.exists() and '"event": "child_started"' in trace.read_text():
                    break
                time.sleep(0.01)
            else:
                self.fail("guard did not start the child before signal deadline")
            process.send_signal(signal.SIGTERM)
            result = process.wait(timeout=3)
            self.assertEqual(result, 128 + signal.SIGTERM)
            rows = self.read_trace(trace)
            self.assertEqual(next(row for row in rows if row.get("event") == "signal")["signal"], signal.SIGTERM)
            self.assertTrue(any(row.get("event") == "terminated" for row in rows))


if __name__ == "__main__":
    unittest.main()
