import json
import inspect
import signal
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "tooling"))
import a733_ddr_pll_guard as guard  # noqa: E402


class ScriptedSampler:
    def __init__(self, *, temperature=40000, drift=False, lock_loss=False):
        self.temperature = temperature
        self.drift = drift
        self.lock_loss = lock_loss
        self.samples = []
        self.target_word = None

    def sample(self, phase):
        if self.drift and self.samples:
            pll = self.target_word ^ 0x1
        elif self.lock_loss and self.samples:
            pll = self.target_word & ~(1 << guard.LOCK_BIT)
        else:
            pll = self.target_word
        clk_summary = (
            "pll-ddr 2400000000 dram0 600000000"
            if self.target_word == 0xF9006300
            else "pll-ddr 2040000000 dram0 510000000"
        )
        observation = {
            "phase": phase,
            "thermal_zones": [{"type": "ddr_thermal_zone", "temp_mC": self.temperature}],
            "cpufreq": [{"policy": "policy0", "current_hz": 1794000}],
            "clk_summary": clk_summary,
            "npu_devfreq": [{"name": "3600000.npu", "current_frequency": "1008000000"}],
            "cooling_devices": [{"type": "thermal-cpufreq-0", "cur_state": "0", "max_state": "10"}],
            "memory": {"mem_available_kB": 10000000, "swap_free_kB": 6000000},
            "registers": {
                "pll_ddr_ctrl": pll,
                "dram_clk_reg": 0x80000003,
            },
        }
        self.samples.append(observation)
        return observation


class MissingTelemetrySampler:
    def sample(self, phase):
        return {"phase": phase, "thermal_zones": [], "cpufreq": [], "clk_summary": None,
                "registers": {"pll_ddr_ctrl": 0xF9005400, "dram_clk_reg": 0x80000003}}


class BadClockSampler(ScriptedSampler):
    def sample(self, phase):
        result = super().sample(phase)
        if self.target_word == 0xF9006300:
            result["clk_summary"] = "pll-ddr 2040000000 dram0 510000000"
        return result


class Watchdog:
    def __init__(self):
        self.armed = 0
        self.disarmed = 0
        self.keepalives = 0
        self.fail_keepalive = False

    def arm(self, timeout_s):
        self.armed += 1

    def disarm(self):
        self.disarmed += 1

    def keepalive(self):
        if self.fail_keepalive:
            raise OSError("mock keepalive failure")
        self.keepalives += 1


class RestoreLockBackend(guard.MockBackend):
    def write32(self, address, value):
        super().write32(address, value)
        if value == 0xF9005400:
            self.registers[address] &= ~(1 << guard.LOCK_BIT)


class WriteAfterSideEffectBackend(guard.MockBackend):
    def __init__(self, registers):
        super().__init__(registers)
        self.failed_target = False

    def write32(self, address, value):
        if value == 0xF9006300 and not self.failed_target:
            self.failed_target = True
            self.registers[address] = value
            self.writes.append(value)
            raise OSError("write transport failure after side effect")
        super().write32(address, value)


def successful_command(argv, timeout_s, abort_event, event_log):
    return {"argv": argv, "stdout": "ok\n", "stderr": "", "returncode": 0, "elapsed_s": 0.01}


def failed_command(argv, timeout_s, abort_event, event_log):
    return {"argv": argv, "stdout": "", "stderr": "failed\n", "returncode": 42, "elapsed_s": 0.01}


class A733DdrPllGuardTests(unittest.TestCase):
    def test_inspect_is_read_only(self):
        backend = guard.MockBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
        with tempfile.TemporaryDirectory() as tmp:
            runner = guard.GuardRunner(
                backend=backend,
                event_log=guard.EventLog(Path(tmp) / "events.jsonl", run_id="inspect"),
                sampler=ScriptedSampler(),
            )
            result = runner.inspect()
        self.assertEqual(result["pll_ddr_ctrl"], 0xF9005400)
        self.assertEqual(backend.writes, [])

    def test_apply_requires_exact_contract(self):
        cases = [
            dict(explicit_apply=False, expected_original=0xF9005400, confirmation=guard.CONFIRMATION),
            dict(explicit_apply=True, expected_original=0xF9006300, confirmation=guard.CONFIRMATION),
            dict(explicit_apply=True, expected_original=0xF9005400, confirmation="wrong"),
            dict(explicit_apply=True, expected_original=0xF9005400, confirmation=guard.CONFIRMATION, target_raw=0x64),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(guard.SafetyError):
                    guard.validate_apply(0xF9005400, **case)

    def test_target_changes_only_pll_raw_field_and_preserves_other_bits(self):
        original = 0xF9005400
        target = guard.compute_target_word(original, 0x63)
        self.assertEqual(target, 0xF9006300)
        self.assertEqual(target & ~(0xFF << 8), original & ~(0xFF << 8))
        self.assertEqual(guard.raw_n(target), 0x63)
        with self.assertRaises(guard.SafetyError):
            guard.compute_target_word(original, 0x55)

    def test_unsafe_addresses_and_invalid_required_bits_are_rejected(self):
        with self.assertRaises(guard.SafetyError):
            guard.validate_address(0x02002021, writable=True)
        with self.assertRaises(guard.SafetyError):
            guard.validate_address(guard.DRAM_CLK_REG, writable=True)
        with self.assertRaises(guard.SafetyError):
            guard.validate_apply(0x00000000, target_raw=0x63, explicit_apply=True,
                                 expected_original=0x00000000, confirmation=guard.CONFIRMATION)

    def make_runner(self, tmp, sampler=None, watchdog=None, command_runner=successful_command):
        backend = guard.MockBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
        return guard.GuardRunner(
            backend=backend,
            event_log=guard.EventLog(Path(tmp) / "events.jsonl", run_id="E005-test"),
            sampler=sampler or ScriptedSampler(),
            watchdog=watchdog or Watchdog(),
            command_runner=command_runner,
            sample_interval_s=0.001,
            lock_path=Path(tmp) / "run.lock",
        ), backend

    def test_success_restores_full_word_and_disarms_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            watchdog = Watchdog()
            runner, backend = self.make_runner(tmp, watchdog=watchdog)
            result = runner.apply(
                explicit_apply=True,
                expected_original=0xF9005400,
                target_raw=0x63,
                confirmation=guard.CONFIRMATION,
            )
        self.assertEqual(result["status"], "restored")
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)
        self.assertEqual(backend.writes, [0xF9006300, 0xF9005400])
        self.assertEqual(watchdog.armed, 1)
        self.assertEqual(watchdog.disarmed, 1)
        self.assertGreater(watchdog.keepalives, 0)

    def test_temperature_abort_restores_and_records_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            watchdog = Watchdog()
            hot = ScriptedSampler(temperature=85000)
            runner, backend = self.make_runner(tmp, sampler=hot, watchdog=watchdog)
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(
                    explicit_apply=True,
                    expected_original=0xF9005400,
                    target_raw=0x63,
                    confirmation=guard.CONFIRMATION,
                )
            rows = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)
        self.assertTrue(any(row["event"] == "abort" for row in rows))

    def test_missing_required_telemetry_fails_closed_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, backend = self.make_runner(tmp, sampler=MissingTelemetrySampler())
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(
                    explicit_apply=True,
                    expected_original=0xF9005400,
                    target_raw=0x63,
                    confirmation=guard.CONFIRMATION,
                )
        self.assertEqual(backend.writes, [])

    def test_workload_failure_restores_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            watchdog = Watchdog()
            runner, backend = self.make_runner(tmp, watchdog=watchdog, command_runner=failed_command)
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(
                    explicit_apply=True,
                    expected_original=0xF9005400,
                    target_raw=0x63,
                    confirmation=guard.CONFIRMATION,
                )
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)
        self.assertEqual(watchdog.disarmed, 0)

    def test_watchdog_heartbeat_failure_aborts_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            watchdog = Watchdog()
            watchdog.fail_keepalive = True
            runner, backend = self.make_runner(tmp, watchdog=watchdog)
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(explicit_apply=True, expected_original=0xF9005400,
                             target_raw=0x63, confirmation=guard.CONFIRMATION)
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)

    def test_candidate_clock_summary_mismatch_aborts_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, backend = self.make_runner(tmp, sampler=BadClockSampler())
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(explicit_apply=True, expected_original=0xF9005400,
                             target_raw=0x63, confirmation=guard.CONFIRMATION)
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)

    def test_restore_lock_loss_is_reported_after_restore_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = RestoreLockBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
            runner = guard.GuardRunner(
                backend=backend,
                event_log=guard.EventLog(Path(tmp) / "events.jsonl", run_id="restore-lock"),
                sampler=ScriptedSampler(), watchdog=Watchdog(), command_runner=successful_command,
                sample_interval_s=0.001,
                lock_path=Path(tmp) / "run.lock",
            )
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(explicit_apply=True, expected_original=0xF9005400,
                             target_raw=0x63, confirmation=guard.CONFIRMATION)
        self.assertEqual(backend.writes[-1], 0xF9005400)

    def test_write_exception_after_side_effect_still_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = WriteAfterSideEffectBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
            runner = guard.GuardRunner(
                backend=backend,
                event_log=guard.EventLog(Path(tmp) / "events.jsonl", run_id="write-fault"),
                sampler=ScriptedSampler(), watchdog=Watchdog(), command_runner=successful_command,
                sample_interval_s=0.001,
                lock_path=Path(tmp) / "run.lock",
            )
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(explicit_apply=True, expected_original=0xF9005400,
                             target_raw=0x63, confirmation=guard.CONFIRMATION)
        self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)

    def test_apply_rejects_missing_watchdog_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, backend = self.make_runner(tmp, watchdog=None)
            runner.watchdog = None
            with self.assertRaises(guard.SafetyError):
                runner.apply(
                    explicit_apply=True,
                    expected_original=0xF9005400,
                    target_raw=0x63,
                    confirmation=guard.CONFIRMATION,
                )
        self.assertEqual(backend.writes, [])

    def test_watchdog_requires_magic_close_capability(self):
        self.assertEqual(guard.validate_watchdog_options(guard.WDIOF_MAGICCLOSE | guard.WDIOF_KEEPALIVEPING), True)
        with self.assertRaises(guard.SafetyError):
            guard.validate_watchdog_options(0)
        with self.assertRaises(guard.SafetyError):
            guard.validate_watchdog_options(guard.WDIOF_MAGICCLOSE)

    def test_workload_json_requires_absolute_allowlisted_executable_and_strict_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.json"
            valid.write_text(json.dumps({
                "name": "q1-short",
                "argv": ["/usr/local/bin/q1_cpu_operator_runner", "--iters", "1"],
                "timeout_s": 30,
            }))
            specs = guard.load_workload_specs([valid])
            self.assertEqual(specs[0]["name"], "q1-short")
            self.assertEqual(specs[0]["argv"][0], "/usr/local/bin/q1_cpu_operator_runner")
            bad_relative = Path(tmp) / "relative.json"
            bad_relative.write_text(json.dumps({"name": "x", "argv": ["q1_cpu_operator_runner"], "timeout_s": 1}))
            bad_extra = Path(tmp) / "extra.json"
            bad_extra.write_text(json.dumps({"name": "x", "argv": ["/bin/mbw"], "timeout_s": 1, "shell": True}))
            for path in (bad_relative, bad_extra):
                with self.subTest(path=path):
                    with self.assertRaises(guard.SafetyError):
                        guard.load_workload_specs([path])
            nested_shell = Path(tmp) / "nested-shell.json"
            nested_shell.write_text(json.dumps({
                "name": "unsafe",
                "argv": ["/usr/bin/taskset", "-c", "0-5", "/bin/sh", "-c", "echo unsafe"],
                "timeout_s": 1,
            }))
            with self.assertRaises(guard.SafetyError):
                guard.load_workload_specs([nested_shell])

    def test_concurrent_run_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.lock"
            first = guard.RunLock(path)
            second = guard.RunLock(path)
            first.acquire()
            try:
                with self.assertRaises(guard.SafetyError):
                    second.acquire()
            finally:
                first.release()

    def test_register_drift_and_lock_loss_restore(self):
        for sampler in (ScriptedSampler(drift=True), ScriptedSampler(lock_loss=True)):
            with self.subTest(sampler=sampler):
                with tempfile.TemporaryDirectory() as tmp:
                    runner, backend = self.make_runner(tmp, sampler=sampler)
                    with self.assertRaises(guard.SafetyAbort):
                        runner.apply(
                            explicit_apply=True,
                            expected_original=0xF9005400,
                            target_raw=0x63,
                            confirmation=guard.CONFIRMATION,
                        )
                    self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)

    def test_lock_poll_is_bounded_and_fails_on_lock_loss(self):
        backend = guard.MockBackend({guard.PLL_DDR_CTRL: 0xF9006300 & ~(1 << guard.LOCK_BIT), guard.DRAM_CLK_REG: 0x80000003})
        with self.assertRaises(guard.SafetyAbort):
            guard.poll_lock(backend, 0xF9006300, timeout_s=0.1, interval_s=0.001)

    def test_signal_handler_requests_abort_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, backend = self.make_runner(tmp)
            runner._handle_signal(signal.SIGTERM, None)
            with self.assertRaises(guard.SafetyAbort):
                runner.apply(
                    explicit_apply=True,
                    expected_original=0xF9005400,
                    target_raw=0x63,
                    confirmation=guard.CONFIRMATION,
                )
            self.assertEqual(backend.registers[guard.PLL_DDR_CTRL], 0xF9005400)

    def test_fixed_workloads_are_argv_only(self):
        workloads = guard.build_workloads()
        self.assertEqual(workloads[0], ["/usr/bin/taskset", "-c", "0-5", "/usr/bin/mbw", "-n", "10", "256"])
        self.assertEqual(workloads[1], ["/usr/bin/taskset", "-c", "0-5", "/usr/local/bin/memtester", "128M", "1"])
        self.assertNotIn("shell", guard.run_command.__code__.co_varnames)
        self.assertNotIn("shell=True", inspect.getsource(guard.run_command))

    def test_devmem_backend_uses_page_aligned_mmap_not_pread_pwrite(self):
        source = inspect.getsource(guard.DevMemBackend)
        self.assertIn("mmap.mmap", source)
        self.assertNotIn("os.pread", source)
        self.assertNotIn("os.pwrite", source)
        self.assertEqual(guard.register_offset(guard.PLL_DDR_CTRL), 0x20)
        self.assertEqual(guard.register_offset(guard.DRAM_CLK_REG), 0xC00)
        fake_mapping = mock.MagicMock()
        with mock.patch.object(guard.os, "open", return_value=9), mock.patch.object(
            guard.mmap, "mmap", return_value=fake_mapping
        ) as mmap_call, mock.patch.object(guard.os, "close"):
            backend = guard.DevMemBackend("/dev/mem", read_only=True)
            backend.close()
        self.assertEqual(mmap_call.call_args.kwargs["offset"], guard.CCU_BASE)

    def test_default_apply_lock_is_global_across_event_logs(self):
        backend = guard.MockBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
        first = guard.GuardRunner(backend=backend, event_log=guard.EventLog(Path(tempfile.gettempdir()) / "a733-one.jsonl", "one"))
        second = guard.GuardRunner(backend=backend, event_log=guard.EventLog(Path(tempfile.gettempdir()) / "a733-two.jsonl", "two"))
        self.assertEqual(first.lock_path, guard.DEFAULT_LOCK_PATH)
        self.assertEqual(second.lock_path, guard.DEFAULT_LOCK_PATH)

    def test_event_log_is_schema_versioned_and_fsynced_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            log = guard.EventLog(path, run_id="r1", host="mock-host")
            log.append("test", phase="before", rc=0, stdout="", stderr="")
            row = json.loads(path.read_text())
        self.assertEqual(row["schema_version"], guard.SCHEMA_VERSION)
        self.assertEqual(row["run_id"], "r1")
        self.assertEqual(row["host"], "mock-host")
        self.assertEqual(row["event"], "test")

    def test_system_sampler_reads_thermal_cpufreq_clocks_and_both_registers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            (root / "class" / "thermal" / "thermal_zone0").mkdir(parents=True)
            (root / "class" / "thermal" / "thermal_zone0" / "type").write_text("ddr_thermal_zone\n")
            (root / "class" / "thermal" / "thermal_zone0" / "temp").write_text("41000\n")
            policy = root / "devices" / "system" / "cpu" / "cpufreq" / "policy0"
            policy.mkdir(parents=True)
            (policy / "scaling_cur_freq").write_text("1794000\n")
            (policy / "scaling_governor").write_text("performance\n")
            clk = root / "kernel" / "debug" / "clk"
            clk.mkdir(parents=True)
            (clk / "clk_summary").write_text("pll-ddr 2040000000\ndram0 510000000\n")
            devfreq = root / "class" / "devfreq" / "3600000.npu"
            devfreq.mkdir(parents=True)
            (devfreq / "cur_freq").write_text("1008000000\n")
            cooling = root / "class" / "thermal" / "cooling_device0"
            cooling.mkdir()
            (cooling / "type").write_text("thermal-cpufreq-0\n")
            (cooling / "cur_state").write_text("0\n")
            (cooling / "max_state").write_text("10\n")
            proc = Path(tmp) / "proc"
            proc.mkdir()
            (proc / "meminfo").write_text("MemAvailable: 10000000 kB\nSwapFree: 6000000 kB\n")
            backend = guard.MockBackend({guard.PLL_DDR_CTRL: 0xF9005400, guard.DRAM_CLK_REG: 0x80000003})
            sample = guard.SystemSampler(backend, sys_root=root, proc_root=proc).sample("before")
        self.assertEqual(sample["thermal_zones"][0]["type"], "ddr_thermal_zone")
        self.assertEqual(sample["cpufreq"][0]["scaling_cur_freq"], "1794000")
        self.assertIn("dram0", sample["clk_summary"])
        self.assertEqual(sample["registers"]["dram_clk_reg"], 0x80000003)
        self.assertEqual(sample["npu_devfreq"][0]["name"], "3600000.npu")
        self.assertEqual(sample["cooling_devices"][0]["cur_state"], "0")
        self.assertEqual(sample["memory"]["swap_free_kB"], 6000000)


if __name__ == "__main__":
    unittest.main()
