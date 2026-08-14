"""Safety and exact-scope tests for the disabled E055 target executor."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tooling.e055_target_executor import (
    E055TargetExecutor,
    TargetCapture,
    TargetRunFailure,
    limited_o3_microgate_plan,
)
from tooling.e055_raw_bundle import load_sealed_bundle


ROOT = Path(__file__).resolve().parents[1]


class FakeTransport:
    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.prepared = []
        self.captures = []
        self.restored = 0

    def prepare(self, artifacts: tuple) -> None:
        self.prepared.extend(artifacts)

    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.captures.append((run, e049c_argv, dict(environment)))
        ordinal = len(self.captures)
        failed = self.fail_at == ordinal
        calls = run.iterations
        elapsed_ns = calls * 1_000_000
        if run.cache_state == "cold_conditioned":
            conditioning = {
                "strategy": "verified_write_read_each_64B_line",
                "requested_bytes": 67_108_864,
                "actual_bytes": 67_108_864,
                "line_bytes": 64,
                "lines_touched": 1_048_576,
                "checksum": "0x8d3ea13d15850279",
                "verified_touched": True,
                "warmup_calls": 0,
            }
        else:
            conditioning = {
                "strategy": "verified_kernel_warmup",
                "requested_bytes": run.actual_working_set_bytes,
                "actual_bytes": run.actual_working_set_bytes,
                "line_bytes": 64,
                "lines_touched": (run.actual_working_set_bytes + 63) // 64,
                "checksum": "0xe055c002",
                "verified_touched": True,
                "warmup_calls": 16,
            }
        harness = {
            "schema": "e055-q1-hot-cold-harness/v1",
            "mode": run.mode,
            "cache_state": run.cache_state,
            "cpu": run.cpu,
            "q1_layout": "E039 stock native block_q1_0x4 4x4 DOTPROD",
            "golden_pass": True,
            "golden_cases": 18,
            "target_working_set_bytes": run.target_working_set_bytes,
            "actual_working_set_bytes": run.actual_working_set_bytes,
            "blocks": run.blocks,
            "iterations": calls,
            "calls": calls,
            "elapsed_ns": elapsed_ns,
            "first_call_ns": 1_000_000,
            "calls_per_second": 1000.0,
            "logical_bytes_per_call": {
                "q1_packed_bytes": run.blocks * 72,
                "q8_bytes": run.blocks * 4 * 34,
                "total_input_bytes": run.blocks * 208,
                "output_bytes": 16,
                "dot_products": run.blocks * 512,
            },
            "checksum": "0xe055d001",
            "cold_conditioning": conditioning,
            "sync": {
                "requested": True, "started": True, "acknowledged": True,
                "ended": True, "sequence": "S/A/E",
            },
            "qualification": "unqualified_harness_output_requires_E049c_join",
        }
        harness_argv = list(e049c_argv[e049c_argv.index("--") + 1:])
        configs = (("cpu_cycles", "0x11"), ("instructions", "0x8"),
                   ("stall_backend", "0x24"))
        e049c = {
            "schema_version": "e049c-arm-pmu/v2",
            "status": "ok",
            "sample_valid": True,
            "event_source": "armv8_pmuv3_raw_config",
            "counter_semantics": "event counts only; no DDR-byte conversion",
            "event_group": "core",
            "software_group_size_limit": 4,
            "event_group_size": 3,
            "software_group_size_limit_semantics": (
                "conservative launcher policy; not measured hardware PMU capacity"
            ),
            "min_running_ratio": 0.95,
            "pid": 1000 + ordinal,
            "process_group": 1000 + ordinal,
            "command": harness_argv,
            "sync": {
                "mode": "start_ack_end", "started": True,
                "acknowledged": True, "ended": True,
            },
            "measured_elapsed_ns": elapsed_ns + 1000,
            "thermal": {
                "guard_enabled": True, "checked": True, "readable": True,
                "limit_c": 85.0, "max_observed_c": 55.0, "tripped": False,
            },
            "exit": {"code": 0, "raw_wait_status": 0},
            "failure_reason": None,
            "events": [{
                "name": name, "config": config,
                "meaning": f"ARMv8 PMUv3 {name} event count",
                "support": "supported", "count_semantics": "event_count_not_bytes",
                "sample_valid": True, "value": 100 + index,
                "time_enabled_ns": elapsed_ns, "time_running_ns": elapsed_ns,
                "running_ratio": 1.0, "errno": 0, "error": None,
            } for index, (name, config) in enumerate(configs)],
        }
        return TargetCapture(
            child_stdout=(json.dumps(harness, sort_keys=True,
                                     separators=(",", ":")) + "\n").encode("ascii"),
            child_stderr=b"exec failed\n" if failed else b"",
            e049c_json=(json.dumps(e049c, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("ascii"),
            wrapper_stdout=b"",
            wrapper_stderr=b"",
            exit_code=127 if failed else 0,
            signal=None,
            effective_cpus=(run.cpu,),
            cpu_start=run.cpu,
            cpu_end=run.cpu,
            migration_count=0,
        )

    def restore(self) -> None:
        self.restored += 1


class RaisingTransport(FakeTransport):
    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.captures.append((run, e049c_argv, dict(environment)))
        raise OSError("injected transport failure")


class MalformedTransport(FakeTransport):
    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.captures.append((run, e049c_argv, dict(environment)))
        return TargetCapture(
            child_stdout="not bytes",  # type: ignore[arg-type]
            child_stderr=b"", e049c_json=b"{}\n", wrapper_stdout=b"",
            wrapper_stderr=b"", exit_code=0, signal=None,
            effective_cpus=(run.cpu,), cpu_start=run.cpu, cpu_end=run.cpu,
            migration_count=0,
        )


class E055TargetExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e055-executor-")
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "E055 Executor Test"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "e055@example.invalid"],
                       cwd=self.root, check=True)
        experiment = self.root / "experiments/E055-q1-hot-cold"
        experiment.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plan_is_exact_bounded_twenty_run_o3_microgate(self) -> None:
        plan = limited_o3_microgate_plan()
        self.assertEqual(len(plan), 20)
        self.assertEqual({item.cpu for item in plan}, {0, 6})
        self.assertEqual({item.build_name for item in plan}, {"O3"})
        self.assertEqual({item.mode for item in plan}, {"full_dotprod"})
        self.assertEqual({item.pmu_group for item in plan}, {"core"})
        self.assertEqual({item.target_working_set_bytes for item in plan}, {65536})
        for cpu in (0, 6):
            scoped = [item for item in plan if item.cpu == cpu]
            self.assertEqual(len(scoped), 10)
            for pair_index in range(1, 6):
                pair = [item for item in scoped if item.pair_index == pair_index]
                expected_states = (
                    ["hot_repeat", "cold_conditioned"] if pair_index % 2
                    else ["cold_conditioned", "hot_repeat"]
                )
                self.assertEqual([item.cache_state for item in pair], expected_states)
                self.assertEqual([item.order_index for item in pair], [1, 2])

    def test_no_injected_transport_means_no_files_or_target_side_effect(self) -> None:
        executor = E055TargetExecutor(self.root)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            executor.execute("phase-a")
        self.assertFalse((self.root / "experiments/E055-q1-hot-cold/raw").exists())

    def test_fake_transport_populates_exact_streams_runner_and_manifest(self) -> None:
        transport = FakeTransport()
        result = E055TargetExecutor(self.root, transport=transport).execute("phase-a")
        self.assertEqual(result.run_count, 20)
        self.assertEqual(len(transport.prepared), 2)
        self.assertEqual({item.target_path for item in transport.prepared}, {
            "/tmp/a733-pmu-exec",
            "experiments/E055-q1-hot-cold/raw/phase-a/artifacts/harness-O3.bin",
        })
        self.assertEqual(len(transport.captures), 20)
        self.assertEqual(transport.restored, 1)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["runs"]), 20)
        self.assertTrue(manifest["target_workload_executed"])
        first = manifest["runs"][0]
        run_dir = result.phase_dir / "runs" / first["run_id"]
        runner = json.loads((run_dir / "runner.json").read_text(encoding="utf-8"))
        launcher = runner["e049c_argv"]
        self.assertEqual(launcher[launcher.index("--child-stdout") + 1],
                         first["artifacts"]["e049c_json"]["path"].replace(
                             "e049c.json", "target-harness.stdout.raw"
                         ))
        self.assertEqual(launcher[launcher.index("--child-stderr") + 1],
                         first["artifacts"]["e049c_json"]["path"].replace(
                             "e049c.json", "target-harness.stderr.raw"
                         ))
        stdout_envelope = json.loads(
            (run_dir / "harness.stdout.capture.json").read_text(encoding="utf-8")
        )
        decoded_stdout = json.loads(
            base64.b64decode(stdout_envelope["payload_base64"]).decode("ascii")
        )
        self.assertEqual(decoded_stdout["mode"], "full_dotprod")
        executable = result.phase_dir / "artifacts/harness-O3.bin"
        published = ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64"
        self.assertEqual(hashlib.sha256(executable.read_bytes()).hexdigest(),
                         hashlib.sha256(published.read_bytes()).hexdigest())
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "sealed fake phase"],
                       cwd=self.root, check=True)
        sealed = load_sealed_bundle(result.manifest_path)
        self.assertEqual(len(sealed.samples), 20)

    def test_failed_target_run_preserves_captured_bytes_and_stops(self) -> None:
        transport = FakeTransport(fail_at=3)
        executor = E055TargetExecutor(self.root, transport=transport)
        with self.assertRaisesRegex(TargetRunFailure, "cpu0-pair2-cold"):
            executor.execute("phase-fail")
        self.assertEqual(len(transport.captures), 3)
        self.assertEqual(transport.restored, 1)
        phase = self.root / "experiments/E055-q1-hot-cold/raw/phase-fail"
        self.assertEqual((phase / "bundle.json").stat().st_size, 0)
        failed = phase / "runs/cpu0-pair2-cold"
        stderr_envelope = json.loads(
            (failed / "harness.stderr.capture.json").read_text(encoding="utf-8")
        )
        self.assertEqual(base64.b64decode(stderr_envelope["payload_base64"]),
                         b"exec failed\n")
        failure = json.loads((failed / "runner.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["schema"], "e055-runner-failure/v1")
        self.assertEqual(failure["exit"], {"code": 127, "signal": None})

    def test_transport_exception_is_recorded_without_leaking_exception_text(self) -> None:
        transport = RaisingTransport()
        with self.assertRaisesRegex(TargetRunFailure, "partial phase preserved"):
            E055TargetExecutor(self.root, transport=transport).execute("phase-error")
        self.assertEqual(transport.restored, 1)
        runner = self.root / (
            "experiments/E055-q1-hot-cold/raw/phase-error/"
            "runs/cpu0-pair1-hot/runner.json"
        )
        document = json.loads(runner.read_text(encoding="utf-8"))
        self.assertEqual(document["failure_kind"], "OSError")
        self.assertNotIn("injected transport failure", runner.read_text(encoding="utf-8"))

    def test_existing_phase_refuses_before_transport_prepare(self) -> None:
        raw = self.root / "experiments/E055-q1-hot-cold/raw"
        raw.mkdir()
        (raw / "phase-existing").mkdir()
        transport = FakeTransport()
        with self.assertRaises(FileExistsError):
            E055TargetExecutor(self.root, transport=transport).execute("phase-existing")
        self.assertEqual(transport.prepared, [])
        self.assertEqual(transport.captures, [])
        self.assertEqual(transport.restored, 0)

    def test_malformed_transport_value_fails_closed_and_records_failure(self) -> None:
        transport = MalformedTransport()
        with self.assertRaises(TargetRunFailure):
            E055TargetExecutor(self.root, transport=transport).execute("phase-malformed")
        runner = self.root / (
            "experiments/E055-q1-hot-cold/raw/phase-malformed/"
            "runs/cpu0-pair1-hot/runner.json"
        )
        document = json.loads(runner.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "e055-runner-failure/v1")
        self.assertEqual(transport.restored, 1)


if __name__ == "__main__":
    unittest.main()
