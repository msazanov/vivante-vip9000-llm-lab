"""Host-only tests for the fail-closed A733 NSI calibration helper."""

from __future__ import annotations

import hashlib
import math
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tooling.nsi_calibrate import (
    JsonlTrace,
    NSIError,
    PINNED_HELPER_SHA256,
    SysfsNSI,
    TimerRestoreFailure,
    classify_unit_hypothesis,
    fit_line,
    plan_exact_read_bytes,
    resolve_pinned_helper,
    raw_thermal_maxima,
    run_under_watchdog,
    summarize,
    timer_window,
    validate_thermal_limit,
    validate_window_alignment,
    write_partial_failure,
    _run_reader,
    _run_idle_window,
    main,
)


def make_fake_sysfs(root: Path, timer: int = 0) -> Path:
    """Create only the files that the helper is allowed to read/write."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pmu_timer").write_text(f"{timer}\n", encoding="utf-8")
    for name in SysfsNSI.PMU_READ_FILES:
        if name == "available_pmu":
            value = "npu cpu0 cpu1 total\n"
        elif name == "pmu_timer":
            continue
        else:
            value = "1 2 3 4\n"
        (root / name).write_text(value, encoding="utf-8")
    # A sentinel proves that the helper does not discover or write port files.
    (root / "port_mode").write_text("sentinel\n", encoding="utf-8")
    return root


class FailingRestoreNSI(SysfsNSI):
    """Fail on the second timer write, which is the restoration write."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, allow_test_filesystem=True)
        self.write_count = 0

    def _write_timer_raw(self, value: int) -> None:
        self.write_count += 1
        if self.write_count == 2:
            raise OSError("simulated restore I/O failure")
        super()._write_timer_raw(value)


class NsiCalibrationUnitTest(unittest.TestCase):
    def test_pinned_helper_sha_matches_target_v4_build(self) -> None:
        self.assertEqual(
            PINNED_HELPER_SHA256,
            "0506e6cff3f22816b3b89c7a334b57af6e71945cb3106c561ed46869af5a5d82",
        )

    def test_read_plan_is_exact_aligned_and_rejects_nonfinite_calibration(self) -> None:
        ready = {
            "buffer_bytes": 32 * 1024 * 1024,
            "calibration_bytes": 16 * 1024 * 1024,
            "calibration_elapsed_ns": 8_000_000,
            "deadline_chunk_bytes": 4096,
            "deadline_guard_ns": 1_000_000,
        }
        plan = plan_exact_read_bytes(ready, window_us=100_000)
        self.assertEqual(plan["planned_bytes"], 32 * 1024 * 1024)
        self.assertEqual(plan["planned_bytes"] % ready["deadline_chunk_bytes"], 0)
        self.assertLessEqual(plan["planned_active_budget_ns"], 50_000_000)

        invalid = dict(ready, calibration_elapsed_ns=math.nan)
        with self.assertRaises(NSIError):
            plan_exact_read_bytes(invalid, window_us=100_000)

    def test_window_alignment_gate_rejects_overrun_outside_workload_and_bad_bytes(self) -> None:
        alignment = {
            "programmed_window_us": 100_000,
            "pmu_arm_before_ns": 1_000,
            "pmu_arm_after_ns": 2_000,
            "pmu_deadline_earliest_ns": 100_001_000,
            "pmu_deadline_latest_ns": 100_002_000,
            "pmu_read_start_ns": 100_002_500,
            "pmu_read_end_ns": 100_012_500,
            "workload_start_ns": 3_000,
            "workload_end_ns": 50_003_000,
            "planned_bytes": 4096,
            "bytes_read": 4096,
        }
        checked = validate_window_alignment(alignment)
        self.assertTrue(checked["workload_fully_contained"])
        self.assertLessEqual(checked["elapsed_programmed_ratio"], 1.01)
        self.assertEqual(checked["active_us"], 50_000.0)
        self.assertEqual(checked["idle_tail_us"], 49_998.0)

        mutations = (
            {"pmu_read_start_ns": 101_100_000},
            {"workload_start_ns": 1_500},
            {"workload_end_ns": 100_001_001},
            {"bytes_read": 4088},
            {"pmu_read_start_ns": math.nan},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(NSIError):
                validate_window_alignment({**alignment, **mutation})

    def test_reader_finishes_exact_bytes_then_holds_idle_to_pmu_deadline(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "nsi_sequential_read"
            compiled = subprocess.run(
                ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-pedantic", str(repo / "tooling" / "nsi_sequential_read.c"),
                 "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            nsi = SysfsNSI(
                make_fake_sysfs(root / "nsi", timer=17),
                allow_test_filesystem=True,
            )
            trace = JsonlTrace(root / "raw.jsonl", "host-integration")
            result, samples, _snapshot = _run_reader(
                binary,
                1,
                100_000,
                nsi,
                root / "empty-sysfs",
                85.0,
                trace,
                mode="read",
                window_us=100_000,
            )
            alignment = result["window_alignment"]
            self.assertEqual(result["bytes_read"], result["planned_bytes"])
            self.assertTrue(alignment["workload_fully_contained"])
            self.assertGreater(alignment["idle_tail_us"], 0.0)
            self.assertLessEqual(alignment["elapsed_programmed_ratio"], 1.01)
            self.assertEqual(nsi.read_timer(), 17)
            self.assertIsInstance(samples, list)

    def test_idle_window_uses_same_elapsed_ratio_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nsi = SysfsNSI(
                make_fake_sysfs(root / "nsi", timer=29),
                allow_test_filesystem=True,
            )
            trace = JsonlTrace(root / "raw.jsonl", "idle-integration")
            samples, _snapshot, alignment, _telemetry = _run_idle_window(
                nsi,
                100_000,
                root / "empty-sysfs",
                85.0,
                trace,
                mode="idle",
                buffer_mib=1,
                window_us=100_000,
            )
            self.assertEqual(alignment["programmed_window_us"], 100_000)
            self.assertGreaterEqual(alignment["elapsed_programmed_ratio"], 1.0)
            self.assertLessEqual(alignment["elapsed_programmed_ratio"], 1.01)
            self.assertEqual(nsi.read_timer(), 29)
            self.assertIsInstance(samples, list)

    def test_c_helper_reports_exact_architected_load_bytes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "nsi_sequential_read"
            compiled = subprocess.run(
                ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-pedantic", str(root / "tooling" / "nsi_sequential_read.c"),
                 "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            start_read_fd, start_write_fd = os.pipe()
            environment = dict(os.environ)
            environment["E049_START_FD"] = str(start_read_fd)
            process = subprocess.Popen(
                [str(binary), "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                pass_fds=(start_read_fd,),
            )
            os.close(start_read_fd)
            assert process.stdout is not None
            ready_line = process.stdout.readline()
            if not ready_line.startswith("READY "):
                os.close(start_write_fd)
                stdout_tail, stderr = process.communicate(timeout=5)
                self.fail(f"helper did not announce JSON READY: {ready_line!r} {stdout_tail!r} {stderr!r}")
            ready = __import__("json").loads(ready_line.removeprefix("READY "))
            self.assertGreater(ready["calibration_bytes"], 0)
            self.assertGreater(ready["calibration_elapsed_ns"], 0)
            planned_bytes = 256 * 1024
            deadline_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) + 1_000_000_000
            os.write(start_write_fd, f"{planned_bytes} {deadline_ns}\n".encode("ascii"))
            os.close(start_write_fd)
            stdout_tail, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            result = __import__("json").loads(stdout_tail)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["load_width_bytes"], 8)
            self.assertEqual(result["bytes_read"], planned_bytes)
            self.assertEqual(result["planned_bytes"], planned_bytes)
            self.assertLessEqual(result["workload_end_ns"], deadline_ns)
            self.assertEqual(
                result["active_ns"],
                result["workload_end_ns"] - result["workload_start_ns"],
            )

    def test_thermal_limit_rejects_nan_and_infinity(self) -> None:
        for value in (math.nan, math.inf, -math.inf, -1.0, 85.001):
            with self.subTest(value=value), self.assertRaises(NSIError):
                validate_thermal_limit(value)

    def test_timer_symlink_is_rejected_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nsi"
            root.mkdir()
            victim = Path(tmp) / "victim"
            victim.write_text("17\n", encoding="utf-8")
            (root / "pmu_timer").symlink_to(victim)
            with self.assertRaises(NSIError):
                SysfsNSI(root, allow_test_filesystem=True)
            self.assertEqual(victim.read_text(encoding="utf-8"), "17\n")

    def test_pinned_helper_rejects_wrong_path_or_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "nsi_calibrate.py"
            script.write_text("# trusted location\n", encoding="utf-8")
            helper = Path(tmp) / "nsi_sequential_read"
            helper.write_bytes(b"known-helper")
            helper.chmod(0o755)
            with self.assertRaises(NSIError):
                resolve_pinned_helper(script, "0" * 64)

    def test_watchdog_restores_timer_after_child_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fake_sysfs(Path(tmp), timer=19)
            nsi = SysfsNSI(root, allow_test_filesystem=True)

            def killed_child() -> int:
                nsi.write_timer(500)
                os.kill(os.getpid(), signal.SIGKILL)
                return 0

            result = run_under_watchdog(nsi, killed_child)
            self.assertEqual(result["signal"], signal.SIGKILL)
            self.assertTrue(result["restore_verified"])
            self.assertEqual(SysfsNSI(root, allow_test_filesystem=True).read_timer(), 19)

    def test_watchdog_restores_timer_after_child_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fake_sysfs(Path(tmp), timer=23)
            nsi = SysfsNSI(root, allow_test_filesystem=True)

            def terminated_child() -> int:
                nsi.write_timer(250)
                os.kill(os.getpid(), signal.SIGTERM)
                return 0

            result = run_under_watchdog(nsi, terminated_child)
            self.assertEqual(result["signal"], signal.SIGTERM)
            self.assertTrue(result["restore_verified"])
            self.assertEqual(SysfsNSI(root, allow_test_filesystem=True).read_timer(), 23)

    def test_timer_window_restores_exact_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fake_sysfs(Path(tmp), timer=37)
            nsi = SysfsNSI(root, allow_test_filesystem=True)
            with timer_window(nsi, 250):
                self.assertEqual(nsi.read_timer(), 250)
            self.assertEqual(nsi.read_timer(), 37)
            self.assertEqual((root / "port_mode").read_text(encoding="utf-8"),
                             "sentinel\n")

    def test_restore_failure_is_explicit_and_not_silenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nsi = FailingRestoreNSI(make_fake_sysfs(Path(tmp), timer=11))
            with self.assertRaises(TimerRestoreFailure) as raised:
                with timer_window(nsi, 100):
                    self.assertEqual(nsi.read_timer(), 100)
            self.assertIn("восстанов", str(raised.exception).lower())
            # The first write was successful; the failed second write must not
            # be replaced with a best-effort write to any other sysfs control.
            self.assertEqual(nsi.write_count, 2)

    def test_rejects_a_timer_path_that_is_not_exactly_pmu_timer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "port_mode").write_text("0\n", encoding="utf-8")
            with self.assertRaises(NSIError):
                SysfsNSI(root / "port_mode")

    def test_parse_snapshot_has_numeric_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nsi = SysfsNSI(make_fake_sysfs(Path(tmp)), allow_test_filesystem=True)
            snapshot = nsi.snapshot()
            self.assertEqual(snapshot["pmu_bandwidth_rd"], [1, 2, 3, 4])
            self.assertEqual(snapshot["available_pmu"], ["npu", "cpu0", "cpu1", "total"])

    def test_fit_and_unit_classification_prefers_rate_when_timer_matters(self) -> None:
        # The synthetic counter is proportional to bytes / measurement window.
        points = []
        for size in (32, 64, 128, 256):
            for timer_ms in (100, 250, 500, 1000):
                x = float(size * 1024 * 1024)
                y = 7.0 * x / (timer_ms / 1000.0)
                points.append({"bytes": x, "active_us": timer_ms * 1000, "value": y})
        fit = fit_line([p["bytes"] for p in points], [p["value"] for p in points])
        self.assertGreaterEqual(fit["r2"], 0.0)
        result = classify_unit_hypothesis(points)
        self.assertEqual(result["classification"], "rate")
        self.assertGreater(result["rate_fit"]["r2"], 0.99)

    def test_fit_line_rejects_degenerate_x(self) -> None:
        with self.assertRaises(NSIError):
            fit_line([1.0, 1.0], [2.0, 3.0])

    def test_summary_reports_cv_per_identical_cell_not_only_aggregate(self) -> None:
        points = []
        for value in (100, 110, 90):
            points.append({
                "mode": "read", "buffer_mib": 32, "window_us": 100_000,
                "pmu": {"pmu_bandwidth_rd": [value]},
                "helper_result": {
                    "planned_bytes": 32 * 1024 * 1024,
                    "bytes_read": 32 * 1024 * 1024,
                    "window_alignment": {
                        "active_us": 50_000.0,
                        "idle_tail_us": 50_000.0,
                        "elapsed_programmed_ratio": 1.0,
                        "workload_fully_contained": True,
                    },
                },
                "telemetry": {"thermal_c": {"cpu": 40.0}},
            })
        summary = summarize(points, ["reported_total_channel"],
                            original_timer_raw=0, thermal_limit_c=85.0,
                            status="complete")
        cells = summary["signals"]["pmu_bandwidth_rd"]["cell_cv"]
        self.assertIn("32MiB@100000us", cells)
        self.assertAlmostEqual(cells["32MiB@100000us"], 0.081649658, places=6)
        self.assertEqual(summary["selected_channel_role"], "reported aggregate channel; not proven sum")

    def test_summary_fits_exact_bytes_and_active_rate_separately(self) -> None:
        points = []
        for bytes_read, active_us in (
            (8_000_000, 10_000.0),
            (16_000_000, 40_000.0),
            (24_000_000, 20_000.0),
            (32_000_000, 80_000.0),
        ):
            points.append({
                "mode": "read",
                "buffer_mib": 32,
                "window_us": 100_000,
                "pmu": {"pmu_bandwidth_rd": [3.0 * bytes_read]},
                "helper_result": {
                    "planned_bytes": bytes_read,
                    "bytes_read": bytes_read,
                    "window_alignment": {
                        "active_us": active_us,
                        "idle_tail_us": 100_000.0 - active_us,
                        "elapsed_programmed_ratio": 1.001,
                        "workload_fully_contained": True,
                    },
                },
                "telemetry": {"thermal_c": {"cpu": 40.0}},
            })
        summary = summarize(
            points,
            ["total"],
            original_timer_raw=0,
            thermal_limit_c=85.0,
            status="complete",
        )
        signal = summary["signals"]["pmu_bandwidth_rd"]
        self.assertEqual(signal["classification"], "volume")
        self.assertGreater(signal["volume_fit"]["r2"], 0.999)
        self.assertLess(signal["active_rate_fit"]["r2"], 0.9)
        self.assertEqual(summary["alignment_gate"]["rejected_points"], 0)

    def test_summary_rejects_unaligned_read_point_from_scientific_fit(self) -> None:
        point = {
            "mode": "read", "buffer_mib": 32, "window_us": 100_000,
            "pmu": {"pmu_bandwidth_rd": [123]},
            "helper_result": {
                "planned_bytes": 4096,
                "bytes_read": 4096,
                "window_alignment": {
                    "active_us": 50_000.0,
                    "idle_tail_us": 50_000.0,
                    "elapsed_programmed_ratio": 1.02,
                    "workload_fully_contained": True,
                },
            },
            "telemetry": {"thermal_c": {}},
        }
        summary = summarize(
            [point], ["total"], original_timer_raw=0,
            thermal_limit_c=85.0, status="complete",
        )
        self.assertEqual(summary["alignment_gate"]["accepted_points"], 0)
        self.assertEqual(summary["alignment_gate"]["rejected_points"], 1)
        self.assertEqual(
            summary["signals"]["pmu_bandwidth_rd"]["classification"],
            "insufficient",
        )
        self.assertIsNone(summary["signals"]["pmu_bandwidth_rd"]["read_cv"])

    def test_failure_always_emits_failure_trace_and_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_partial_failure(output, RuntimeError("boom"), run_id="test-run")
            self.assertTrue((output / "failure.json").is_file())
            partial = __import__("json").loads(
                (output / "summary.partial.json").read_text(encoding="utf-8"))
            self.assertTrue(partial["partial"])
            events = [__import__("json").loads(line)["event"] for line in
                      (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("failure", events)

    def test_setup_nan_failure_emits_raw_and_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "setup-failure"
            exit_code = main([
                "--output-dir", str(output),
                "--thermal-limit-c", "nan",
            ])
            self.assertEqual(exit_code, 4)
            failure = __import__("json").loads(
                (output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "setup_failure")
            partial = __import__("json").loads(
                (output / "summary.partial.json").read_text(encoding="utf-8"))
            self.assertTrue(partial["partial"])
            self.assertEqual(partial["status"], "setup_failure")
            events = [__import__("json").loads(line)["event"] for line in
                      (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("setup_failure", events)

    def test_setup_failure_never_appends_to_existing_raw_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent / "existing-v3"
            output.mkdir()
            old_raw = b'{"old":true}\n'
            (output / "raw.jsonl").write_bytes(old_raw)
            exit_code = main([
                "--output-dir", str(output),
                "--thermal-limit-c", "nan",
            ])
            self.assertEqual(exit_code, 4)
            self.assertEqual((output / "raw.jsonl").read_bytes(), old_raw)
            failure_dirs = list(parent.glob("existing-v3.setup-failure-*"))
            self.assertEqual(len(failure_dirs), 1)
            self.assertTrue((failure_dirs[0] / "summary.partial.json").is_file())

    def test_nonempty_requested_output_tree_is_immutable_even_with_allow_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent / "published-v4"
            output.mkdir()
            (output / "raw.jsonl").write_bytes(b'{"old":true}\n')
            (output / "manifest.json").write_bytes(b'{"status":"complete"}\n')

            def tree_hashes() -> dict[str, str]:
                return {
                    path.relative_to(output).as_posix(): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                }

            before = tree_hashes()

            def run_child(_nsi: object, child: object) -> dict[str, object]:
                return {
                    "exit_code": child(),
                    "signal": None,
                    "forwarded_signals": [],
                    "restore_verified": True,
                    "original_pmu_timer": 0,
                }

            with (
                mock.patch("tooling.nsi_calibrate.SysfsNSI", return_value=object()),
                mock.patch("tooling.nsi_calibrate.resolve_pinned_helper", return_value=Path("/pinned")),
                mock.patch("tooling.nsi_calibrate.run_under_watchdog", side_effect=run_child),
            ):
                exit_code = main([
                    "--output-dir", str(output),
                    "--allow-existing",
                ])

            self.assertEqual(tree_hashes(), before)
            self.assertEqual(exit_code, 4)
            failure_dirs = list(parent.glob("published-v4.setup-failure-*"))
            self.assertEqual(len(failure_dirs), 1)
            self.assertEqual(
                {path.name for path in failure_dirs[0].iterdir()},
                {"failure.json", "raw.jsonl", "summary.partial.json"},
            )

    def test_raw_thermal_maxima_uses_all_sample_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                '{"telemetry":{"thermal_c":{"cpu":41.0,"ddr":39.0}}}\n'
                '{"telemetry":{"thermal_c":{"cpu":47.5,"ddr":42.3}}}\n',
                encoding="utf-8")
            self.assertEqual(raw_thermal_maxima(path), {"cpu": 47.5, "ddr": 42.3})


if __name__ == "__main__":
    unittest.main()
