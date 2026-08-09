import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "tooling" / "profile_command.py"


class ProfileCommandTests(unittest.TestCase):
    def invoke(
        self,
        output_dir: Path,
        child_args: list[str],
        *,
        run_id: str | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if run_id is None:
            run_id = output_dir.name
        return subprocess.run(
            [
                sys.executable,
                str(PROFILER),
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--interval-ms",
                "20",
                "--label",
                "unit-test",
                *(extra_args or []),
                "--",
                *child_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_successful_command_retains_output_metadata_and_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-success"
            completed = self.invoke(
                output_dir,
                [
                    sys.executable,
                    "-c",
                    "import time; print('profile-stdout'); time.sleep(0.12)",
                ],
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((output_dir / "stdout.log").read_text().strip(), "profile-stdout")
            self.assertEqual((output_dir / "stderr.log").read_text(), "")

            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["run_id"], "run-success")
            self.assertEqual(metadata["label"], "unit-test")
            self.assertEqual(metadata["exit_code"], 0)
            self.assertGreater(metadata["elapsed_ns"], 0)
            self.assertGreaterEqual(metadata["telemetry_samples"], 1)
            self.assertEqual(metadata["command"][0], sys.executable)

            samples = [
                json.loads(line)
                for line in (output_dir / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertGreaterEqual(len(samples), 1)
            self.assertIn("monotonic_ns", samples[0])
            self.assertIn("system", samples[0])
            self.assertIn("process", samples[0])
            phases = [
                json.loads(line)
                for line in (output_dir / "phases.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["event"] for row in phases], ["profiler_start", "profiler_complete"])
            self.assertLessEqual(phases[0]["monotonic_ns"], phases[1]["monotonic_ns"])

    def test_failing_command_is_captured_and_status_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-failure"
            completed = self.invoke(
                output_dir,
                [
                    sys.executable,
                    "-c",
                    "import sys; print('profile-stderr', file=sys.stderr); sys.exit(7)",
                ],
            )

            self.assertEqual(completed.returncode, 7)
            self.assertEqual((output_dir / "stderr.log").read_text().strip(), "profile-stderr")
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["exit_code"], 7)
            self.assertGreaterEqual(metadata["telemetry_samples"], 1)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("keep")

            completed = self.invoke(output_dir, [sys.executable, "-c", "print('must not run')"])

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertFalse((output_dir / "metadata.json").exists())

    def test_run_id_is_required_validated_and_matches_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mismatch_dir = root / "run-name"
            mismatch = self.invoke(
                mismatch_dir,
                [sys.executable, "-c", "print('must not run')"],
                run_id="other-name",
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("run-id", mismatch.stderr)
            self.assertFalse(mismatch_dir.exists())

            for invalid in ("", "-leading", ".leading", "bad/id", "bad id"):
                invalid_dir = root / "valid-name"
                completed = self.invoke(
                    invalid_dir,
                    [sys.executable, "-c", "print('must not run')"],
                    run_id=invalid,
                )
                self.assertNotEqual(completed.returncode, 0, invalid)
                self.assertIn("run-id", completed.stderr, invalid)
                self.assertFalse(invalid_dir.exists(), invalid)

    def test_missing_executable_retains_all_artifacts_and_returns_127(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-missing"
            completed = self.invoke(output_dir, [str(output_dir / "does-not-exist")])

            self.assertEqual(completed.returncode, 127, completed.stderr)
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertTrue(metadata["launch_error"])
            self.assertEqual(metadata["exit_code"], 127)
            self.assertIsNone(metadata["child_return_code"])
            for name in ("stdout.log", "stderr.log", "telemetry.jsonl", "phases.jsonl", "metadata.json"):
                self.assertTrue((output_dir / name).exists(), name)
            self.assertEqual(metadata["phase_file"]["path"], "phases.jsonl")
            self.assertGreater(metadata["phase_file"]["size_bytes"], 0)
            phases = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in phases], ["profiler_start", "profiler_complete"])

    def test_phase_file_is_passed_to_child_and_retains_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-phases"
            completed = self.invoke(
                output_dir,
                [
                    sys.executable,
                    "-c",
                    "import os; open(os.environ['VIP9000_PHASE_FILE'], 'ab').write(b'{invalid\\n\\xff')",
                ],
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            phases = (output_dir / "phases.jsonl").read_bytes()
            self.assertIn(b'"event":"profiler_start"', phases)
            self.assertIn(b"{invalid\n\xff", phases)
            self.assertIn(b'"event":"profiler_complete"', phases)
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["phase_file"]["path"], "phases.jsonl")
            self.assertEqual(metadata["phase_file"]["size_bytes"], len(phases))

    def test_valid_child_phase_is_ordered_between_profiler_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-valid-child-phase"
            child = (
                "import json, os, time; "
                "open(os.environ['VIP9000_PHASE_FILE'], 'a').write(json.dumps({"
                "'event':'child_measured','monotonic_ns':time.monotonic_ns(),'step':1})+'\\n')"
            )

            completed = self.invoke(output_dir, [sys.executable, "-c", child])

            self.assertEqual(completed.returncode, 0, completed.stderr)
            phases = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
            self.assertEqual(
                [row["event"] for row in phases],
                ["profiler_start", "child_measured", "profiler_complete"],
            )
            self.assertEqual(
                [row["monotonic_ns"] for row in phases],
                sorted(row["monotonic_ns"] for row in phases),
            )

    def test_descendant_cannot_append_after_profiler_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-descendant-phase"
            writer = (
                "import json, os, time; time.sleep(0.4); "
                "open(os.environ['VIP9000_PHASE_FILE'], 'a').write(json.dumps({"
                "'event':'late_descendant','monotonic_ns':time.monotonic_ns(),'step':1})+'\\n')"
            )
            child = (
                "import subprocess, sys; "
                f"subprocess.Popen([sys.executable, '-c', {writer!r}])"
            )

            completed = self.invoke(output_dir, [sys.executable, "-c", child])
            size_at_return = (output_dir / "phases.jsonl").stat().st_size
            time.sleep(0.7)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((output_dir / "phases.jsonl").stat().st_size, size_at_return)
            phases = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in phases], ["profiler_start", "profiler_complete"])

    def test_phase_setup_error_retains_failure_metadata(self) -> None:
        sys.path.insert(0, str(ROOT / "tooling"))
        try:
            import profile_command
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-phase-open-error"
            original_open = Path.open

            def fail_phase_open(path: Path, *args: object, **kwargs: object):
                if path.name == "phases.jsonl":
                    raise OSError("phase storage unavailable")
                return original_open(path, *args, **kwargs)

            argv = [
                "--output-dir",
                str(output_dir),
                "--run-id",
                output_dir.name,
                "--",
                sys.executable,
                "-c",
                "print('must not run')",
            ]
            with mock.patch.object(Path, "open", fail_phase_open):
                result = profile_command.main(argv)

            self.assertNotEqual(result, 0)
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertIn("phase storage unavailable", metadata["profiler_error"])
            self.assertIsNone(metadata["child_return_code"])

    def test_relative_phase_file_stays_in_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-phases-validation"
            for phase_file in (
                "/tmp/phase.jsonl",
                "../phase.jsonl",
                "nested/../../phase.jsonl",
                "./phase.jsonl",
                "nested//phase.jsonl",
                "C:/phase.jsonl",
                "bad\nname.jsonl",
            ):
                completed = self.invoke(
                    output_dir,
                    [sys.executable, "-c", "print('must not run')"],
                    extra_args=["--phase-file", phase_file],
                )
                self.assertNotEqual(completed.returncode, 0, phase_file)
                self.assertIn("phase-file", completed.stderr, phase_file)
                self.assertFalse(output_dir.exists(), phase_file)

    def test_signal_status_is_mapped_and_raw_return_code_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-signal"
            completed = self.invoke(
                output_dir,
                [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
            )

            self.assertEqual(completed.returncode, 128 + signal.SIGTERM, completed.stderr)
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["exit_code"], 128 + signal.SIGTERM)
            self.assertEqual(metadata["child_return_code"], -signal.SIGTERM)
            self.assertEqual(metadata["signal"], signal.SIGTERM)
            phases = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in phases], ["profiler_start", "profiler_complete"])

    def test_snapshot_failure_terminates_child_and_still_writes_metadata(self) -> None:
        sys.path.insert(0, str(ROOT / "tooling"))
        try:
            import profile_command
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "run-snapshot-error"
            pid_file = root / "child.pid"
            argv = [
                "--output-dir",
                str(output_dir),
                "--run-id",
                output_dir.name,
                "--interval-ms",
                "20",
                "--",
                sys.executable,
                "-c",
                f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)",
            ]
            def fail_after_child_starts(pid: int) -> dict[str, object]:
                for _ in range(100):
                    if pid_file.exists():
                        break
                    import time

                    time.sleep(0.01)
                raise RuntimeError("sensor exploded")

            with mock.patch.object(profile_command, "snapshot", side_effect=fail_after_child_starts):
                result = profile_command.main(argv)

            self.assertNotEqual(result, 0)
            self.assertTrue((output_dir / "metadata.json").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertIn("sensor exploded", metadata["profiler_error"])
            self.assertIsNotNone(metadata["child_return_code"])
            phases = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in phases], ["profiler_start", "profiler_complete"])
            child_pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
