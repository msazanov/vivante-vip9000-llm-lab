"""Safety gates for the reservation-only E055 capture scaffold."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
import tempfile
import unittest

from tooling.e055_capture_scaffold import RunPlan, reserve_phase, safe_environment


class E055CaptureScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e055-capture-")
        self.root = Path(self.temp.name)
        self.phase = self.root / "phase-a"
        self.plans = (
            RunPlan("run-a", "O3"),
            RunPlan("run-b", "O3-flto"),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reserves_every_file_once_without_executing_a_workload(self) -> None:
        paths = reserve_phase(self.phase, self.plans)
        self.assertEqual(len(paths), 13)
        self.assertTrue(all(path.is_file() and path.stat().st_nlink == 1 for path in paths))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertFalse((self.phase / "target-workload-executed").exists())

    def test_existing_directory_regular_file_or_symlink_is_refused(self) -> None:
        for collision in ("directory", "regular", "symlink"):
            with self.subTest(collision=collision):
                phase = self.root / f"phase-{collision}"
                if collision == "directory":
                    phase.mkdir()
                elif collision == "regular":
                    phase.write_bytes(b"collision")
                else:
                    phase.symlink_to(self.root / "missing-target")
                with self.assertRaises(FileExistsError):
                    reserve_phase(phase, self.plans)

    def test_invalid_or_duplicate_run_and_build_names_fail_before_reservation(self) -> None:
        cases = (
            (RunPlan("../escape", "O3"),),
            (RunPlan("run-a", "forged"),),
            (RunPlan("run-a", "O3"), RunPlan("run-a", "O3")),
            (),
        )
        for index, plans in enumerate(cases):
            with self.subTest(index=index):
                phase = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    reserve_phase(phase, plans)
                self.assertFalse(phase.exists())

    def test_partial_open_failure_closes_every_reserved_descriptor(self) -> None:
        real_open = os.open
        opened: list[int] = []

        def failing_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
            if len(opened) == 3:
                raise OSError("injected reservation failure")
            descriptor = real_open(path, flags, mode)
            opened.append(descriptor)
            return descriptor

        with mock.patch("tooling.e055_capture_scaffold.os.open", side_effect=failing_open):
            with self.assertRaisesRegex(OSError, "injected"):
                reserve_phase(self.phase, self.plans)
        for descriptor in opened:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_environment_is_exact_and_non_secret(self) -> None:
        self.assertEqual(
            safe_environment("O3"),
            {"LC_ALL": "C", "LANG": "C", "E055_BUILD_NAME": "O3"},
        )
        source = (Path(__file__).parents[1] / "tooling/e055_capture_scaffold.py")
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("subprocess", text)
        self.assertNotIn("ssh", text.lower())
        self.assertNotIn("password", text.lower())


if __name__ == "__main__":
    unittest.main()
