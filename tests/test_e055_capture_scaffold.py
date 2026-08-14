"""Safety gates for the reservation-only E055 capture scaffold."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
import tempfile
import unittest

import tooling.e055_capture_scaffold as capture_scaffold
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

    def test_symlink_parent_is_refused_before_any_reservation(self) -> None:
        target = self.root / "real-parent"
        target.mkdir()
        alias = self.root / "parent-alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "parent"):
            reserve_phase(alias / "phase-a", self.plans)
        self.assertFalse((target / "phase-a").exists())

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
        sentinel = self.root / "preexisting-sentinel"
        sentinel.write_text("keep", encoding="utf-8")

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
        self.assertFalse(self.phase.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_unsafe_post_open_validation_rolls_back_created_phase(self) -> None:
        real_fstat = os.fstat
        injected = False

        def unsafe_first_fstat(descriptor: int) -> os.stat_result:
            nonlocal injected
            status = real_fstat(descriptor)
            if injected:
                return status
            injected = True
            fields = list(status)
            fields[3] = 2
            return os.stat_result(fields)

        sentinel = self.root / "validation-sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with mock.patch(
            "tooling.e055_capture_scaffold.os.fstat", side_effect=unsafe_first_fstat
        ):
            with self.assertRaisesRegex(OSError, "regular file"):
                reserve_phase(self.phase, self.plans)
        self.assertFalse(self.phase.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_fstat_exception_after_open_rolls_back_created_phase(self) -> None:
        sentinel = self.root / "fstat-sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with mock.patch(
            "tooling.e055_capture_scaffold.os.fstat",
            side_effect=OSError("injected fstat failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected fstat failure"):
                reserve_phase(self.phase, self.plans)
        self.assertFalse(self.phase.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_new_path_is_rolled_back_if_an_external_hardlink_appears(self) -> None:
        real_fstat = os.fstat
        external_link = self.root / "external-link"
        injected = False

        def hardlink_before_validation(descriptor: int) -> os.stat_result:
            nonlocal injected
            if not injected:
                injected = True
                os.link(self.phase / "bundle.json", external_link)
            return real_fstat(descriptor)

        with mock.patch(
            "tooling.e055_capture_scaffold.os.fstat",
            side_effect=hardlink_before_validation,
        ):
            with self.assertRaisesRegex(OSError, "regular file"):
                reserve_phase(self.phase, self.plans)
        self.assertFalse(self.phase.exists())
        self.assertTrue(external_link.is_file())

    def test_rollback_preserves_replacement_phase_and_moved_original(self) -> None:
        moved_original = self.root / "moved-original-phase"
        real_open = os.open
        injected = False

        def replace_phase_before_failure(
            path: os.PathLike[str] | str, flags: int, mode: int = 0o777,
        ) -> int:
            nonlocal injected
            if not injected:
                injected = True
                self.phase.rename(moved_original)
                self.phase.mkdir(mode=0o700)
                raise OSError("injected after phase replacement")
            return real_open(path, flags, mode)

        with mock.patch(
            "tooling.e055_capture_scaffold.os.open",
            side_effect=replace_phase_before_failure,
        ):
            with self.assertRaisesRegex(OSError, "phase replacement"):
                reserve_phase(self.phase, self.plans)
        self.assertTrue(self.phase.is_dir(), "replacement phase is not invocation-owned")
        self.assertTrue(moved_original.is_dir(), "moved original must not be path-deleted")
        self.assertTrue((moved_original / "runs/run-a").is_dir())

    def test_rollback_preserves_replacement_run_directory_and_moved_original(self) -> None:
        original_run = self.phase / "runs/run-a"
        moved_original = self.root / "moved-original-run-a"
        real_open = os.open
        injected = False

        def replace_run_before_failure(
            path: os.PathLike[str] | str, flags: int, mode: int = 0o777,
        ) -> int:
            nonlocal injected
            if not injected:
                injected = True
                original_run.rename(moved_original)
                original_run.mkdir(mode=0o700)
                raise OSError("injected after run replacement")
            return real_open(path, flags, mode)

        with mock.patch(
            "tooling.e055_capture_scaffold.os.open",
            side_effect=replace_run_before_failure,
        ):
            with self.assertRaisesRegex(OSError, "run replacement"):
                reserve_phase(self.phase, self.plans)
        self.assertTrue(original_run.is_dir(), "replacement run is not invocation-owned")
        self.assertTrue(moved_original.is_dir(), "moved original must remain intact")

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

    def test_reserved_outputs_are_populated_once_through_inode_bound_handle(self) -> None:
        populate = getattr(capture_scaffold, "populate_reserved", None)
        self.assertTrue(
            callable(populate),
            "O_EXCL placeholders need an inode-bound one-shot population primitive",
        )
        reservations = reserve_phase(self.phase, self.plans)
        reservation = next(
            item for item in reservations
            if getattr(item, "path", None) == self.phase / "bundle.json"
        )
        payload = b'{"schema":"e055-raw-bundle/v2"}\n'
        result = populate(reservation, payload)
        self.assertEqual((self.phase / "bundle.json").read_bytes(), payload)
        self.assertEqual(result.size_bytes, len(payload))
        with self.assertRaisesRegex(FileExistsError, "already populated"):
            populate(reservation, payload)

    def test_population_rejects_replaced_placeholder_inode(self) -> None:
        populate = getattr(capture_scaffold, "populate_reserved", None)
        self.assertTrue(callable(populate))
        reservations = reserve_phase(self.phase, self.plans)
        reservation = next(
            item for item in reservations
            if getattr(item, "path", None) == self.phase / "bundle.json"
        )
        reservation.path.unlink()
        reservation.path.write_bytes(b"")
        with self.assertRaisesRegex(OSError, "identity"):
            populate(reservation, b"replacement must fail\n")


if __name__ == "__main__":
    unittest.main()
