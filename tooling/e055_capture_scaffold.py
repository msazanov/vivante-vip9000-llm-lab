"""Reservation-only scaffold for future E055 board captures.

This module creates exclusive output slots.  It deliberately has no process
execution capability and cannot start a target measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Sequence


RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
BUILD_NAMES = frozenset(("O3", "O3-flto"))
RUN_FILENAMES = (
    "harness.stdout.capture.json",
    "harness.stderr.capture.json",
    "e049c.json",
    "e049c.stderr.capture.json",
    "runner.json",
)
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 32 * 1024 * 1024
RUN_FILE_LIMITS = {
    "harness.stdout.capture.json": 256 * 1024,
    "harness.stderr.capture.json": 1024 * 1024,
    "e049c.json": 256 * 1024,
    "e049c.stderr.capture.json": 1024 * 1024,
    "runner.json": 256 * 1024,
}


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    build_name: str


@dataclass(frozen=True)
class ReservedOutput:
    """One exclusive empty placeholder bound to its original inode."""

    path: Path
    device: int
    inode: int
    maximum_size_bytes: int

    def is_file(self) -> bool:
        return self.path.is_file()

    def stat(self) -> os.stat_result:
        return self.path.stat()

    def as_posix(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class PopulationResult:
    """Observable result of one completed placeholder population."""

    path: Path
    size_bytes: int
    sha256: str


def safe_environment(build_name: str) -> dict[str, str]:
    """Return the only non-sensitive environment accepted by E055."""

    if build_name not in BUILD_NAMES:
        raise ValueError("build_name must be O3 or O3-flto")
    return {"LC_ALL": "C", "LANG": "C", "E055_BUILD_NAME": build_name}


def _validate_plans(planned_runs: Sequence[RunPlan]) -> tuple[RunPlan, ...]:
    plans = tuple(planned_runs)
    if not plans:
        raise ValueError("at least one run plan is required")
    run_ids: set[str] = set()
    for plan in plans:
        if not isinstance(plan, RunPlan) or RUN_ID_RE.fullmatch(plan.run_id) is None:
            raise ValueError("run_id must be canonical and bounded")
        safe_environment(plan.build_name)
        if plan.run_id in run_ids:
            raise ValueError("run_id must be unique")
        run_ids.add(plan.run_id)
    return plans


def _reserve(path: Path, maximum_size_bytes: int) -> tuple[int, ReservedOutput]:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
    except BaseException:
        try:
            # Keep cleanup identity-bound even when the validation call itself
            # fails. ``os.stat(fd)`` is an independent Python entry point to
            # the already-open inode; never unlink from the pathname alone.
            opened = os.stat(descriptor)
            current = path.lstat()
            if stat.S_ISREG(current.st_mode) and \
                    (current.st_dev, current.st_ino) == (
                        opened.st_dev, opened.st_ino
                    ):
                path.unlink()
        except (FileNotFoundError, OSError):
            pass
        finally:
            os.close(descriptor)
        raise
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        try:
            current = path.lstat()
            if stat.S_ISREG(current.st_mode) and \
                    (current.st_dev, current.st_ino) == (status.st_dev, status.st_ino):
                path.unlink()
        finally:
            os.close(descriptor)
        raise OSError(f"reserved path is not one regular file: {path}")
    return descriptor, ReservedOutput(
        path=path,
        device=status.st_dev,
        inode=status.st_ino,
        maximum_size_bytes=maximum_size_bytes,
    )


def _rollback_reservations(
    reservations: Sequence[ReservedOutput], directories: Sequence[Path],
) -> None:
    """Remove only inodes and empty directories created by this invocation."""

    for reservation in reversed(tuple(reservations)):
        try:
            current = reservation.path.lstat()
            if stat.S_ISREG(current.st_mode) and \
                    (current.st_dev, current.st_ino) == (
                        reservation.device, reservation.inode
                    ):
                reservation.path.unlink()
        except FileNotFoundError:
            pass
    for directory in reversed(tuple(directories)):
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # A replaced or concurrently populated path is not ours to remove.
            pass


def populate_reserved(reservation: ReservedOutput, payload: bytes) -> PopulationResult:
    """Populate one original placeholder exactly once without following links.

    Producers that insist on creating their own output with ``O_EXCL`` cannot
    receive the placeholder path directly. A future runner must capture such
    output separately and pass the exact resulting bytes through this function.
    """

    if not isinstance(reservation, ReservedOutput):
        raise TypeError("reservation must be an E055 ReservedOutput")
    if not isinstance(payload, bytes) or not payload or \
            len(payload) > reservation.maximum_size_bytes:
        raise ValueError("population payload must be nonempty and within its role bound")
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(reservation.path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = os.fstat(descriptor)
        identity = (reservation.device, reservation.inode)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or \
                (before.st_dev, before.st_ino) != identity:
            raise OSError("reserved output identity changed before population")
        if before.st_size != 0:
            raise FileExistsError("reserved output is already populated")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("reserved output population made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != identity or \
                after.st_size != len(payload) or after.st_nlink != 1:
            raise OSError("reserved output identity or size changed during population")
        path_status = reservation.path.lstat()
        if (path_status.st_dev, path_status.st_ino) != identity or \
                not stat.S_ISREG(path_status.st_mode) or path_status.st_nlink != 1:
            raise OSError("reserved output identity changed after population")
    finally:
        os.close(descriptor)
    return PopulationResult(
        path=reservation.path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def reserve_phase(
    phase_dir: Path, planned_runs: Sequence[RunPlan],
) -> tuple[ReservedOutput, ...]:
    """Exclusively reserve every raw role without starting any measurement."""

    plans = _validate_plans(planned_runs)
    phase = Path(phase_dir)
    if RUN_ID_RE.fullmatch(phase.name) is None:
        raise ValueError("phase directory name must be canonical and bounded")
    parent_status = phase.parent.lstat()
    if not stat.S_ISDIR(parent_status.st_mode) or \
            phase.parent.resolve(strict=True) != phase.parent.absolute():
        raise ValueError("phase parent must be one existing non-symlink directory")

    directories: list[Path] = []
    descriptors: list[int] = []
    reservations: list[ReservedOutput] = []
    try:
        os.mkdir(phase, 0o700)
        directories.append(phase)
        artifact_dir = phase / "artifacts"
        runs_dir = phase / "runs"
        os.mkdir(artifact_dir, 0o700)
        directories.append(artifact_dir)
        os.mkdir(runs_dir, 0o700)
        directories.append(runs_dir)

        outputs: list[tuple[Path, int]] = [(phase / "bundle.json", MAX_BUNDLE_BYTES)]
        for build_name in sorted({plan.build_name for plan in plans}):
            outputs.append((
                artifact_dir / f"harness-{build_name}.bin", MAX_EXECUTABLE_BYTES,
            ))
        for plan in plans:
            run_dir = runs_dir / plan.run_id
            os.mkdir(run_dir, 0o700)
            directories.append(run_dir)
            outputs.extend(
                (run_dir / name, RUN_FILE_LIMITS[name]) for name in RUN_FILENAMES
            )
        for path, maximum_size_bytes in outputs:
            descriptor, reservation = _reserve(path, maximum_size_bytes)
            descriptors.append(descriptor)
            reservations.append(reservation)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        _rollback_reservations(reservations, directories)
        raise
    for descriptor in descriptors:
        os.close(descriptor)
    return tuple(reservations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.plan_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("plan JSON must be a list")
    plans = tuple(RunPlan(**item) for item in raw)
    reservations = reserve_phase(arguments.phase_dir, plans)
    print(json.dumps({
        "schema": "e055-capture-reservation/v1",
        "target_workload_executed": False,
        "reserved_paths": [item.path.as_posix() for item in reservations],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
