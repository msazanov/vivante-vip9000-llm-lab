"""Reservation-only scaffold for future E055 board captures.

This module creates exclusive output slots.  It deliberately has no process
execution capability and cannot start a target measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    build_name: str


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


def _reserve(path: Path) -> int:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        os.close(descriptor)
        raise OSError(f"reserved path is not one regular file: {path}")
    return descriptor


def reserve_phase(
    phase_dir: Path, planned_runs: Sequence[RunPlan],
) -> tuple[Path, ...]:
    """Exclusively reserve every raw role without starting any measurement."""

    plans = _validate_plans(planned_runs)
    phase = Path(phase_dir)
    if RUN_ID_RE.fullmatch(phase.name) is None:
        raise ValueError("phase directory name must be canonical and bounded")
    parent_status = phase.parent.stat()
    if not stat.S_ISDIR(parent_status.st_mode):
        raise ValueError("phase parent must be an existing directory")

    os.mkdir(phase, 0o700)
    artifact_dir = phase / "artifacts"
    runs_dir = phase / "runs"
    os.mkdir(artifact_dir, 0o700)
    os.mkdir(runs_dir, 0o700)

    paths: list[Path] = [phase / "bundle.json"]
    for build_name in sorted({plan.build_name for plan in plans}):
        paths.append(artifact_dir / f"harness-{build_name}.bin")
    for plan in plans:
        run_dir = runs_dir / plan.run_id
        os.mkdir(run_dir, 0o700)
        paths.extend(run_dir / name for name in RUN_FILENAMES)

    descriptors: list[int] = []
    try:
        for path in paths:
            descriptors.append(_reserve(path))
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.plan_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("plan JSON must be a list")
    plans = tuple(RunPlan(**item) for item in raw)
    paths = reserve_phase(arguments.phase_dir, plans)
    print(json.dumps({
        "schema": "e055-capture-reservation/v1",
        "target_workload_executed": False,
        "reserved_paths": [path.as_posix() for path in paths],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
