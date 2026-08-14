#!/usr/bin/env python3
"""Generate the source/raw manifest for the E055 Stage 1 publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/E055-q1-hot-cold"
OUTPUT = EXPERIMENT / "data/manifest.json"

PUBLISHED = (
    ROOT / "experiments/E055-q1-hot-cold/README.md",
    ROOT / "experiments/E055-q1-hot-cold/data/branch-preflight.json",
    ROOT / "experiments/E055-q1-hot-cold/data/failure-qemu-uninitialized-lut.txt",
    ROOT / "experiments/E055-q1-hot-cold/data/sample.schema.json",
    ROOT / "tooling/e055_q1_hotcold.cpp",
    ROOT / "tooling/e055_q1_hotcold.py",
    ROOT / "tooling/generate_e055_manifest.py",
    ROOT / "tests/test_e055_harness_contract.py",
    ROOT / "tests/test_e055_q1_hotcold.py",
    ROOT / "experiments/E039-q1-pair-wholek/e039_wholek_harness.cpp",
    ROOT / "experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in PUBLISHED if not path.is_file()]
    if missing:
        raise SystemExit("missing publication files: " + ", ".join(missing))
    files = []
    for path in PUBLISHED:
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "stock-reference" if "E039" in path.as_posix() else "e055-stage1",
            }
        )
    payload = {
        "schema": "e055-q1-hot-cold-manifest/v1",
        "experiment": "E055-Q1-HOT-COLD",
        "status": "stage1_source_only_no_target_workload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit_at_generation": git_commit(),
        "model_payload_included": False,
        "target_workload_executed": False,
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
