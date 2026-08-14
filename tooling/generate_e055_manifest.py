#!/usr/bin/env python3
"""Generate the source/raw manifest for the E055 Stage 1 publication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from tooling.e055_q1_hotcold import UPSTREAM_REPACK_SHA256
except ModuleNotFoundError as exc:
    if exc.name != "tooling":
        raise
    from e055_q1_hotcold import UPSTREAM_REPACK_SHA256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/E055-q1-hot-cold"
OUTPUT = EXPERIMENT / "data/manifest.json"
PREFLIGHT = EXPERIMENT / "data/branch-preflight.json"

PUBLISHED = (
    ROOT / "experiments/E055-q1-hot-cold/README.md",
    ROOT / "experiments/E055-q1-hot-cold/data/branch-preflight.json",
    ROOT / "experiments/E055-q1-hot-cold/data/failure-qemu-uninitialized-lut.txt",
    ROOT / "experiments/E055-q1-hot-cold/data/review-rejection-stage1-51d1c1c.md",
    ROOT / "experiments/E055-q1-hot-cold/data/review-rejection-stage2-bba62cb.md",
    ROOT / "experiments/E055-q1-hot-cold/data/disassembly-review.json",
    ROOT / "experiments/E055-q1-hot-cold/data/upstream-source-binding.json",
    ROOT / "experiments/E055-q1-hot-cold/data/sample.schema.json",
    ROOT / "tooling/e055_q1_hotcold.cpp",
    ROOT / "tooling/e055_q1_hotcold.py",
    ROOT / "tooling/check_e055_disassembly.py",
    ROOT / "tooling/verify_e055_publication.py",
    ROOT / "tooling/generate_e055_manifest.py",
    ROOT / "tests/test_e055_harness_contract.py",
    ROOT / "tests/test_e055_q1_hotcold.py",
    ROOT / "tests/test_e055_aarch64_gate.py",
    ROOT / "tests/test_e055_publication.py",
    ROOT / "experiments/E039-q1-pair-wholek/e039_wholek_harness.cpp",
    ROOT / "experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(
    *,
    root: Path = ROOT,
    experiment: Path = EXPERIMENT,
    published: tuple[Path, ...] = PUBLISHED,
    preflight_path: Path = PREFLIGHT,
) -> dict:
    """Bind files to the staged tree; the containing commit is verified later.

    A commit cannot contain its own SHA. E047 therefore binds every non-self-
    referential file through the stage-0 index, then verifies after commit and
    push that HEAD, the local branch, and its remote-tracking ref are identical.
    """

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    active = preflight["active_worktree"]
    files = []
    for path in published:
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "stock-reference" if "E039" in path.as_posix() else "e055-stage1",
        })
    return {
        "schema": "e055-q1-hot-cold-manifest/v2",
        "experiment": "E055-Q1-HOT-COLD",
        "status": "revised_stage1_source_only_no_target_workload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_binding": {
            "mode": "e047-staged-index-plus-post-commit-ref-verification",
            "preflight_base_commit": active["base_commit"],
            "staged_tree_binding_sha256": active["tree_binding_sha256"],
            "active_ref": active["ref"],
            "upstream_ref": active["upstream_ref"],
            "binding_excludes": active["binding_excludes"],
            "post_commit_verifier": "python3 tooling/verify_e055_publication.py",
        },
        "upstream_source_binding": {
            "path": "ggml/src/ggml-cpu/arch/arm/repack.cpp",
            "repack_cpp_sha256": UPSTREAM_REPACK_SHA256,
            "payload_publication": "local-only source; hash published",
        },
        "sample_provenance_contract": (
            "Every target series must declare actual source, binary, compiler, and "
            "upstream repack hashes; validate_sample requires their exact binding."
        ),
        "model_payload_included": False,
        "target_workload_executed": False,
        "files": files,
    }


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in PUBLISHED if not path.is_file()]
    if missing:
        raise SystemExit("missing publication files: " + ", ".join(missing))
    payload = build_payload()
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
