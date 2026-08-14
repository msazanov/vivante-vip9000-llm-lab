#!/usr/bin/env python3
"""Generate the source/raw manifest for the E055 Stage 1 publication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from tooling.e055_q1_hotcold import (
        HARNESS_SCHEMA,
        PMU_GROUP_CONFIGS,
        UPSTREAM_COMMIT,
        UPSTREAM_REF,
        UPSTREAM_REPACK_SHA256,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tooling":
        raise
    from e055_q1_hotcold import (
        HARNESS_SCHEMA,
        PMU_GROUP_CONFIGS,
        UPSTREAM_COMMIT,
        UPSTREAM_REF,
        UPSTREAM_REPACK_SHA256,
    )


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
    ROOT / "experiments/E055-q1-hot-cold/data/review-rejection-stage3-0ed991b.md",
    ROOT / "experiments/E055-q1-hot-cold/data/review-rejection-stage4-7679828.md",
    ROOT / "experiments/E055-q1-hot-cold/data/disassembly-review.json",
    ROOT / "experiments/E055-q1-hot-cold/data/upstream-source-binding.json",
    ROOT / "experiments/E055-q1-hot-cold/data/sample.schema.json",
    ROOT / "experiments/E055-q1-hot-cold/data/raw-bundle.schema.json",
    ROOT / "experiments/E055-q1-hot-cold/data/runner-capture.schema.json",
    ROOT / "experiments/E055-q1-hot-cold/data/stream-capture.schema.json",
    ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64",
    ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-flto-aarch64",
    ROOT / "tooling/e055_q1_hotcold.cpp",
    ROOT / "tooling/e055_q1_hotcold.py",
    ROOT / "tooling/e055_raw_bundle.py",
    ROOT / "tooling/e055_capture_scaffold.py",
    ROOT / "tooling/a733_pmu_exec.c",
    ROOT / "tooling/check_e055_disassembly.py",
    ROOT / "tooling/verify_e055_publication.py",
    ROOT / "tooling/generate_e055_manifest.py",
    ROOT / "tests/test_e055_harness_contract.py",
    ROOT / "tests/test_e055_q1_hotcold.py",
    ROOT / "tests/test_e055_raw_bundle.py",
    ROOT / "tests/test_e055_sealed_promotion.py",
    ROOT / "tests/test_e055_capture_scaffold.py",
    ROOT / "tests/test_e055_aarch64_gate.py",
    ROOT / "tests/test_e055_publication.py",
    ROOT / "experiments/E039-q1-pair-wholek/e039_wholek_harness.cpp",
    ROOT / "experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S",
    ROOT / "docs/superpowers/specs/2026-08-14-e055-git-sealed-raw-bundles-design.md",
    ROOT / "docs/superpowers/plans/2026-08-14-e055-git-sealed-raw-bundles.md",
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
    disassembly = json.loads(
        (experiment / "data/disassembly-review.json").read_text(encoding="utf-8")
    )
    upstream = json.loads(
        (experiment / "data/upstream-source-binding.json").read_text(encoding="utf-8")
    )
    if disassembly.get("status") != "PASS" or not disassembly.get("builds") or any(
        item.get("golden_pass") is not True or item.get("golden_cases") != 18
        for item in disassembly.get("builds", []) if isinstance(item, dict)
    ):
        raise ValueError("every allowed runtime build must pass exactly 18 golden cases")
    disassembly_provenance = disassembly["provenance"]
    allowed_builds = {
        item["name"]: item["binary_sha256"] for item in disassembly["builds"]
    }
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
            "repository": upstream["repository"],
            "commit": upstream["commit"],
            "immutable_ref": upstream["immutable_ref"],
            "repack_cpp_sha256": UPSTREAM_REPACK_SHA256,
            "payload_publication": "local-only source; hash published",
        },
        "runtime_qualification": {
            "schema": "e055-runtime-qualification/v1",
            "harness_schema": HARNESS_SCHEMA,
            "golden_cases": 18,
            "source_sha256": disassembly_provenance["source_sha256"],
            "compiler_sha256": disassembly_provenance["compiler_sha256"],
            "compiler_id": disassembly_provenance["compiler_id"],
            "allowed_builds": allowed_builds,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_ref": UPSTREAM_REF,
            "upstream_repack_sha256": UPSTREAM_REPACK_SHA256,
            "pmu_group_configs": PMU_GROUP_CONFIGS,
            "pmu_source_sha256": sha256(root / "tooling/a733_pmu_exec.c"),
        },
        "sample_provenance_contract": (
            "infer_bottleneck accepts only a committed raw-bundle manifest path. "
            "It derives rows from Git-sealed artifacts and publication-bound source, "
            "binary, compiler, PMU config, and upstream identities; caller-created "
            "sample mappings are never measurement evidence."
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
