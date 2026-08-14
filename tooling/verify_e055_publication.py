#!/usr/bin/env python3
"""Verify E055's E047-style staged-tree binding after commit and push."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from tooling.branch_preflight import tree_binding_sha256
except ModuleNotFoundError as exc:
    if exc.name != "tooling":
        raise
    from branch_preflight import tree_binding_sha256


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_publication(
    root: Path,
    preflight_path: Path,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        preflight: Mapping[str, Any] = json.loads(preflight_path.read_text(encoding="utf-8"))
        manifest: Mapping[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read publication metadata: {exc}"]
    if manifest.get("schema") != "e055-q1-hot-cold-manifest/v2":
        errors.append("manifest schema must be e055-q1-hot-cold-manifest/v2")
    active = preflight.get("active_worktree")
    binding = manifest.get("publication_binding")
    if not isinstance(active, Mapping) or not isinstance(binding, Mapping):
        return errors + ["preflight active_worktree and manifest publication_binding are required"]
    expected = {
        "preflight_base_commit": active.get("base_commit"),
        "staged_tree_binding_sha256": active.get("tree_binding_sha256"),
        "active_ref": active.get("ref"),
        "upstream_ref": active.get("upstream_ref"),
        "binding_excludes": active.get("binding_excludes"),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            errors.append(f"manifest {key} does not match preflight")
    if binding.get("mode") != "e047-staged-index-plus-post-commit-ref-verification":
        errors.append("manifest publication mode is not the accepted E047 pattern")
    head = _git(root, "rev-parse", "HEAD")
    base = active.get("base_commit")
    if not isinstance(head, str) or len(head) != 40:
        errors.append("cannot resolve exact HEAD commit")
        return errors
    if head == base:
        errors.append("publication is not yet in a commit after the preflight base")
    elif _git(root, "merge-base", "--is-ancestor", str(base), head) is None:
        errors.append("preflight base is not an ancestor of publication HEAD")
    excludes = active.get("binding_excludes")
    declared_tree = active.get("tree_binding_sha256")
    if not isinstance(excludes, list) or tree_binding_sha256(root, excludes, source="HEAD") != declared_tree:
        errors.append("committed HEAD tree does not match staged preflight binding")
    active_ref = active.get("ref")
    upstream_ref = active.get("upstream_ref")
    if not isinstance(active_ref, str) or _git(root, "rev-parse", active_ref) != head:
        errors.append("active local ref does not equal publication HEAD")
    if not isinstance(upstream_ref, str) or _git(root, "rev-parse", upstream_ref) != head:
        errors.append("upstream remote-tracking ref does not equal publication HEAD")
    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest files must be a list")
    else:
        for item in files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                errors.append("manifest file entry is malformed")
                continue
            path = root / item["path"]
            if not path.is_file():
                errors.append(f"manifest file is missing: {item['path']}")
            elif _sha256(path) != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
                errors.append(f"manifest file binding mismatch: {item['path']}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/E055-q1-hot-cold/data"
    errors = verify_publication(
        root, experiment / "branch-preflight.json", experiment / "manifest.json"
    )
    print(json.dumps({"schema": "e055-publication-verification/v1",
                      "status": "PASS" if not errors else "FAIL",
                      "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
