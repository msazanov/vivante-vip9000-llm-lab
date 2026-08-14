#!/usr/bin/env python3
"""Verify E055's E047-style staged-tree binding after commit and push."""

from __future__ import annotations

import hashlib
import json
import stat
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


def _metadata_matches_head(root: Path, path: Path) -> bool:
    """Require one regular metadata file identical in worktree, index, and HEAD."""

    try:
        relative = path.relative_to(root).as_posix()
        status = path.lstat()
    except (OSError, ValueError):
        return False
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        return False
    head = _git(root, "ls-tree", "HEAD", "--", relative)
    index = _git(root, "ls-files", "--stage", "--", relative)
    worktree_oid = _git(root, "hash-object", "--", relative)
    if not head or not index or not worktree_oid:
        return False
    head_fields = head.partition("\t")[0].split()
    index_fields = index.partition("\t")[0].split()
    return (
        len(head_fields) == 3 and head_fields[0] in ("100644", "100755")
        and head_fields[1] == "blob" and len(index_fields) == 3
        and index_fields[0] == head_fields[0] and index_fields[1] == head_fields[2]
        and index_fields[2] == "0" and worktree_oid == head_fields[2]
    )


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
    if not _metadata_matches_head(root, manifest_path):
        errors.append("manifest worktree/index bytes do not match committed HEAD")
    if not _metadata_matches_head(root, preflight_path):
        errors.append("preflight worktree/index bytes do not match committed HEAD")
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
        seen_paths: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                errors.append("manifest file entry is malformed")
                continue
            relative = item["path"]
            parsed = Path(relative)
            if parsed.is_absolute() or parsed.as_posix() != relative or \
                    any(part in ("", ".", "..") for part in parsed.parts) or \
                    relative in seen_paths:
                errors.append(f"manifest file path is noncanonical or duplicated: {relative}")
                continue
            seen_paths.add(relative)
            path = root / relative
            if not path.is_file():
                errors.append(f"manifest file is missing: {relative}")
            elif _sha256(path) != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
                errors.append(f"manifest file binding mismatch: {relative}")
            elif not _metadata_matches_head(root, path):
                errors.append(f"manifested file worktree/index mismatch: {relative}")
    return errors


def verify_global_publication(
    root: Path,
    preflight_path: Path,
    manifest_path: Path,
) -> list[str]:
    """Verify publication bindings plus a globally clean index/worktree."""

    errors = verify_publication(root, preflight_path, manifest_path)
    status = subprocess.run(
        ["git", "status", "--porcelain=v2", "--untracked-files=all"],
        cwd=root, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if status.returncode != 0:
        errors.append(f"cannot verify globally clean worktree: {status.stderr.strip()}")
    elif status.stdout:
        errors.append("publication requires a globally clean index/worktree")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/E055-q1-hot-cold/data"
    errors = verify_global_publication(
        root, experiment / "branch-preflight.json", experiment / "manifest.json"
    )
    print(json.dumps({"schema": "e055-publication-verification/v1",
                      "status": "PASS" if not errors else "FAIL",
                      "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
