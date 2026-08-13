#!/usr/bin/env python3
"""Search every locally available branch/ref before starting a hypothesis.

The lab intentionally keeps this tool read-only.  It searches the remote-tracking
refs already present in a clone; fetching is a separate, explicit operation.
The JSON manifest is suitable for committing next to an experiment so a later
reader can see which refs and terms were checked.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "e053-branch-preflight/v3"
DEFAULT_TERMS = ("LFM2.5", "LFM", "E047", "E053")
EXPERIMENT_RE = re.compile(
    r"(?<![A-Za-z0-9])E\d{3,}(?:-[A-Za-z0-9][A-Za-z0-9-]*)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
MAX_LINE_CHARS = 400


def _experiment_ids(value: str) -> list[str]:
    identifiers: set[str] = set()
    for match in EXPERIMENT_RE.findall(value):
        full = match.upper()
        identifiers.add(full)
        base = re.match(r"^E\d{3,}", full)
        if base:
            identifiers.add(base.group(0))
    return sorted(identifiers)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _match_lines(path: str, data: bytes, terms: Iterable[str]) -> list[dict[str, object]]:
    if _is_binary(data):
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    term_patterns = [
        (
            term,
            re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            ),
        )
        for term in terms
    ]
    matches: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for term, pattern in term_patterns:
            if pattern.search(line):
                matches.append(
                    {
                        "term": term,
                        "path": path,
                        "line": line_number,
                        "text": line[:MAX_LINE_CHARS],
                        "experiment_ids": _experiment_ids(f"{path} {line}"),
                    }
                )
    return matches


def _filesystem_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def _read_ref_files(root: Path, ref: str) -> list[tuple[str, bytes]]:
    try:
        names = _git(root, "ls-tree", "-r", "--name-only", ref).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [(str(path.relative_to(root)), path.read_bytes()) for path in _filesystem_files(root)]
    files: list[tuple[str, bytes]] = []
    for name in names:
        try:
            data = subprocess.run(
                ["git", "-C", str(root), "show", f"{ref}:{name}"],
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError:
            continue
        files.append((name, data))
    return files


def scan_ref(root: Path, ref: str, terms: Iterable[str]) -> dict[str, object]:
    """Return text matches for one git ref, or the working tree if no git exists."""

    terms = tuple(dict.fromkeys(term for term in terms if term))
    matches: list[dict[str, object]] = []
    experiment_paths: list[dict[str, object]] = []
    for path, data in _read_ref_files(root, ref):
        path_experiment_ids = _experiment_ids(path)
        if path_experiment_ids:
            experiment_paths.append({"path": path, "experiment_ids": path_experiment_ids})
        matches.extend(_match_lines(path, data, terms))
    experiment_ids = sorted(
        {
            item
            for match in matches
            for item in _experiment_ids(f"{match['path']} {match['text']}")
        }
    )
    grouped_roots: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for item in experiment_paths:
        path = Path(str(item["path"]))
        if len(path.parts) < 2 or path.parts[0].lower() != "experiments":
            continue
        canonical = Path(path.parts[0], path.parts[1]).as_posix()
        key = (canonical, tuple(str(value) for value in item["experiment_ids"]))
        grouped_roots.setdefault(key, []).append(str(item["path"]))
    experiment_roots = [
        {
            "canonical_path": canonical,
            "experiment_ids": list(experiment_ids_value),
            "paths": sorted(paths),
        }
        for (canonical, experiment_ids_value), paths in sorted(grouped_roots.items())
    ]
    return {
        "ref": ref,
        "match_count": len(matches),
        "matches": matches,
        "experiment_ids": experiment_ids,
        "experiment_paths": experiment_paths,
        "experiment_roots": experiment_roots,
    }


def list_refs(root: Path) -> list[str]:
    try:
        output = _git(root, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["HEAD"]
    refs = [
        ref
        for ref in output.splitlines()
        if ref and not ref.endswith("/HEAD")
    ]
    return refs or ["HEAD"]


def list_remote_refs(root: Path) -> list[str]:
    return sorted(ref for ref in list_refs(root) if ref.startswith("refs/remotes/"))


def _commit(root: Path, ref: str) -> str | None:
    try:
        return _git(root, "rev-parse", ref).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _duplicate_candidates(
    ref_results: Iterable[dict[str, object]], experiment_id: str
) -> list[dict[str, object]]:
    experiment_id = experiment_id.upper()
    path_pattern = re.compile(
        rf"(?:^|/)experiments/{re.escape(experiment_id)}(?:[-/]|$)",
        re.IGNORECASE,
    )
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for ref_result in ref_results:
        if not isinstance(ref_result, dict):
            continue
        ref = str(ref_result.get("ref", ""))
        commit = ref_result.get("commit")
        experiment_roots = ref_result.get("experiment_roots")
        if not isinstance(experiment_roots, list):
            continue
        for path_item in experiment_roots:
            if not isinstance(path_item, dict):
                continue
            canonical_path = str(path_item.get("canonical_path", ""))
            raw_paths = path_item.get("paths")
            raw_ids = path_item.get("experiment_ids")
            paths = sorted(str(path) for path in raw_paths) if isinstance(raw_paths, list) else []
            experiment_ids = raw_ids if isinstance(raw_ids, list) else []
            if (
                experiment_id not in experiment_ids
                or not path_pattern.search(canonical_path)
            ):
                continue
            key = (ref, canonical_path, commit if isinstance(commit, str) else None)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "experiment_id": experiment_id,
                    "canonical_path": canonical_path,
                    "path": paths[0] if paths else canonical_path,
                    "ref": ref,
                    "commit": commit,
                }
            )
    return sorted(
        candidates,
        key=lambda item: (str(item["canonical_path"]), str(item["ref"])),
    )


def _compact_matches(matches: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for match in matches:
        key = (str(match.get("term", "")), str(match.get("path", "")))
        line = match.get("line")
        if isinstance(line, int):
            grouped.setdefault(key, []).append(line)
    return [
        {
            "term": term,
            "path": path,
            "occurrences": len(lines),
            "first_lines": sorted(lines)[:5],
        }
        for (term, path), lines in sorted(grouped.items())
    ]


def _active_ref(root: Path) -> str | None:
    try:
        return _git(root, "symbolic-ref", "--quiet", "HEAD").strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _upstream_ref(root: Path) -> str | None:
    try:
        return _git(root, "rev-parse", "--symbolic-full-name", "@{upstream}").strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _tree_binding(root: Path, excludes: Iterable[str]) -> str:
    excluded = {Path(value).as_posix() for value in excludes}
    try:
        output = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
        ).stdout
        names = sorted(name.decode("utf-8") for name in output.split(b"\0") if name)
        files = [(name, (root / name).read_bytes()) for name in names if name not in excluded and (root / name).is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
        files = [
            (path.relative_to(root).as_posix(), path.read_bytes())
            for path in _filesystem_files(root)
            if path.relative_to(root).as_posix() not in excluded
        ]
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def tree_binding_sha256(root: Path, excludes: Iterable[str]) -> str:
    """Return the self-reference-safe binding used by persisted preflights."""

    return _tree_binding(root.resolve(), excludes)


def ref_commit(root: Path, ref: str) -> str | None:
    """Resolve a ref exactly, returning ``None`` outside a git repository."""

    return _commit(root.resolve(), ref)


def duplicate_decision_from_refs(
    ref_results: Iterable[dict[str, object]], experiment_id: str
) -> dict[str, object]:
    """Recompute the duplicate decision from recorded, grouped ref evidence."""

    candidates = _duplicate_candidates(ref_results, experiment_id)
    return {
        "status": "duplicate_found" if candidates else "no_duplicate",
        "experiment_id": experiment_id.upper(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_manifest(
    root: Path,
    terms: Iterable[str],
    refs: Iterable[str] | None = None,
    *,
    experiment_id: str,
    hypothesis_query: str,
    binding_excludes: Iterable[str] = (),
) -> dict[str, object]:
    root = root.resolve()
    terms = list(dict.fromkeys(term for term in terms if term))
    if not EXPERIMENT_RE.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id!r}")
    if not hypothesis_query.strip():
        raise ValueError("hypothesis_query must be non-empty")
    refs = list(refs) if refs is not None else list_refs(root)
    active_ref = _active_ref(root)
    upstream_ref = _upstream_ref(root)
    binding_excludes = sorted(dict.fromkeys(Path(value).as_posix() for value in binding_excludes))
    ref_results = []
    for ref in refs:
        result = scan_ref(root, ref, terms)
        result["commit"] = _commit(root, ref)
        ref_results.append(result)
    candidates = _duplicate_candidates(ref_results, experiment_id)
    for result in ref_results:
        result.pop("experiment_paths", None)
        result["matches"] = _compact_matches(result["matches"])
    match_count = sum(int(item["match_count"]) for item in ref_results)
    snapshot_exclusions = {value for value in (active_ref, upstream_ref) if value}
    ref_snapshot = [
        {"ref": str(item["ref"]), "commit": item.get("commit")}
        for item in ref_results
        if item["ref"] not in snapshot_exclusions
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "search_mode": "local_git_refs_read_only",
        "experiment_id": experiment_id.upper(),
        "hypothesis_query": hypothesis_query.strip(),
        "terms": terms,
        "refs": ref_results,
        "ref_snapshot": ref_snapshot,
        "active_worktree": {
            "ref": active_ref,
            "base_commit": _commit(root, "HEAD"),
            "upstream_ref": upstream_ref,
            "upstream_base_commit": _commit(root, upstream_ref) if upstream_ref else None,
            "tree_binding_sha256": _tree_binding(root, binding_excludes),
            "binding_excludes": binding_excludes,
        },
        "remote_refs_at_scan": sorted(
            ref for ref in refs if ref.startswith("refs/remotes/")
        ),
        "searched_ref_count": len(ref_results),
        "match_count": match_count,
        "match_found": match_count > 0,
        "duplicate_found": bool(candidates),
        "duplicate_decision": duplicate_decision_from_refs(ref_results, experiment_id),
        "command": "git for-each-ref --format=%(refname) refs/heads refs/remotes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--term", action="append", dest="terms", help="term to search (repeatable)")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis-query", required=True)
    parser.add_argument("--output", type=Path, help="write JSON manifest to this path")
    parser.add_argument("--binding-exclude", action="append", default=[])
    args = parser.parse_args()
    manifest = build_manifest(
        args.repo,
        args.terms or DEFAULT_TERMS,
        experiment_id=args.experiment_id,
        hypothesis_query=args.hypothesis_query,
        binding_excludes=args.binding_exclude,
    )
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
