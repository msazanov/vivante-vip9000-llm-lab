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
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "e053-branch-preflight/v2"
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
    return {
        "ref": ref,
        "match_count": len(matches),
        "matches": matches,
        "experiment_ids": experiment_ids,
        "experiment_paths": experiment_paths,
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
        ref = str(ref_result["ref"])
        commit = ref_result.get("commit")
        for path_item in ref_result.get("experiment_paths", []):
            path = str(path_item["path"])
            if experiment_id not in path_item.get("experiment_ids", []) or not path_pattern.search(path):
                continue
            key = (ref, path, commit if isinstance(commit, str) else None)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "experiment_id": experiment_id,
                    "path": path,
                    "ref": ref,
                    "commit": commit,
                }
            )
    return sorted(candidates, key=lambda item: (str(item["path"]), str(item["ref"])))


def build_manifest(
    root: Path,
    terms: Iterable[str],
    refs: Iterable[str] | None = None,
    *,
    experiment_id: str,
    hypothesis_query: str,
) -> dict[str, object]:
    root = root.resolve()
    terms = list(dict.fromkeys(term for term in terms if term))
    if not EXPERIMENT_RE.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id!r}")
    if not hypothesis_query.strip():
        raise ValueError("hypothesis_query must be non-empty")
    refs = list(refs) if refs is not None else list_refs(root)
    ref_results = []
    for ref in refs:
        result = scan_ref(root, ref, terms)
        result["commit"] = _commit(root, ref)
        for match in result["matches"]:
            match["ref"] = ref
            match["commit"] = result["commit"]
        ref_results.append(result)
    candidates = _duplicate_candidates(ref_results, experiment_id)
    for result in ref_results:
        result.pop("experiment_paths", None)
    match_count = sum(int(item["match_count"]) for item in ref_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "search_mode": "local_git_refs_read_only",
        "experiment_id": experiment_id.upper(),
        "hypothesis_query": hypothesis_query.strip(),
        "terms": terms,
        "refs": ref_results,
        "remote_refs_at_scan": sorted(
            ref for ref in refs if ref.startswith("refs/remotes/")
        ),
        "searched_ref_count": len(ref_results),
        "match_count": match_count,
        "match_found": match_count > 0,
        "duplicate_found": bool(candidates),
        "duplicate_decision": {
            "status": "duplicate_found" if candidates else "no_duplicate",
            "experiment_id": experiment_id.upper(),
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "command": "git for-each-ref --format=%(refname) refs/heads refs/remotes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--term", action="append", dest="terms", help="term to search (repeatable)")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis-query", required=True)
    parser.add_argument("--output", type=Path, help="write JSON manifest to this path")
    args = parser.parse_args()
    manifest = build_manifest(
        args.repo,
        args.terms or DEFAULT_TERMS,
        experiment_id=args.experiment_id,
        hypothesis_query=args.hypothesis_query,
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
