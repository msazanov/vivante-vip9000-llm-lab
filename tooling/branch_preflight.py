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


SCHEMA_VERSION = "e053-branch-preflight/v1"
DEFAULT_TERMS = ("LFM2.5", "LFM", "E047", "E053")
EXPERIMENT_RE = re.compile(r"\bE\d{3,}(?:[a-z0-9-]*)?\b", re.IGNORECASE)
MAX_LINE_CHARS = 400


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
    lowered_terms = [(term, term.casefold()) for term in terms]
    matches: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        folded = line.casefold()
        for term, folded_term in lowered_terms:
            if folded_term in folded:
                matches.append(
                    {
                        "term": term,
                        "path": path,
                        "line": line_number,
                        "text": line[:MAX_LINE_CHARS],
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
    for path, data in _read_ref_files(root, ref):
        matches.extend(_match_lines(path, data, terms))
    experiment_ids = sorted(
        {
            item.upper()
            for match in matches
            for item in EXPERIMENT_RE.findall(f"{match['path']} {match['text']}")
        }
    )
    return {
        "ref": ref,
        "match_count": len(matches),
        "matches": matches,
        "experiment_ids": experiment_ids,
    }


def _refs(root: Path) -> list[str]:
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


def _commit(root: Path, ref: str) -> str | None:
    try:
        return _git(root, "rev-parse", ref).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(root: Path, terms: Iterable[str], refs: Iterable[str] | None = None) -> dict[str, object]:
    root = root.resolve()
    terms = list(dict.fromkeys(term for term in terms if term))
    refs = list(refs) if refs is not None else _refs(root)
    ref_results = []
    for ref in refs:
        result = scan_ref(root, ref, terms)
        result["commit"] = _commit(root, ref)
        ref_results.append(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "search_mode": "local_git_refs_read_only",
        "terms": terms,
        "refs": ref_results,
        "searched_ref_count": len(ref_results),
        "match_count": sum(int(item["match_count"]) for item in ref_results),
        "duplicate_found": any(int(item["match_count"]) for item in ref_results),
        "command": "git for-each-ref --format=%(refname) refs/heads refs/remotes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--term", action="append", dest="terms", help="term to search (repeatable)")
    parser.add_argument("--output", type=Path, help="write JSON manifest to this path")
    args = parser.parse_args()
    manifest = build_manifest(args.repo, args.terms or DEFAULT_TERMS)
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
