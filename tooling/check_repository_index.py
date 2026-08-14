#!/usr/bin/env python3
"""Validate the canonical English repository index.

The checker is deliberately deterministic and offline.  It validates only
tracked canonical files and the exact remote-reference inventory committed in
this repository; it never fetches branches or reads proprietary artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


CANONICAL_MARKDOWN = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("docs/objective.md"),
    Path("docs/hardware/a733.md"),
    Path("docs/hardware/orange-pi-zero-3w.md"),
    Path("docs/profiling/profiling-contract.md"),
    Path("docs/profiling/provenance-and-gates.md"),
    Path("docs/architecture/backend-plan.md"),
    Path("docs/npu/vip9000-capabilities.md"),
    Path("docs/npu/q1-and-partitioning.md"),
    Path("docs/experiments/current-best.md"),
    Path("docs/experiments/branch-inventory.md"),
    Path("docs/experiments/migration-map.md"),
)

REQUIRED_STATUSES = frozenset(
    {"accepted", "rejected", "failed", "diagnostic", "unqualified", "planned"}
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REF_RE = re.compile(r"^(?P<branch>[^@\s]+)@(?P<commit>[0-9a-f]{40})$")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff\u0500-\u052f]")
LINK_RE = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")

EXPECTED_BRANCHES = {
    "main": "426d1467563da211f9621cc2692ec25cf065880d",
    "codex/e022-fused-q1": "d5ac6d0a4570b0a5cab761a2f5afd046baaa67a4",
    "codex/e023-ggml-q1-seam": "7a79466f7f65a67d64c0624eabf60d63652960b8",
    "codex/e047-hard-profiling": "203f3db5454ab99a4b39b2376d43a2ccfdca77f7",
    "codex/e048-per-op-trace": "f5bd54eebbfc12c7399a64fd0f64ab56fa9172f5",
    "codex/e049-nsi-calibration": "c8b7e2696c8f5df89d752dd41e55a5855354cca0",
    "codex/e049b-trace-analysis": "06a810c2f3330f51ec0a8d97de041c4fdde8a452",
    "codex/e049c-arm-pmu": "2b33f6fc03878fdd0f992d758efcecb71d3ae0d1",
    "codex/e054-a76-q1-multiversion": "9ac72e4e7486d85b427ae93d73dcb2ed5ddee8c0",
    "codex/e055-q1-hot-cold": "51d1c1cfd3d2e963344e79dc11719c278b234569",
    "codex/profiling-foundation": "c071476773ad0f7fc499b6a39270a98bc1e25878",
}


def _load_json(root: Path, relative: str) -> tuple[Any | None, list[str]]:
    path = root / relative
    if not path.is_file():
        return None, [f"missing JSON file: {relative}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"invalid JSON {relative}: {exc}"]


def check_registry(root: Path) -> list[str]:
    registry, errors = _load_json(root, "docs/experiments/registry.json")
    schema, schema_errors = _load_json(root, "docs/experiments/schema.json")
    errors.extend(schema_errors)
    if registry is None or schema is None:
        return errors
    if registry.get("schema") != "repository-experiment-registry/v1":
        errors.append("registry schema marker is not repository-experiment-registry/v1")
    if schema.get("$id") != "repository-experiment-registry/v1":
        errors.append("schema $id is not repository-experiment-registry/v1")
    rows = registry.get("experiments")
    if not isinstance(rows, list) or not rows:
        errors.append("registry experiments must be a non-empty list")
        return errors

    ids: set[str] = set()
    statuses: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"registry row {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix} is not an object")
            continue
        required = {
            "id",
            "title",
            "status",
            "claim_class",
            "branch",
            "commit",
            "evidence",
            "metric",
            "notes",
            "is_current_best",
            "end_to_end",
            "optimization_claim",
        }
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")
        experiment_id = row.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            errors.append(f"{prefix} has an invalid id")
        elif experiment_id in ids:
            errors.append(f"duplicate experiment id: {experiment_id}")
        else:
            ids.add(experiment_id)
        status = row.get("status")
        statuses.add(status)
        if status not in REQUIRED_STATUSES:
            errors.append(f"{prefix} has unsupported status: {status!r}")
        commit = row.get("commit")
        if not isinstance(commit, str) or not SHA_RE.fullmatch(commit):
            errors.append(f"{prefix} commit is not a full lowercase SHA-1")
        if not isinstance(row.get("branch"), str) or not row.get("branch"):
            errors.append(f"{prefix} branch is empty")
        if not isinstance(row.get("evidence"), list) or not row.get("evidence"):
            errors.append(f"{prefix} evidence must be a non-empty list")
        if not isinstance(row.get("metric"), dict):
            errors.append(f"{prefix} metric must be an object")
        if not isinstance(row.get("is_current_best"), bool):
            errors.append(f"{prefix} is_current_best must be boolean")
        if not isinstance(row.get("end_to_end"), bool):
            errors.append(f"{prefix} end_to_end must be boolean")
        if not isinstance(row.get("optimization_claim"), bool):
            errors.append(f"{prefix} optimization_claim must be boolean")

    missing_statuses = REQUIRED_STATUSES - statuses
    if missing_statuses:
        errors.append("registry missing statuses: " + ", ".join(sorted(missing_statuses)))

    e035 = next((row for row in rows if isinstance(row, dict) and row.get("id") == "E035"), None)
    if e035 is None:
        errors.append("registry is missing E035")
    else:
        if e035.get("status") != "accepted":
            errors.append("E035 must be accepted")
        if e035.get("metric", {}).get("decode_tok_s") != 0.972497:
            errors.append("E035 decode_tok_s must be exactly 0.972497")
        if e035.get("is_current_best") is not True:
            errors.append("E035 must be current best")
        if e035.get("metric", {}).get("exact_quality") is not True:
            errors.append("E035 must carry exact quality evidence")

    current_best = [row for row in rows if row.get("is_current_best") is True]
    if [row.get("id") for row in current_best] != ["E035"]:
        errors.append("E035 must be the only current-best row")

    e044 = next((row for row in rows if row.get("id") == "E044"), None)
    if e044 is None:
        errors.append("registry is missing E044")
    elif e044.get("is_current_best") is True or e044.get("status") == "accepted":
        errors.append("E044 must not be current best or accepted")

    e049d = next((row for row in rows if row.get("id") == "E049d-v2"), None)
    if e049d is None:
        errors.append("registry is missing E049d-v2")
    else:
        if e049d.get("status") != "diagnostic":
            errors.append("E049d-v2 must be diagnostic")
        if e049d.get("end_to_end") is not False:
            errors.append("E049d-v2 must not be end-to-end")
        if e049d.get("optimization_claim") is not False:
            errors.append("E049d-v2 must not claim optimization")
        if e049d.get("metric", {}).get("steady_marker_tok_s") != 1.11698354:
            errors.append("E049d-v2 marker speed must be exactly 1.11698354")
    return errors


def check_provenance(root: Path) -> list[str]:
    registry, errors = _load_json(root, "docs/experiments/registry.json")
    inventory, inventory_errors = _load_json(
        root, "docs/experiments/branch-inventory.json"
    )
    errors.extend(inventory_errors)
    if registry is None or inventory is None:
        return errors
    branches = {
        row.get("branch")
        for row in inventory.get("branches", [])
        if isinstance(row, dict)
    }
    for row in registry.get("experiments", []):
        if not isinstance(row, dict):
            continue
        prefix = f"{row.get('id', '<unknown>')} provenance"
        if row.get("branch") not in branches:
            errors.append(f"{prefix} branch is absent from branch inventory")
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, dict):
                errors.append(f"{prefix} evidence item is not an object")
                continue
            if not isinstance(evidence.get("kind"), str):
                errors.append(f"{prefix} evidence kind is missing")
            if not isinstance(evidence.get("path"), str) or not evidence.get("path"):
                errors.append(f"{prefix} evidence path is missing")
            ref = evidence.get("ref")
            if ref is not None and not REF_RE.fullmatch(ref):
                errors.append(f"{prefix} evidence ref is not branch@full-sha: {ref!r}")
    return errors


def check_english_only(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in CANONICAL_MARKDOWN:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing canonical Markdown: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot read canonical Markdown {relative}: {exc}")
            continue
        if CYRILLIC_RE.search(text):
            errors.append(f"canonical Markdown contains Cyrillic text: {relative}")
    return errors


def check_links(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in CANONICAL_MARKDOWN:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            target_path = parsed.path
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"link escapes repository: {relative}: {target}")
                continue
            if not resolved.exists():
                errors.append(f"broken link: {relative}: {target}")
    return errors


def check_hardware_facts(root: Path) -> list[str]:
    path = root / "docs/hardware/a733.md"
    if not path.is_file():
        return ["missing hardware fact document: docs/hardware/a733.md"]
    text = path.read_text(encoding="utf-8")
    required = {
        "Verified upstream": "upstream hardware class is absent",
        "Verified on target": "target hardware class is absent",
        "Unknown": "unknown hardware class is absent",
        "32-bit": "memory interface width is absent",
        "LPDDR5-4800": "LPDDR5-4800 ceiling is absent",
        "19.2 GB/s": "theoretical bandwidth ceiling is absent",
        "510 MHz": "controller readback is absent",
        "secure firmware": "secure-firmware boundary is absent",
    }
    return [message for marker, message in required.items() if marker not in text]


def check_branch_inventory(root: Path) -> list[str]:
    inventory, errors = _load_json(
        root, "docs/experiments/branch-inventory.json"
    )
    if inventory is None:
        return errors
    rows = inventory.get("branches")
    if not isinstance(rows, list):
        return ["branch inventory branches must be a list"]
    actual: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"branch inventory row {index} is not an object")
            continue
        branch, commit = row.get("branch"), row.get("commit")
        if not isinstance(branch, str) or not branch:
            errors.append(f"branch inventory row {index} has no branch")
            continue
        if branch in actual:
            errors.append(f"duplicate branch inventory row: {branch}")
        actual[branch] = commit
        if not isinstance(commit, str) or not SHA_RE.fullmatch(commit):
            errors.append(f"branch inventory row {index} has invalid commit")
    if actual != EXPECTED_BRANCHES:
        missing = sorted(set(EXPECTED_BRANCHES) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_BRANCHES))
        changed = sorted(
            branch
            for branch in set(actual) & set(EXPECTED_BRANCHES)
            if actual[branch] != EXPECTED_BRANCHES[branch]
        )
        if missing:
            errors.append("branch inventory missing: " + ", ".join(missing))
        if extra:
            errors.append("branch inventory has unexpected branches: " + ", ".join(extra))
        if changed:
            errors.append("branch inventory SHA changed: " + ", ".join(changed))
    return errors


def run_checks(root: Path) -> list[str]:
    checks = (
        check_registry,
        check_provenance,
        check_english_only,
        check_links,
        check_hardware_facts,
        check_branch_inventory,
    )
    errors: list[str] = []
    for check in checks:
        errors.extend(check(root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    errors = run_checks(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS repository-index/v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
