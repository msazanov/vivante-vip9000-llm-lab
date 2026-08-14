"""Tests for the canonical English repository index."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tooling"))

from check_repository_index import (  # noqa: E402
    CANONICAL_MARKDOWN,
    REQUIRED_STATUSES,
    check_branch_inventory,
    check_english_only,
    check_hardware_facts,
    check_links,
    check_provenance,
    check_registry,
    run_checks,
)


def test_canonical_repository_index_passes_all_checks() -> None:
    assert run_checks(ROOT) == []


def test_canonical_markdown_set_is_present_and_english_only() -> None:
    missing = [path for path in CANONICAL_MARKDOWN if not (ROOT / path).is_file()]
    assert missing == []
    assert check_english_only(ROOT) == []


def test_relative_links_in_canonical_markdown_resolve() -> None:
    assert check_links(ROOT) == []


def test_registry_covers_status_taxonomy_and_key_result_boundaries() -> None:
    registry = json.loads((ROOT / "docs/experiments/registry.json").read_text())
    rows = registry["experiments"]
    assert {row["status"] for row in rows} >= REQUIRED_STATUSES
    assert len({row["id"] for row in rows}) == len(rows)

    e035 = next(row for row in rows if row["id"] == "E035")
    assert e035["status"] == "accepted"
    assert e035["claim_class"] == "full_model"
    assert e035["metric"]["decode_tok_s"] == pytest.approx(0.972497)
    assert e035["metric"]["exact_quality"] is True

    e044 = next(row for row in rows if row["id"] == "E044")
    assert e044["status"] != "accepted"
    assert e044["is_current_best"] is False

    e049d = next(row for row in rows if row["id"] == "E049d-v2")
    assert e049d["status"] == "diagnostic"
    assert e049d["metric"]["steady_marker_tok_s"] == pytest.approx(1.11698354)
    assert e049d["end_to_end"] is False
    assert e049d["optimization_claim"] is False


def test_registry_rows_have_exact_provenance() -> None:
    errors = check_registry(ROOT) + check_provenance(ROOT)
    assert errors == []
    registry = json.loads((ROOT / "docs/experiments/registry.json").read_text())
    sha = re.compile(r"^[0-9a-f]{40}$")
    for row in registry["experiments"]:
        assert sha.fullmatch(row["commit"]), row["id"]
        assert row["branch"], row["id"]
        assert row["evidence"], row["id"]


def test_hardware_facts_are_explicitly_classified() -> None:
    assert check_hardware_facts(ROOT) == []
    text = (ROOT / "docs/hardware/a733.md").read_text()
    for label in ("Verified upstream", "Verified on target", "Unknown"):
        assert label in text
    assert "32-bit" in text
    assert "19.2 GB/s" in text
    assert "510 MHz" in text
    assert "secure firmware" in text.lower()


def test_branch_inventory_is_complete_and_does_not_touch_e055() -> None:
    assert check_branch_inventory(ROOT) == []
    inventory = json.loads(
        (ROOT / "docs/experiments/branch-inventory.json").read_text()
    )
    names = {row["branch"] for row in inventory["branches"]}
    assert "codex/profiling-foundation" in names
    assert "codex/e055-q1-hot-cold" in names
    assert "codex/repository-index" not in names
    assert all(row["commit"] != "51d1c1cfd3d2e963344e79dc11719c278b234569" for row in inventory["branches"] if row["branch"] != "codex/e055-q1-hot-cold")


def test_canonical_layer_contains_pointers_not_raw_payload_copies() -> None:
    canonical_roots = [ROOT / "docs/objective.md", ROOT / "docs/experiments"]
    forbidden_suffixes = {".nb", ".gguf", ".bin", ".so", ".tar.gz"}
    copied = [
        path
        for root in canonical_roots
        for path in root.rglob("*")
        if path.is_file() and any(str(path).endswith(suffix) for suffix in forbidden_suffixes)
    ]
    assert copied == []
