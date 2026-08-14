"""Tests for the canonical English repository index."""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from shutil import copy2

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tooling"))
import check_repository_index as repository_index  # noqa: E402

from check_repository_index import (  # noqa: E402
    CANONICAL_MARKDOWN,
    REQUIRED_STATUSES,
    check_artifact_policy,
    check_branch_inventory,
    check_experiment_coverage,
    check_english_only,
    check_hardware_facts,
    check_links,
    check_privacy,
    check_public_artifact_manifest,
    check_manifest_coverage,
    check_provenance,
    check_registry,
    discover_canonical_markdown,
    run_checks,
    _check_commit_path,
)


def test_canonical_repository_index_passes_all_checks() -> None:
    assert run_checks(ROOT) == []


def test_canonical_markdown_set_is_present_and_english_only() -> None:
    discovered = discover_canonical_markdown(ROOT)
    assert discovered == tuple(sorted(discovered))
    missing = [path for path in discovered if not (ROOT / path).is_file()]
    assert missing == []
    assert check_english_only(ROOT) == []


def test_relative_links_in_canonical_markdown_resolve() -> None:
    assert check_links(ROOT) == []


def test_canonical_layer_contains_a_real_fragment_link() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "current-best.md#qualified-full-model-best" in readme


def test_link_checker_rejects_missing_heading_fragment(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    (fixture / "README.md").write_text("[target](target.md#missing-heading)\n")
    (fixture / "target.md").write_text("# Present heading\n")
    errors = check_links(fixture)
    assert any("broken fragment" in error for error in errors)


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

    e049_nsi = next(row for row in rows if row["id"] == "E049-nsi")
    assert e049_nsi["branch"] == "codex/e049-nsi-calibration"
    assert e049_nsi["commit"] == "c8b7e2696c8f5df89d752dd41e55a5855354cca0"
    assert e049_nsi["status"] == "diagnostic"
    assert e049_nsi["evidence"][0]["path"].startswith(
        "experiments/E049-nsi-calibration/"
    )
    assert "inconclusive" in e049_nsi["notes"]
    assert "MB/s" in e049_nsi["notes"]
    assert "saturation" in e049_nsi["notes"]


def test_registry_rows_have_exact_provenance() -> None:
    errors = check_registry(ROOT) + check_provenance(ROOT)
    assert errors == []
    registry = json.loads((ROOT / "docs/experiments/registry.json").read_text())
    sha = re.compile(r"^[0-9a-f]{40}$")
    for row in registry["experiments"]:
        assert sha.fullmatch(row["commit"]), row["id"]
        assert row["branch"], row["id"]
        assert row["evidence"], row["id"]


def test_every_inventoried_experiment_directory_is_indexed() -> None:
    assert check_experiment_coverage(ROOT) == []


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


def test_public_artifact_manifest_and_privacy_gate_pass() -> None:
    assert check_public_artifact_manifest(ROOT) == []
    assert check_privacy(ROOT) == []


def _copy_registry_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    for name in ("registry.json", "schema.json", "branch-inventory.json"):
        copy2(ROOT / "docs/experiments" / name, fixture / "docs/experiments" / name)
    return fixture


def test_schema_rejects_bogus_claim_class_and_malformed_id(tmp_path: Path) -> None:
    fixture = _copy_registry_fixture(tmp_path)
    registry_path = fixture / "docs/experiments/registry.json"
    registry = json.loads(registry_path.read_text())
    registry["experiments"][0]["claim_class"] = "invented"
    registry["experiments"][1]["id"] = "not-an-experiment"
    registry_path.write_text(json.dumps(registry))
    errors = check_registry(fixture)
    assert any("claim_class" in error for error in errors)
    assert any("id" in error for error in errors)


def test_schema_rejects_wrong_repository_and_missing_reference(tmp_path: Path) -> None:
    fixture = _copy_registry_fixture(tmp_path)
    registry_path = fixture / "docs/experiments/registry.json"
    registry = json.loads(registry_path.read_text())
    registry["repository"] = "other/project"
    del registry["experiments"][0]["evidence"][0]["ref"]
    registry_path.write_text(json.dumps(registry))
    errors = check_registry(fixture) + check_provenance(fixture)
    assert any("repository" in error for error in errors)
    assert any("ref" in error for error in errors)


def test_provenance_rejects_mismatched_ref_and_nonexistent_commit(tmp_path: Path) -> None:
    fixture = _copy_registry_fixture(tmp_path)
    registry_path = fixture / "docs/experiments/registry.json"
    registry = json.loads(registry_path.read_text())
    evidence = registry["experiments"][0]["evidence"][0]
    evidence["ref"] = "main@0000000000000000000000000000000000000000"
    registry_path.write_text(json.dumps(registry))
    errors = check_provenance(fixture)
    assert any("does not match row" in error for error in errors)
    assert any("all-zero" in error or "does not exist" in error for error in errors)


def test_git_provenance_rejects_missing_path_at_exact_commit() -> None:
    errors = _check_commit_path(
        ROOT,
        "main",
        "426d1467563da211f9621cc2692ec25cf065880d",
        "experiments/does-not-exist/README.md",
        "adversarial",
    )
    assert any("path does not exist" in error for error in errors)


def test_english_checker_rejects_non_latin_scripts_but_allows_technical_symbols(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    (fixture / "README.md").write_text("# English\n\nASCII code `x -> y` and × ° →.\n")
    (fixture / "bad.md").write_text("# English\n\nGreek α and CJK 中 are not canonical English.\n")
    errors = check_english_only(fixture)
    assert any("bad.md" in error for error in errors)
    assert not any("README.md" in error for error in errors)


def test_hardware_checker_rejects_unclassified_garbage(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/hardware").mkdir(parents=True)
    (fixture / "docs/hardware/a733.md").write_text(
        "Verified upstream, Verified on target, Unknown, 32-bit, LPDDR5-4800, "
        "19.2 GB/s, 510 MHz, secure firmware\n"
    )
    assert check_hardware_facts(fixture)


def test_privacy_gate_rejects_secret_pattern(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    fake_prefix = "gh" + "p_"
    (fixture / "README.md").write_text(
        f"public note token={fake_prefix}123456789012345678901234567890\n"
    )
    assert check_privacy(fixture)


def test_privacy_scans_untracked_tooling_and_encoded_credentials(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "tooling").mkdir(parents=True)
    secret = "pass" + "word=encoded-secret-value"
    encoded = base64.b64encode(secret.encode()).decode()
    hex_encoded = secret.encode().hex()
    (fixture / "tooling/unknown.py").write_text(f"blob = '{encoded}'\n")
    (fixture / "notes.txt").write_text(f"hex={hex_encoded}\n")
    (fixture / "payload.bin").write_bytes(b"\x00" + b"pass" + b"word=binary-secret-value")
    errors = check_privacy(fixture)
    assert any("unknown.py" in error for error in errors)
    assert any("notes.txt" in error for error in errors)
    assert any("payload.bin" in error for error in errors)
    assert all(secret not in error for error in errors)


def test_privacy_recursively_decodes_percent_and_c_escaped_credentials(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    secret = "pass" + "word=recursive-secret-value"
    percent_encoded = "".join(f"%{byte:02X}" for byte in secret.encode())
    c_escaped = "".join(f"\\x{byte:02x}" for byte in secret.encode())
    nested_base64 = base64.b64encode(percent_encoded.encode()).decode()
    (fixture / "percent.txt").write_text(percent_encoded)
    (fixture / "escaped.txt").write_text(c_escaped)
    (fixture / "nested.txt").write_text(nested_base64)
    errors = check_privacy(fixture)
    assert any("percent.txt" in error for error in errors)
    assert any("escaped.txt" in error for error in errors)
    assert any("nested.txt" in error for error in errors)
    assert all(secret not in error for error in errors)


def test_manifest_coverage_rejects_unmanifested_public_payload(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    (fixture / "tooling").mkdir()
    (fixture / "new-model.bin").write_bytes(b"public payload")
    (fixture / "tooling/npu_tool.py").write_text("print('public tool')\n")
    (fixture / "docs/experiments/public-artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema": "repository-public-artifact-manifest/v1",
                "repository": "msazanov/vivante-vip9000-llm-lab",
                "policy": {
                    "allowed_public_artifacts": ["binaries"],
                    "prohibited_data": ["credentials"],
                },
                "artifacts": [],
            }
        )
    )
    errors = check_manifest_coverage(fixture)
    assert any("new-model.bin" in error for error in errors)
    assert any("npu_tool.py" in error for error in errors)


def test_manifest_coverage_is_not_suffix_or_name_heuristic(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    payloads = {
        "tooling/unknown_tool.py": b"#!/usr/bin/env python3\n",
        "patches/vendor-sdk.tar.gz": b"archive",
        "kernels/opaque.tgz": b"archive",
        "sdk/release.zip": b"archive",
        "models/weights.tar.xz": b"archive",
        "artifacts/download": b"PK\x03\x04archive",
        "artifacts/runner": b"\x7fELF\x02\x01binary",
    }
    for relative, contents in payloads.items():
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (fixture / "docs/experiments/public-artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema": "repository-public-artifact-manifest/v1",
                "repository": "msazanov/vivante-vip9000-llm-lab",
                "policy": {
                    "allowed_public_artifacts": ["binaries"],
                    "prohibited_data": ["credentials"],
                },
                "artifacts": [],
            }
        )
    )
    errors = check_manifest_coverage(fixture)
    for relative in payloads:
        assert any(relative in error for error in errors), relative


def test_artifact_class_policy_keeps_lfs_and_manifest_classes_in_sync() -> None:
    assert repository_index.check_artifact_class_policy(ROOT) == []


def test_release_manifest_requires_nonplaceholder_digest_and_positive_size(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    manifest_path = fixture / "docs/experiments/public-artifact-manifest.json"
    manifest = {
        "schema": "repository-public-artifact-manifest/v1",
        "repository": "msazanov/vivante-vip9000-llm-lab",
        "policy": {
            "allowed_public_artifacts": ["binaries"],
            "prohibited_data": ["credentials"],
        },
        "artifacts": [
            {
                "path": "release/missing.bin",
                "sha256": "0123456789abcdef" * 4,
                "byte_size": 42,
                "origin": "release source",
                "source_commit": "c071476773ad0f7fc499b6a39270a98bc1e25878",
                "build_runtime_toolchain": "compiler 1",
                "destination": "release-assets",
                "class": "binaries",
                "status": "published",
                "scientific_use": False,
                "release_asset_locator": "https://github.com/msazanov/vivante-vip9000-llm-lab/releases/download/v1/missing.bin",
                "checksum_provenance": "signed release SHA-256 attestation",
                "verification_state": "verified-attestation",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    assert check_public_artifact_manifest(fixture) == []
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest["artifacts"][0]["byte_size"] = 0
    manifest_path.write_text(json.dumps(manifest))
    errors = check_public_artifact_manifest(fixture)
    assert any("nonzero" in error or "placeholder" in error for error in errors)
    assert any("byte_size" in error for error in errors)


def test_policy_checker_rejects_blanket_artifact_ban(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    for relative in (
        "README.md",
        "AGENTS.md",
        "docs/profiling/provenance-and-gates.md",
        "docs/npu/vip9000-capabilities.md",
        "docs/experiments/migration-map.md",
        "tooling/README.md",
        ".gitignore",
        "docs/toolchain/inventory.md",
        ".gitattributes",
        "docs/experiments/public-artifacts.md",
    ):
        destination = fixture / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT / relative, destination)
    (fixture / "README.md").write_text(
        (fixture / "README.md").read_text() + "\nPublic weights must never be published.\n"
    )
    errors = check_artifact_policy(fixture)
    assert any("blanket" in error or "publish" in error for error in errors)


def test_markdown_discovery_ignores_untracked_test_cache(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    (fixture / "README.md").write_text("# English\n")
    (fixture / ".pytest_cache").mkdir()
    (fixture / ".pytest_cache/bad.md").write_text("Greek α\n")
    assert discover_canonical_markdown(fixture) == (Path("README.md"),)
    assert check_english_only(fixture) == []


def test_hardware_checker_rejects_theoretical_ceiling_as_measurement(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    destination = fixture / "docs/hardware/a733.md"
    destination.parent.mkdir(parents=True)
    copy2(ROOT / "docs/hardware/a733.md", destination)
    destination.write_text(destination.read_text() + "\n19.2 GB/s measured sustained bandwidth.\n")
    errors = check_hardware_facts(fixture)
    assert any("theoretical" in error or "measured" in error for error in errors)


@pytest.mark.parametrize(
    "claim",
    (
        "Observed throughput was 19.2 GB/s on the board.",
        "The actual sustained bandwidth reached 19.2 GB/s.",
    ),
)
def test_hardware_checker_rejects_semantic_19_2_gb_s_measurement_claims(
    tmp_path: Path, claim: str
) -> None:
    fixture = tmp_path / "repo"
    destination = fixture / "docs/hardware/a733.md"
    destination.parent.mkdir(parents=True)
    copy2(ROOT / "docs/hardware/a733.md", destination)
    destination.write_text(destination.read_text() + "\n" + claim + "\n")
    errors = check_hardware_facts(fixture)
    assert any("theoretical" in error or "measured" in error for error in errors)


def test_registry_rejects_metric_claim_semantic_mismatch(tmp_path: Path) -> None:
    fixture = _copy_registry_fixture(tmp_path)
    registry_path = fixture / "docs/experiments/registry.json"
    registry = json.loads(registry_path.read_text())
    e049d = next(row for row in registry["experiments"] if row["id"] == "E049d-v2")
    e049d["claim_class"] = "full_model"
    e049d["metric"]["steady_marker_tok_s"] = -1
    registry_path.write_text(json.dumps(registry))
    errors = check_registry(fixture)
    assert any("claim_class" in error or "full_model" in error for error in errors)
    assert any("metric" in error or "positive" in error for error in errors)


def test_registry_requires_quality_provenance_and_repeatability_for_qualified_full_model(
    tmp_path: Path,
) -> None:
    fixture = _copy_registry_fixture(tmp_path)
    registry_path = fixture / "docs/experiments/registry.json"
    registry = json.loads(registry_path.read_text())
    e035 = next(row for row in registry["experiments"] if row["id"] == "E035")
    e035.pop("provenance", None)
    e035.pop("repeatability", None)
    registry_path.write_text(json.dumps(registry))
    errors = check_registry(fixture)
    assert any("provenance" in error for error in errors)
    assert any("repeatability" in error for error in errors)


def test_release_manifest_requires_immutable_locator_attestation_and_state(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    manifest_path = fixture / "docs/experiments/public-artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "repository-public-artifact-manifest/v1",
                "repository": "msazanov/vivante-vip9000-llm-lab",
                "policy": {
                    "allowed_public_artifacts": ["binaries"],
                    "prohibited_data": ["credentials"],
                },
                "artifacts": [
                    {
                        "path": "release/planned.bin",
                        "sha256": "0123456789abcdef" * 4,
                        "byte_size": 42,
                        "origin": "release source",
                        "source_commit": "c071476773ad0f7fc499b6a39270a98bc1e25878",
                        "build_runtime_toolchain": "compiler 1",
                        "destination": "release-assets",
                        "class": "binaries",
                        "status": "planned",
                        "scientific_use": True,
                        "verification_state": "verified-attestation",
                    }
                ],
            }
        )
    )
    errors = check_public_artifact_manifest(fixture)
    assert any("locator" in error or "URL" in error for error in errors)
    assert any("checksum" in error or "attestation" in error for error in errors)
    assert any("unverified" in error or "scientific" in error for error in errors)


def test_manifest_rejects_unknown_policy_and_artifact_classes(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    manifest_path = fixture / "docs/experiments/public-artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "repository-public-artifact-manifest/v1",
                "repository": "msazanov/vivante-vip9000-llm-lab",
                "policy": {
                    "allowed_public_artifacts": ["invented"],
                    "prohibited_data": ["private databases"],
                },
                "artifacts": [
                    {
                        "path": "release/missing.bin",
                        "sha256": "0123456789abcdef" * 4,
                        "byte_size": 42,
                        "origin": "release source",
                        "source_commit": "c071476773ad0f7fc499b6a39270a98bc1e25878",
                        "build_runtime_toolchain": "compiler 1",
                        "destination": "release-assets",
                        "class": "invented",
                        "status": "planned",
                        "scientific_use": False,
                        "verification_state": "unverified",
                        "release_asset_locator": "https://github.com/msazanov/vivante-vip9000-llm-lab/releases/download/v1/missing.bin",
                        "checksum_provenance": "planned checksum",
                    }
                ],
            }
        )
    )
    errors = check_public_artifact_manifest(fixture)
    assert any("unsupported public artifact class" in error for error in errors)
    assert any("prohibited-data class" in error for error in errors)
    assert any("class is outside" in error for error in errors)


def test_candidate_scope_requires_explicit_safe_source_policy(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    copy2(ROOT / "docs/experiments/public-artifact-classes.json", fixture / "docs/experiments/public-artifact-classes.json")
    (fixture / "docs/new-canonical.md").write_text("# Canonical source\n")
    (fixture / "tooling/unknown_tool.py").parent.mkdir()
    (fixture / "tooling/unknown_tool.py").write_text("print('payload')\n")
    (fixture / "docs/experiments/public-artifact-manifest.json").write_text(
        json.dumps({"artifacts": []})
    )
    errors = check_manifest_coverage(fixture)
    assert any("tooling/unknown_tool.py" in error for error in errors)
    assert not any("docs/new-canonical.md" in error for error in errors)


def test_artifact_manifest_rejects_inconsistent_hash_and_size(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    (fixture / "docs/experiments").mkdir(parents=True)
    (fixture / "public.txt").write_text("public\n")
    (fixture / "docs/experiments/public-artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema": "repository-public-artifact-manifest/v1",
                "repository": "msazanov/vivante-vip9000-llm-lab",
                "policy": {
                    "allowed_public_artifacts": ["binaries"],
                    "prohibited_data": ["credentials"],
                },
                "artifacts": [
                    {
                        "path": "public.txt",
                        "sha256": "0" * 64,
                        "byte_size": 99,
                        "origin": "test",
                        "source_commit": "c071476773ad0f7fc499b6a39270a98bc1e25878",
                        "build_runtime_toolchain": "none",
                        "destination": "git",
                        "class": "binaries",
                        "status": "published",
                        "scientific_use": False,
                        "verification_state": "verified-local",
                    }
                ],
            }
        )
    )
    errors = check_public_artifact_manifest(fixture)
    assert any("sha256" in error for error in errors)
    assert any("byte_size" in error for error in errors)


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
