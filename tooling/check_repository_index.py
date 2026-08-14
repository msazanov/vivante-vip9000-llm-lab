#!/usr/bin/env python3
"""Run deterministic gates for the canonical repository index.

The checker validates the small, public index layer and the exact immutable
branch snapshot recorded in it.  It never fetches, rewrites, or imports raw
experiment payloads.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import fnmatch
import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

try:  # The project test image supplies jsonschema; keep an offline fallback.
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - exercised only in minimal environments
    Draft202012Validator = None


REPOSITORY = "msazanov/vivante-vip9000-llm-lab"
SCHEMA_MARKER = "repository-experiment-registry/v1"
MANIFEST_MARKER = "repository-public-artifact-manifest/v1"
REQUIRED_STATUSES = frozenset(
    {"accepted", "rejected", "failed", "diagnostic", "unqualified", "planned"}
)
REQUIRED_CLAIM_CLASSES = frozenset(
    {"full_model", "operator", "hardware", "tooling", "diagnostic", "screen", "planned"}
)
REQUIRED_EVIDENCE_KINDS = frozenset(
    {"repository_path", "legacy_pointer", "branch_pointer", "external_pointer"}
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REF_RE = re.compile(r"^(?P<branch>[^@\s]+)@(?P<commit>[0-9a-f]{40})$")
ID_RE = re.compile(r"^E[0-9]{3}[A-Za-z0-9-]*$")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))")

# These are deliberately explicit.  They are immutable historical evidence,
# not canonical orientation, so their original language must not be rewritten.
IMMUTABLE_LEGACY_MARKDOWN: dict[str, str] = {
    ".superpowers/": "historical generated reports",
    "docs/evidence/": "raw historical evidence",
    "docs/superpowers/": "historical design plans and specifications",
    "experiments/": "authoritative experiment-branch documents",
    "benchmarks/models/": "legacy generated model card",
}

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

# An inventoried directory may be excluded only by adding its exact name and
# a reason here.  The current snapshot intentionally has no exclusions.
EXCLUDED_EXPERIMENT_DIRECTORIES: dict[str, str] = {}

POLICY_FILES = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("docs/profiling/provenance-and-gates.md"),
    Path("docs/npu/vip9000-capabilities.md"),
    Path("docs/experiments/migration-map.md"),
    Path("tooling/README.md"),
    Path("docs/toolchain/inventory.md"),
    Path(".gitignore"),
)

SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("telegram-token", re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("url-credential", re.compile(r"(?i)\bhttps?://[^\s/@:]+:[^\s/@]+@[^\s]+")),
    ("query-credential", re.compile(
        r"(?i)\b(?:password|passwd|token|secret|api[_-]?key|login|username)"
        r"\s*[:=]\s*(?!<[^>]+>|\{[^}]+\}|\[redacted\])[^\s`'\"]{8,}"
    )),
)

ENCODED_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
ENCODED_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
PUBLIC_ARTIFACT_SUFFIXES = (
    (".gguf", "weights"),
    (".safetensors", "weights"),
    (".onnx", "weights"),
    (".tflite", "weights"),
    (".nb", "NBGs"),
    (".nbg", "NBGs"),
    (".bin", "binaries"),
    (".blob", "binaries"),
    (".so", "binaries"),
    (".a", "binaries"),
    (".ko", "binaries"),
    (".vx", "custom kernels/source"),
    (".cl", "custom kernels/source"),
    (".patch", "SDK or kernel patches"),
    (".c", "custom kernels/source"),
    (".cc", "custom kernels/source"),
    (".cpp", "custom kernels/source"),
    (".h", "custom kernels/source"),
    (".hpp", "custom kernels/source"),
)
PRIVACY_SCAN_ALLOWLIST: dict[str, str] = {
    "docs/evidence/*": "immutable raw historical evidence",
    "experiments/*/results/*": "immutable raw experiment payloads",
    "benchmarks/results/*": "immutable raw benchmark payloads",
    "benchmarks/charts/*": "generated chart binaries",
}
PLACEHOLDER_DIGESTS = frozenset({"0" * 64, "f" * 64, "deadbeef" * 8})
BASELINE_COMMIT = "c071476773ad0f7fc499b6a39270a98bc1e25878"
IGNORED_DISCOVERY_PREFIXES = (".git/", ".pytest_cache/", "__pycache__/", ".mypy_cache/")
FORBIDDEN_POLICY_PATTERNS = (
    ("blanket artifact ban", re.compile(
        r"(?i)\b(?:weights|binaries?|NBGs?|kernels?|SDK(?: archives?| files?)?|NPU tools)\b"
        r"[^.\n]{0,100}\b(?:remain|stay)\s+(?:outside Git|external)"
    )),
    ("blanket publication ban", re.compile(
        r"(?i)\b(?:weights|binaries?|NBGs?|kernels?|SDK(?: archives?| files?)?|NPU tools)\b"
        r"[^.\n]{0,100}\b(?:must never|must not|may not|cannot)\s+(?:be\s+)?"
        r"(?:published|committed|added|tracked)"
    )),
)


def discover_canonical_markdown(root: Path) -> tuple[Path, ...]:
    """Return all canonical Markdown paths in stable lexical order."""

    git_result = _git_index_markdown(root)
    relative_paths = git_result if git_result is not None else [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.md")
        if path.is_file()
    ]
    paths: list[Path] = []
    for relative in relative_paths:
        if any(relative.startswith(prefix) for prefix in IGNORED_DISCOVERY_PREFIXES):
            continue
        if any(
            relative == prefix.rstrip("/") or relative.startswith(prefix)
            for prefix in IMMUTABLE_LEGACY_MARKDOWN
        ):
            continue
        paths.append(Path(relative))
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _git_index_markdown(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "-z", "--", "*.md"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


CANONICAL_MARKDOWN = discover_canonical_markdown(Path(__file__).resolve().parents[1])


def _load_json(root: Path, relative: str) -> tuple[Any | None, list[str]]:
    path = root / relative
    if not path.is_file():
        return None, [f"missing JSON file: {relative}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"invalid JSON {relative}: {exc}"]


def _schema_error_strings(instance: Any, schema: dict[str, Any]) -> list[str]:
    if Draft202012Validator is not None:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
        return [
            "schema validation at "
            + ("/" + "/".join(str(part) for part in error.path) if error.path else "/")
            + f": {error.message}"
            for error in errors
        ]

    # Small standard-library fallback for environments without jsonschema.
    errors: list[str] = []

    def visit(value: Any, rule: dict[str, Any], path: str) -> None:
        if "$ref" in rule:
            ref = rule["$ref"]
            target: Any = schema
            for part in ref.removeprefix("#/").split("/"):
                target = target[part]
            visit(value, target, path)
            return
        if "const" in rule and value != rule["const"]:
            errors.append(f"schema validation at {path}: {value!r} is not {rule['const']!r}")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"schema validation at {path}: unsupported value {value!r}")
        expected = rule.get("type")
        type_ok = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
        }
        if expected in type_ok and not type_ok[expected]:
            errors.append(f"schema validation at {path}: expected {expected}")
            return
        if isinstance(value, dict):
            for key in rule.get("required", []):
                if key not in value:
                    errors.append(f"schema validation at {path}: missing {key!r}")
            for key, child in rule.get("properties", {}).items():
                if key in value:
                    visit(value[key], child, f"{path}/{key}")
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, rule.get("items", {}), f"{path}/{index}")
            if len(value) < rule.get("minItems", 0):
                errors.append(f"schema validation at {path}: too few items")
        if isinstance(value, str):
            if len(value) < rule.get("minLength", 0):
                errors.append(f"schema validation at {path}: string is empty")
            if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
                errors.append(f"schema validation at {path}: pattern mismatch")

    visit(instance, schema, "/")
    return errors


def _git_root(root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip())


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None


def _git_file_paths(root: Path) -> list[str] | None:
    result = _git(root, "ls-files", "--cached", "-z")
    if result is None or result.returncode != 0:
        return None
    return [item for item in result.stdout.split("\0") if item]


def _candidate_file_paths(root: Path) -> tuple[Path, ...]:
    """Return tracked plus non-ignored untracked regular files deterministically."""

    tracked = _git_file_paths(root)
    if tracked is not None:
        untracked_result = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        if untracked_result is not None and untracked_result.returncode == 0:
            untracked = [item for item in untracked_result.stdout.split("\0") if item]
        else:
            untracked = []
        names = sorted(set(tracked + untracked))
    else:
        names = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not any(path.relative_to(root).as_posix().startswith(prefix) for prefix in IGNORED_DISCOVERY_PREFIXES)
        )
    return tuple(root / name for name in names if (root / name).is_file() and not (root / name).is_symlink())


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _privacy_allowlisted(relative: str) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in PRIVACY_SCAN_ALLOWLIST)


def _is_new_since_baseline(root: Path, relative: str) -> bool:
    git_root = _git_root(root)
    if git_root is None:
        return True
    tracked = _git_file_paths(git_root) or []
    if relative not in tracked:
        return True
    result = _git(git_root, "diff", "--quiet", BASELINE_COMMIT, "--", relative)
    return result is None or result.returncode != 0


def _artifact_class(relative: str) -> str | None:
    lowered = relative.lower()
    for suffix, artifact_class in PUBLIC_ARTIFACT_SUFFIXES:
        if lowered.endswith(suffix):
            return artifact_class
    return None


def _publication_candidates(root: Path) -> tuple[Path, ...]:
    candidates = []
    for path in _candidate_file_paths(root):
        relative = _relative_path(root, path)
        if _artifact_class(relative) and _is_new_since_baseline(root, relative):
            candidates.append(path)
    return tuple(candidates)


def _secret_labels(text: str) -> set[str]:
    labels = {label for label, pattern in SECRET_PATTERNS if pattern.search(text)}
    for candidate in ENCODED_BASE64_RE.findall(text):
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except (ValueError, binascii.Error):
            continue
        decoded_text = decoded.decode("utf-8", errors="ignore")
        if decoded_text and sum(character.isprintable() for character in decoded_text) / len(decoded_text) >= 0.85:
            labels.update(f"encoded-{label}" for label, pattern in SECRET_PATTERNS if pattern.search(decoded_text))
    for candidate in ENCODED_HEX_RE.findall(text):
        try:
            decoded = bytes.fromhex(candidate)
        except ValueError:
            continue
        decoded_text = decoded.decode("utf-8", errors="ignore")
        if decoded_text and sum(character.isprintable() for character in decoded_text) / len(decoded_text) >= 0.85:
            labels.update(f"encoded-{label}" for label, pattern in SECRET_PATTERNS if pattern.search(decoded_text))
    return labels


def _scan_file_secret_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    carry = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                combined = carry + chunk
                text = combined.decode("utf-8", errors="ignore")
                labels.update(_secret_labels(text))
                for run in re.findall(rb"[\x20-\x7e]{8,}", combined):
                    labels.update(_secret_labels(run.decode("ascii", errors="ignore")))
                carry = combined[-512:]
    except OSError:
        return {"unreadable-file"}
    return labels


def _resolve_branch_ref(git_root: Path, branch: str) -> str | None:
    for candidate in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
        result = _git(git_root, "show-ref", "--verify", "--quiet", candidate)
        if result is not None and result.returncode == 0:
            return candidate
    return None


def _check_commit_reference(root: Path, branch: str, commit: str, prefix: str) -> list[str]:
    errors: list[str] = []
    if commit == "0" * 40:
        errors.append(f"{prefix} commit is all-zero and cannot be evidence")
    git_root = _git_root(root)
    if git_root is None:
        return errors
    commit_object = _git(git_root, "cat-file", "-e", f"{commit}^{{commit}}")
    if commit_object is None or commit_object.returncode != 0:
        errors.append(f"{prefix} commit does not exist: {commit}")
        return errors
    branch_ref = _resolve_branch_ref(git_root, branch)
    if branch_ref is None:
        errors.append(f"{prefix} branch ref does not exist: {branch}")
    else:
        reachable = _git(git_root, "merge-base", "--is-ancestor", commit, branch_ref)
        if reachable is None or reachable.returncode != 0:
            errors.append(f"{prefix} commit is not reachable from branch {branch}")
    return errors


def _check_commit_path(
    root: Path, branch: str, commit: str, path: str, prefix: str
) -> list[str]:
    errors = _check_commit_reference(root, branch, commit, prefix)
    git_root = _git_root(root)
    if git_root is None or any("does not exist" in error for error in errors):
        return errors
    clean_path = path.rstrip("/")
    if not clean_path or Path(clean_path).is_absolute() or ".." in Path(clean_path).parts:
        errors.append(f"{prefix} evidence path is not repository-relative: {path!r}")
    else:
        path_object = _git(git_root, "cat-file", "-e", f"{commit}:{clean_path}")
        if path_object is None or path_object.returncode != 0:
            errors.append(f"{prefix} path does not exist at {commit}: {path}")
    return errors


def check_registry(root: Path) -> list[str]:
    registry, errors = _load_json(root, "docs/experiments/registry.json")
    schema, schema_errors = _load_json(root, "docs/experiments/schema.json")
    errors.extend(schema_errors)
    if registry is None or schema is None:
        return errors
    errors.extend(_schema_error_strings(registry, schema))
    if not isinstance(registry, dict) or not isinstance(schema, dict):
        return errors
    if registry.get("schema") != SCHEMA_MARKER:
        errors.append(f"registry schema marker is not {SCHEMA_MARKER}")
    if schema.get("$id") != SCHEMA_MARKER:
        errors.append(f"schema $id is not {SCHEMA_MARKER}")
    if registry.get("repository") != REPOSITORY:
        errors.append(f"registry repository must be {REPOSITORY}")
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
            "id", "title", "status", "claim_class", "branch", "commit", "evidence",
            "metric", "notes", "is_current_best", "end_to_end", "optimization_claim",
        }
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")
        experiment_id = row.get("id")
        if not isinstance(experiment_id, str) or not ID_RE.fullmatch(experiment_id):
            errors.append(f"{prefix} has malformed id: {experiment_id!r}")
        elif experiment_id in ids:
            errors.append(f"duplicate experiment id: {experiment_id}")
        else:
            ids.add(experiment_id)
        status = row.get("status")
        if isinstance(status, str):
            statuses.add(status)
        if status not in REQUIRED_STATUSES:
            errors.append(f"{prefix} has unsupported status: {status!r}")
        claim_class = row.get("claim_class")
        if claim_class not in REQUIRED_CLAIM_CLASSES:
            errors.append(f"{prefix} has unsupported claim_class: {claim_class!r}")
        commit = row.get("commit")
        if not isinstance(commit, str) or not SHA_RE.fullmatch(commit):
            errors.append(f"{prefix} commit is not a full lowercase SHA-1")
        if not isinstance(row.get("branch"), str) or not row.get("branch"):
            errors.append(f"{prefix} branch is empty")
        if not isinstance(row.get("evidence"), list) or not row.get("evidence"):
            errors.append(f"{prefix} evidence must be a non-empty list")
        if not isinstance(row.get("metric"), dict):
            errors.append(f"{prefix} metric must be an object")
        for boolean_key in ("is_current_best", "end_to_end", "optimization_claim"):
            if not isinstance(row.get(boolean_key), bool):
                errors.append(f"{prefix} {boolean_key} must be boolean")
        metric = row.get("metric")
        if isinstance(metric, dict):
            for metric_name, metric_value in metric.items():
                if not (metric_name.endswith(("_tok_s", "_ms")) or "bandwidth" in metric_name):
                    continue
                if metric_value is None:
                    continue
                if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                    errors.append(f"{prefix} metric {metric_name} must be numeric or null")
                elif not math.isfinite(metric_value) or metric_value <= 0:
                    errors.append(f"{prefix} metric {metric_name} must be positive and finite")
        if claim_class == "full_model" and row.get("end_to_end") is not True:
            errors.append(f"{prefix} full_model claim must be end_to_end")
        if claim_class == "diagnostic" and row.get("end_to_end") is not False:
            errors.append(f"{prefix} diagnostic claim must not be end_to_end")
        if status == "diagnostic" and row.get("optimization_claim") is not False:
            errors.append(f"{prefix} diagnostic status cannot claim optimization")
        if row.get("is_current_best") is True:
            if claim_class != "full_model" or status != "accepted":
                errors.append(f"{prefix} current best must be an accepted full_model claim")
            if row.get("metric", {}).get("exact_quality") is not True:
                errors.append(f"{prefix} current best must carry exact quality evidence")
            if not isinstance(row.get("metric", {}).get("decode_tok_s"), (int, float)) or row["metric"]["decode_tok_s"] <= 0:
                errors.append(f"{prefix} current best must carry a positive decode_tok_s")

    missing_statuses = REQUIRED_STATUSES - statuses
    if missing_statuses:
        errors.append("registry missing statuses: " + ", ".join(sorted(missing_statuses)))

    row_by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    e035 = row_by_id.get("E035")
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
    current_best = [row.get("id") for row in rows if isinstance(row, dict) and row.get("is_current_best")]
    if current_best != ["E035"]:
        errors.append("E035 must be the only current-best row")

    e044 = row_by_id.get("E044")
    if e044 is None:
        errors.append("registry is missing E044")
    elif e044.get("is_current_best") is True or e044.get("status") == "accepted":
        errors.append("E044 must not be current best or accepted")
    e049d = row_by_id.get("E049d-v2")
    if e049d is None:
        errors.append("registry is missing E049d-v2")
    else:
        if e049d.get("status") != "diagnostic":
            errors.append("E049d-v2 must be diagnostic")
        if e049d.get("end_to_end") is not False or e049d.get("optimization_claim") is not False:
            errors.append("E049d-v2 must be neither end-to-end nor optimization")
        if e049d.get("metric", {}).get("steady_marker_tok_s") != 1.11698354:
            errors.append("E049d-v2 marker speed must be exactly 1.11698354")
    e049_nsi = row_by_id.get("E049-nsi")
    if e049_nsi is None:
        errors.append("registry is missing E049-nsi NSI calibration")
    else:
        if e049_nsi.get("branch") != "codex/e049-nsi-calibration":
            errors.append("E049-nsi branch is not the authoritative calibration branch")
        if e049_nsi.get("commit") != EXPECTED_BRANCHES["codex/e049-nsi-calibration"]:
            errors.append("E049-nsi commit is not the authoritative calibration head")
        if e049_nsi.get("status") not in {"diagnostic", "hardware"}:
            errors.append("E049-nsi must be diagnostic or hardware")
        notes = str(e049_nsi.get("notes", "")).lower()
        if not all(marker in notes for marker in ("inconclusive", "low", "mb/s", "saturation")):
            errors.append("E049-nsi notes must bound v4 as inconclusive/low without MB/s or saturation")
        if not any(
            isinstance(item, dict)
            and str(item.get("path", "")).startswith("experiments/E049-nsi-calibration/")
            for item in e049_nsi.get("evidence", [])
        ):
            errors.append("E049-nsi must point to experiments/E049-nsi-calibration")
    return errors


def check_provenance(root: Path) -> list[str]:
    registry, errors = _load_json(root, "docs/experiments/registry.json")
    inventory, inventory_errors = _load_json(root, "docs/experiments/branch-inventory.json")
    errors.extend(inventory_errors)
    if registry is None or inventory is None:
        return errors
    if not isinstance(registry, dict) or not isinstance(inventory, dict):
        return errors + ["registry and branch inventory must be objects"]
    if inventory.get("repository") != REPOSITORY:
        errors.append(f"branch inventory repository must be {REPOSITORY}")
    branches = {
        row.get("branch"): row.get("commit")
        for row in inventory.get("branches", [])
        if isinstance(row, dict)
    }
    for row in registry.get("experiments", []):
        if not isinstance(row, dict):
            continue
        row_id = row.get("id", "<unknown>")
        prefix = f"{row_id} provenance"
        branch = row.get("branch")
        commit = row.get("commit")
        if branch not in branches:
            errors.append(f"{prefix} branch is absent from branch inventory")
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, dict):
                errors.append(f"{prefix} evidence item is not an object")
                continue
            kind = evidence.get("kind")
            path = evidence.get("path")
            if kind not in REQUIRED_EVIDENCE_KINDS:
                errors.append(f"{prefix} has unsupported evidence kind: {kind!r}")
            if not isinstance(path, str) or not path:
                errors.append(f"{prefix} evidence path is missing")
            ref = evidence.get("ref")
            match = REF_RE.fullmatch(ref) if isinstance(ref, str) else None
            if match is None:
                errors.append(f"{prefix} evidence ref is required as branch@full-sha: {ref!r}")
                continue
            if match.group("branch") != branch or match.group("commit") != commit:
                errors.append(f"{prefix} evidence ref does not match row branch/commit")
            if isinstance(path, str):
                errors.extend(_check_commit_path(root, match.group("branch"), match.group("commit"), path, prefix))
    return errors


def _is_english_markdown_text(text: str) -> str | None:
    for character in text:
        if character.isascii() or not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "unknown")
        return f"non-English Unicode letter U+{ord(character):04X} ({name})"
    return None


def check_english_only(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in discover_canonical_markdown(root):
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot read canonical Markdown {relative}: {exc}")
            continue
        violation = _is_english_markdown_text(text)
        if violation:
            errors.append(f"canonical Markdown {relative}: {violation}")
    return errors


def _slugify_heading(heading: str) -> str:
    heading = re.sub(r"[`*_~]", "", heading).strip().lower()
    heading = re.sub(r"[^a-z0-9 _-]", "", heading)
    return re.sub(r"[ _]+", "-", heading).strip("-")


def _fragment_exists(path: Path, fragment: str) -> bool:
    wanted = unquote(fragment).lower()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if re.search(rf"(?:id|name)=[\"']{re.escape(wanted)}[\"']", text, re.IGNORECASE):
        return True
    for line in text.splitlines():
        match = re.match(r"^\s*#+\s+(.+?)\s*#*\s*$", line)
        if match and _slugify_heading(match.group(1)) == wanted:
            return True
    return False


def check_links(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in discover_canonical_markdown(root):
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in LINK_RE.finditer(text):
            target = (match.group(1) or match.group(2) or "").strip()
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            target_path = unquote(parsed.path)
            resolved = (path.parent / target_path).resolve() if target_path else path.resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"link escapes repository: {relative}: {target}")
                continue
            if not resolved.exists():
                errors.append(f"broken link: {relative}: {target}")
                continue
            if parsed.fragment and not _fragment_exists(resolved, parsed.fragment):
                errors.append(f"broken fragment: {relative}: {target}")
    return errors


def check_hardware_facts(root: Path) -> list[str]:
    path = root / "docs/hardware/a733.md"
    if not path.is_file():
        return ["missing hardware fact document: docs/hardware/a733.md"]
    text = path.read_text(encoding="utf-8")
    required_rows = {
        "CPU topology": re.compile(r"\|\s*6x Cortex-A55 plus 2x Cortex-A76\s*\|\s*\*\*Verified upstream\*\*"),
        "memory interface": re.compile(r"\|\s*32-bit external memory interface\s*\|\s*\*\*Verified upstream\*\*"),
        "LPDDR ceiling": re.compile(r"\|\s*LPDDR5-4800 support\s*\|\s*\*\*Verified upstream\*\*"),
        "theoretical bandwidth": re.compile(r"\|\s*19\.2 GB/s.*\|\s*\*\*Derived upstream ceiling\*\*"),
        "controller readback": re.compile(r"\|\s*Current memory-controller readback is 510 MHz\s*\|\s*\*\*Verified on target\*\*"),
        "unknown effective rate": re.compile(r"\|\s*Effective LPDDR data rate\s*\|\s*\*\*Unknown\*\*"),
        "unknown sustained bandwidth": re.compile(r"\|\s*Sustained bandwidth for Bonsai decode\s*\|\s*\*\*Unknown\*\*"),
    }
    errors = [f"hardware fact row is missing or misclassified: {name}" for name, pattern in required_rows.items() if not pattern.search(text)]
    contradictory = re.compile(
        r"(?i)19\.2\s*GB/s[^.\n]{0,100}\b(?:measured\s+sustained|sustained\s+bandwidth|observed\s+bandwidth)\b"
        r"|\b(?:measured\s+sustained|sustained\s+bandwidth|observed\s+bandwidth)\b[^.\n]{0,100}19\.2\s*GB/s"
    )
    if contradictory.search(text):
        errors.append("19.2 GB/s must remain a theoretical ceiling, not measured sustained bandwidth")
    if not re.search(r"secure firmware.*safety", text, re.IGNORECASE | re.DOTALL):
        errors.append("secure-firmware safety boundary is missing or misclassified")
    return errors


def check_branch_inventory(root: Path) -> list[str]:
    inventory, errors = _load_json(root, "docs/experiments/branch-inventory.json")
    if inventory is None:
        return errors
    if not isinstance(inventory, dict):
        return errors + ["branch inventory must be an object"]
    if inventory.get("repository") != REPOSITORY:
        errors.append(f"branch inventory repository must be {REPOSITORY}")
    rows = inventory.get("branches")
    if not isinstance(rows, list):
        return errors + ["branch inventory branches must be a list"]
    actual: dict[str, Any] = {}
    for index, row in enumerate(rows):
        prefix = f"branch inventory row {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix} is not an object")
            continue
        branch, commit = row.get("branch"), row.get("commit")
        if not isinstance(branch, str) or not branch:
            errors.append(f"{prefix} has no branch")
            continue
        if branch in actual:
            errors.append(f"duplicate branch inventory row: {branch}")
        actual[branch] = commit
        if not isinstance(commit, str) or not SHA_RE.fullmatch(commit):
            errors.append(f"{prefix} has invalid commit")
        elif isinstance(branch, str):
            errors.extend(_check_commit_reference(root, branch, commit, prefix))
    if actual != EXPECTED_BRANCHES:
        for label, values in (
            ("missing", sorted(set(EXPECTED_BRANCHES) - set(actual))),
            ("unexpected", sorted(set(actual) - set(EXPECTED_BRANCHES))),
            ("changed", sorted(branch for branch in set(actual) & set(EXPECTED_BRANCHES) if actual[branch] != EXPECTED_BRANCHES[branch])),
        ):
            if values:
                errors.append(f"branch inventory {label}: " + ", ".join(values))
    if "point-in-time snapshot" not in str(inventory.get("source", "")).lower():
        errors.append("branch inventory must identify a point-in-time snapshot")
    e055 = next((row for row in rows if isinstance(row, dict) and row.get("branch") == "codex/e055-q1-hot-cold"), None)
    if e055 is None or "ongoing" not in str(e055.get("role", "")).lower():
        errors.append("E055 inventory row must remain an ongoing-work snapshot caveat")
    return errors


def _inventory_branch_ref(root: Path, branch: str) -> str | None:
    git_root = _git_root(root)
    return _resolve_branch_ref(git_root, branch) if git_root else None


def check_experiment_coverage(root: Path) -> list[str]:
    inventory, errors = _load_json(root, "docs/experiments/branch-inventory.json")
    registry, registry_errors = _load_json(root, "docs/experiments/registry.json")
    errors.extend(registry_errors)
    if not isinstance(inventory, dict) or not isinstance(registry, dict):
        return errors
    indexed_dirs: dict[str, str] = {}
    for row in registry.get("experiments", []):
        if not isinstance(row, dict):
            continue
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            path = str(evidence.get("path", ""))
            if path.startswith("experiments/"):
                parts = path.split("/")
                if len(parts) >= 2 and parts[1]:
                    indexed_dirs[parts[1]] = str(row.get("id", ""))
    git_root = _git_root(root)
    if git_root is None:
        return errors + ["cannot inspect inventoried experiment directories without Git"]
    for branch_row in inventory.get("branches", []):
        if not isinstance(branch_row, dict):
            continue
        branch = branch_row.get("branch")
        branch_ref = _resolve_branch_ref(git_root, branch) if isinstance(branch, str) else None
        if branch_ref is None:
            errors.append(f"cannot inspect experiment directories for branch {branch!r}")
            continue
        listing = _git(git_root, "ls-tree", "-d", "--name-only", branch_ref, "experiments/")
        if listing is None or listing.returncode != 0:
            errors.append(f"cannot list experiments for branch {branch}")
            continue
        for line in listing.stdout.splitlines():
            directory = line.removeprefix("experiments/").strip("/")
            if not directory:
                continue
            if directory in EXCLUDED_EXPERIMENT_DIRECTORIES:
                if not EXCLUDED_EXPERIMENT_DIRECTORIES[directory].strip():
                    errors.append(f"experiment directory exclusion has no reason: {directory}")
            elif directory not in indexed_dirs:
                errors.append(f"experiment directory is not represented in registry: {branch}: {directory}")
    return errors


def check_privacy(root: Path) -> list[str]:
    errors: list[str] = []
    publication_candidates = set(_publication_candidates(root))
    for path in _candidate_file_paths(root):
        relative = _relative_path(root, path)
        candidate = path in publication_candidates
        if _privacy_allowlisted(relative) and not candidate:
            continue
        labels = _scan_file_secret_labels(path)
        for label in sorted(labels):
            # Never include a matched value in diagnostics.
            errors.append(f"privacy scanner found {label} in {relative}")
    return errors


def check_public_artifact_manifest(root: Path) -> list[str]:
    manifest, errors = _load_json(root, "docs/experiments/public-artifact-manifest.json")
    if manifest is None:
        return errors
    if not isinstance(manifest, dict):
        return errors + ["public artifact manifest must be an object"]
    if manifest.get("schema") != MANIFEST_MARKER:
        errors.append(f"public artifact manifest schema must be {MANIFEST_MARKER}")
    if manifest.get("repository") != REPOSITORY:
        errors.append(f"public artifact manifest repository must be {REPOSITORY}")
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        errors.append("public artifact manifest policy must be an object")
    else:
        if not isinstance(policy.get("allowed_public_artifacts"), list) or not policy.get("allowed_public_artifacts"):
            errors.append("manifest policy must list allowed public artifact classes")
        if not isinstance(policy.get("prohibited_data"), list) or not policy.get("prohibited_data"):
            errors.append("manifest policy must list prohibited personal/sensitive data")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return errors + ["public artifact manifest artifacts must be a list"]
    git_root = _git_root(root)
    for index, artifact in enumerate(artifacts):
        prefix = f"artifact {index}"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} is not an object")
            continue
        required = {"path", "sha256", "byte_size", "origin", "source_commit", "build_runtime_toolchain", "destination"}
        missing = sorted(required - artifact.keys())
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")
            continue
        path_value = artifact["path"]
        source_commit = artifact["source_commit"]
        if not isinstance(path_value, str) or not path_value or ".." in Path(path_value).parts:
            errors.append(f"{prefix} has unsafe path")
            continue
        digest = artifact["sha256"]
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"{prefix} sha256 is not a 64-character lowercase digest")
        elif digest in PLACEHOLDER_DIGESTS or len(set(digest)) == 1:
            errors.append(f"{prefix} sha256 is zero or a placeholder, not nonzero evidence")
        byte_size = artifact["byte_size"]
        if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size <= 0:
            errors.append(f"{prefix} byte_size must be a positive exact size")
        if not isinstance(artifact["origin"], str) or not artifact["origin"].strip():
            errors.append(f"{prefix} origin must be nonempty")
        if not isinstance(source_commit, str) or not SHA_RE.fullmatch(source_commit) or source_commit == "0" * 40:
            errors.append(f"{prefix} source_commit must be a nonzero full lowercase SHA-1")
        if not isinstance(artifact["build_runtime_toolchain"], str) or not artifact["build_runtime_toolchain"].strip():
            errors.append(f"{prefix} build/runtime/toolchain provenance must be nonempty")
        if artifact["destination"] not in {"git", "release-assets", "external"}:
            errors.append(f"{prefix} destination is invalid")
        if artifact["destination"] == "git":
            path = root / path_value
            if not path.is_file():
                errors.append(f"{prefix} Git destination does not exist: {path_value}")
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                size = path.stat().st_size
                if digest != artifact["sha256"]:
                    errors.append(f"{prefix} sha256 does not match {path_value}")
                if size != byte_size:
                    errors.append(f"{prefix} byte_size does not match {path_value}")
        if git_root and isinstance(source_commit, str) and SHA_RE.fullmatch(source_commit):
            result = _git(git_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
            if result is None or result.returncode != 0:
                errors.append(f"{prefix} source_commit does not exist: {source_commit}")
            elif artifact["destination"] == "git":
                source_path = _git(git_root, "cat-file", "-e", f"{source_commit}:{path_value}")
                if source_path is None or source_path.returncode != 0:
                    errors.append(f"{prefix} path is absent at source_commit: {path_value}")
    return errors


def check_manifest_coverage(root: Path) -> list[str]:
    manifest, errors = _load_json(root, "docs/experiments/public-artifact-manifest.json")
    if not isinstance(manifest, dict):
        return errors
    listed = {
        artifact.get("path")
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }
    for path in _publication_candidates(root):
        relative = _relative_path(root, path)
        if relative not in listed:
            errors.append(f"new public artifact candidate is not listed in manifest: {relative}")
    return errors


def discover_policy_documents(root: Path) -> tuple[Path, ...]:
    """Find policy-bearing documents instead of relying on a frozen file list."""

    candidates = set(discover_canonical_markdown(root))
    candidates.update(POLICY_FILES)
    candidates.add(Path("docs/toolchain/inventory.md"))
    markers = ("may be published", "personal/sensitive data", "public artifact")
    documents = []
    for relative in candidates:
        path = root / relative
        if path.is_file() and any(marker in path.read_text(encoding="utf-8").lower() for marker in markers):
            documents.append(relative)
    return tuple(sorted(set(documents), key=lambda item: item.as_posix()))


def check_artifact_policy(root: Path) -> list[str]:
    errors: list[str] = []
    required_markers = ("may be published", "personal/sensitive data", "private keys")
    for relative in POLICY_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing artifact policy file: {relative}")
            continue
        text = path.read_text(encoding="utf-8").lower()
        for marker in required_markers:
            if marker not in text:
                errors.append(f"{relative} is missing approved artifact-policy wording: {marker}")
    for relative in discover_policy_documents(root):
        text = (root / relative).read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_POLICY_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative} contains a {label}; use the approved manifest-backed policy")
    if not (root / ".gitattributes").is_file():
        errors.append("missing .gitattributes Git LFS guidance")
    if not (root / "docs/experiments/public-artifacts.md").is_file():
        errors.append("missing release-assets and public-artifact guidance")
    return errors


def run_checks(root: Path) -> list[str]:
    checks = (
        check_registry,
        check_provenance,
        check_english_only,
        check_links,
        check_hardware_facts,
        check_branch_inventory,
        check_experiment_coverage,
        check_artifact_policy,
        check_public_artifact_manifest,
        check_manifest_coverage,
        check_privacy,
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
