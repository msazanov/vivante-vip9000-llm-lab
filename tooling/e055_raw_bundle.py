"""Git-sealed raw artifact ingestion for E055.

This module treats the current Git commit as a byte-immutability boundary. It
does not claim that Git proves device truth. Statistical qualification is built
only after raw files pass this sealing layer.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping


BUNDLE_SCHEMA = "e055-raw-bundle/v1"
PHASE_PREFIX = PurePosixPath("experiments/E055-q1-hot-cold/raw")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
ROLE_SIZE_LIMITS = {
    "harness_executable": 32 * 1024 * 1024,
    "harness_stdout": 256 * 1024,
    "harness_stderr": 1024 * 1024,
    "e049c_json": 256 * 1024,
    "e049c_stderr": 1024 * 1024,
    "runner_metadata": 256 * 1024,
}
RUN_ROLES = {
    "harness_stdout", "harness_stderr", "e049c_json",
    "e049c_stderr", "runner_metadata",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class SealedArtifact:
    """One regular file whose worktree, index, and HEAD bytes agree."""

    relative_path: str
    role: str
    sha256: str
    size_bytes: int
    git_blob_oid: str
    payload: bytes


@dataclass(frozen=True)
class SealedBundleFiles:
    """The immutable file layer of one E055 raw bundle."""

    repository_root: Path
    phase_root: str
    commit: str
    tree: str
    manifest: SealedArtifact
    artifacts: tuple[SealedArtifact, ...]
    document: Mapping[str, Any]


@dataclass(frozen=True)
class RawArtifactEvidence:
    role: str
    relative_path: str
    sha256: str
    size_bytes: int
    git_blob_oid: str


@dataclass(frozen=True)
class DerivedSample:
    """A measurement row constructed only from sealed raw artifact bytes."""

    run_id: str
    pair_id: str
    pair_index: int
    pair_order: str
    order_index: int
    build_name: str
    mode: str
    cache_state: str
    cpu: int
    target_working_set_bytes: int
    actual_working_set_bytes: int
    blocks: int
    iterations: int
    calls: int
    elapsed_ns: int
    checksum: str
    pmu_group: str
    publication_identity: tuple[tuple[str, Any], ...]
    commit: str
    tree: str
    manifest_path: str
    manifest_blob_oid: str
    raw_artifacts: tuple[RawArtifactEvidence, ...]


@dataclass(frozen=True)
class SealedBundle:
    files: SealedBundleFiles
    samples: tuple[DerivedSample, ...]


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace") if binary else result.stderr
        raise ValueError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("artifact path must be a bounded nonempty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError("artifact path contains a backslash or control character")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(
        part in ("", ".", "..") for part in parsed.parts
    ):
        raise ValueError("artifact path must be a canonical repository-relative path")
    return parsed


def _read_regular_file(path: Path, maximum: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open regular file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"artifact is not a single-link regular file: {path}")
        if before.st_size < 1 or before.st_size > maximum:
            raise ValueError(f"artifact exceeds its role size bound: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"artifact was truncated while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"artifact was appended while reading: {path}")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size
        ):
            raise ValueError(f"artifact changed while reading: {path}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _repository_root(manifest: Path) -> Path:
    absolute = Path(os.path.abspath(manifest))
    output = _git(absolute.parent, "rev-parse", "--show-toplevel")
    assert isinstance(output, str)
    root = Path(output.strip())
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError("bundle manifest is outside its Git worktree") from exc
    return root


def _git_entry(root: Path, source: str, relative: str) -> tuple[str, str]:
    if source == "HEAD":
        output = _git(root, "ls-tree", "HEAD", "--", relative)
    else:
        output = _git(root, "ls-files", "--stage", "--", relative)
    assert isinstance(output, str)
    lines = [line for line in output.splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"{relative} is not one exact tracked {source} entry")
    metadata, tab, path = lines[0].partition("\t")
    if tab != "\t" or path != relative:
        raise ValueError(f"Git returned an aliased path for {relative}")
    fields = metadata.split()
    if source == "HEAD":
        if len(fields) != 3 or fields[1] != "blob":
            raise ValueError(f"{relative} is not a regular HEAD blob")
        mode, _, oid = fields
    else:
        if len(fields) != 3 or fields[2] != "0":
            raise ValueError(f"{relative} has a non-stage-zero index entry")
        mode, oid, _ = fields
    if mode not in ("100644", "100755") or GIT_OID_RE.fullmatch(oid) is None:
        raise ValueError(f"{relative} has an unsupported Git mode or object ID")
    return mode, oid


def _seal_file(
    root: Path,
    relative: str,
    role: str,
    maximum: int,
    declared: Mapping[str, Any] | None,
) -> tuple[SealedArtifact, tuple[int, int]]:
    path = root / relative
    payload, status = _read_regular_file(path, maximum)
    head_mode, head_oid = _git_entry(root, "HEAD", relative)
    index_mode, index_oid = _git_entry(root, "INDEX", relative)
    if (head_mode, head_oid) != (index_mode, index_oid):
        raise ValueError(f"index and HEAD differ for sealed artifact {relative}")
    committed = _git(root, "cat-file", "blob", head_oid, binary=True)
    assert isinstance(committed, bytes)
    if committed != payload:
        raise ValueError(f"worktree bytes differ from HEAD for {relative}")
    digest = _sha256(payload)
    if declared is not None:
        if set(declared) != {"role", "path", "sha256", "size_bytes", "git_blob_oid"}:
            raise ValueError(f"artifact declaration has unexpected fields: {relative}")
        if declared.get("role") != role or declared.get("path") != relative:
            raise ValueError(f"artifact role/path binding mismatch: {relative}")
        if not isinstance(declared.get("sha256"), str) or \
                SHA256_RE.fullmatch(declared["sha256"]) is None or \
                declared["sha256"] != digest:
            raise ValueError(f"artifact SHA-256 mismatch: {relative}")
        if not _is_int(declared.get("size_bytes")) or \
                declared["size_bytes"] != len(payload):
            raise ValueError(f"artifact size mismatch: {relative}")
        if declared.get("git_blob_oid") != head_oid:
            raise ValueError(f"artifact Git blob mismatch: {relative}")
    return (
        SealedArtifact(relative, role, digest, len(payload), head_oid, payload),
        (status.st_dev, status.st_ino),
    )


def _declared_artifacts(document: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    builds = document.get("build_artifacts")
    runs = document.get("runs")
    if not isinstance(builds, list) or not builds or not isinstance(runs, list) or not runs:
        raise ValueError("raw bundle must declare nonempty builds and runs")
    declared: list[tuple[str, Mapping[str, Any]]] = []
    build_names: set[str] = set()
    for build in builds:
        if not isinstance(build, Mapping) or set(build) != {"build_name", "artifact"}:
            raise ValueError("build artifact entry is malformed")
        name = build.get("build_name")
        artifact = build.get("artifact")
        if name not in ("O3", "O3-flto") or name in build_names or \
                not isinstance(artifact, Mapping):
            raise ValueError("build names must be unique O3/O3-flto values")
        build_names.add(str(name))
        declared.append(("harness_executable", artifact))
    run_ids: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("run declaration must be an object")
        run_id = run.get("run_id")
        artifacts = run.get("artifacts")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise ValueError("run_id must be a unique nonempty string")
        run_ids.add(run_id)
        if not isinstance(artifacts, Mapping) or set(artifacts) != RUN_ROLES:
            raise ValueError("run must declare every exact raw artifact role")
        for role in sorted(RUN_ROLES):
            artifact = artifacts[role]
            if not isinstance(artifact, Mapping):
                raise ValueError(f"{role} artifact declaration must be an object")
            declared.append((role, artifact))
    return declared


def _phase_files(root: Path, phase_relative: PurePosixPath) -> tuple[set[str], list[Path]]:
    phase = root / phase_relative.as_posix()
    files: set[str] = set()
    symlinks: list[Path] = []
    for directory, directories, names in os.walk(phase, followlinks=False):
        directory_path = Path(directory)
        for name in list(directories):
            candidate = directory_path / name
            if candidate.is_symlink():
                symlinks.append(candidate)
        for name in names:
            candidate = directory_path / name
            files.add(candidate.relative_to(root).as_posix())
            if candidate.is_symlink():
                symlinks.append(candidate)
    return files, symlinks


def seal_bundle_files(manifest_path: str | Path) -> SealedBundleFiles:
    """Seal a committed manifest and all declared non-manifest raw artifacts."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError("E055 evidence must be a committed raw-bundle manifest path")
    manifest_input = Path(manifest_path)
    root = _repository_root(manifest_input)
    manifest_absolute = Path(os.path.abspath(manifest_input))
    manifest_relative = manifest_absolute.relative_to(root).as_posix()
    canonical_manifest = _canonical_relative_path(manifest_relative)
    if canonical_manifest.name != "bundle.json" or \
            canonical_manifest.parts[:len(PHASE_PREFIX.parts)] != PHASE_PREFIX.parts or \
            len(canonical_manifest.parts) != len(PHASE_PREFIX.parts) + 2:
        raise ValueError("manifest must be one canonical E055 raw phase bundle.json")
    phase_relative = canonical_manifest.parent

    status_output = _git(root, "status", "--porcelain=v2", "--", phase_relative.as_posix())
    assert isinstance(status_output, str)
    if status_output:
        raise ValueError("raw phase has dirty, staged, deleted, or untracked files")

    manifest_sealed, manifest_inode = _seal_file(
        root, manifest_relative, "bundle_manifest", MAX_MANIFEST_BYTES, None
    )
    try:
        document = json.loads(manifest_sealed.payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"raw bundle manifest is not exact UTF-8 JSON: {exc}") from exc
    if not isinstance(document, Mapping) or document.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("raw bundle manifest schema is not e055-raw-bundle/v1")

    sealed: list[SealedArtifact] = []
    paths: set[str] = {manifest_relative}
    blobs: set[str] = set()
    inodes: set[tuple[int, int]] = {manifest_inode}
    for role, declaration in _declared_artifacts(document):
        parsed = _canonical_relative_path(declaration.get("path"))
        if parsed.parts[:len(phase_relative.parts)] != phase_relative.parts:
            raise ValueError("every raw artifact must remain under its phase root")
        relative = parsed.as_posix()
        if relative in paths:
            raise ValueError(f"artifact path is reused across roles: {relative}")
        paths.add(relative)
        artifact, inode = _seal_file(
            root, relative, role, ROLE_SIZE_LIMITS[role], declaration
        )
        if artifact.git_blob_oid in blobs:
            raise ValueError("one Git blob cannot be reused by multiple raw roles")
        if inode in inodes:
            raise ValueError("hardlink alias detected across raw bundle files")
        blobs.add(artifact.git_blob_oid)
        inodes.add(inode)
        sealed.append(artifact)

    discovered, symlinks = _phase_files(root, phase_relative)
    if symlinks:
        raise ValueError("raw phase contains a symlink")
    if discovered != paths:
        raise ValueError("raw phase contains missing or unmanifested files")

    commit_output = _git(root, "rev-parse", "HEAD")
    tree_output = _git(root, "rev-parse", "HEAD^{tree}")
    assert isinstance(commit_output, str) and isinstance(tree_output, str)
    return SealedBundleFiles(
        repository_root=root,
        phase_root=phase_relative.as_posix(),
        commit=commit_output.strip(),
        tree=tree_output.strip(),
        manifest=manifest_sealed,
        artifacts=tuple(sealed),
        document=document,
    )


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain the exact documented fields")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _zero_int(value: Any, label: str) -> int:
    if not _is_int(value) or value != 0:
        raise ValueError(f"{label} must be integer zero")
    return value


def _finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite runtime float")
    return value


def _parse_json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not one exact UTF-8 JSON object: {exc}") from exc
    if text[end:].strip() or not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain exactly one JSON object")
    return value


def _decode_stream(
    artifact: SealedArtifact,
    *,
    run_id: str,
    producer: str,
    stream: str,
) -> bytes:
    envelope = _exact_keys(
        _parse_json_object(artifact.payload, f"{artifact.role} envelope"),
        {
            "schema", "run_id", "producer", "stream", "encoding",
            "payload_size_bytes", "payload_sha256", "payload_base64",
        },
        f"{artifact.role} envelope",
    )
    if envelope.get("schema") != "e055-stream-capture/v1" or \
            envelope.get("run_id") != run_id or envelope.get("producer") != producer or \
            envelope.get("stream") != stream or envelope.get("encoding") != "base64":
        raise ValueError(f"{artifact.role} stream identity mismatch")
    encoded = envelope.get("payload_base64")
    if not isinstance(encoded, str):
        raise ValueError(f"{artifact.role} payload_base64 must be a string")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{artifact.role} has invalid Base64: {exc}") from exc
    size = envelope.get("payload_size_bytes")
    digest = envelope.get("payload_sha256")
    if not _is_int(size) or size < 0 or size != len(payload) or \
            not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or \
            digest != _sha256(payload):
        raise ValueError(f"{artifact.role} decoded payload hash/size mismatch")
    return payload


def _publication_contract() -> tuple[Mapping[str, Any], str]:
    # Local import avoids an import cycle when e055_q1_hotcold exposes the
    # bundle-only promotion entry point.
    from tooling.e055_q1_hotcold import PUBLICATION_MANIFEST, load_publication_contract

    contract = load_publication_contract()
    return contract, _sha256(PUBLICATION_MANIFEST.read_bytes())


def canonical_harness_argv(
    cell: Mapping[str, Any], executable_path: str, iterations: int,
) -> tuple[str, ...]:
    """Return the only command sequence accepted for a qualified raw run."""

    arguments = [
        "taskset", "-c", str(cell["cpu"]), executable_path,
        "--mode", str(cell["mode"]),
        "--cache-state", str(cell["cache_state"]),
        "--working-set-bytes", str(cell["target_working_set_bytes"]),
        "--cpu", str(cell["cpu"]),
        "--iterations", str(iterations),
    ]
    if cell["cache_state"] == "cold_conditioned":
        arguments.extend(("--budget-ms", "0", "--warmup", "0"))
    else:
        arguments.extend(("--warmup", "16"))
    arguments.extend(("--thrash-bytes", str(64 * 1024 * 1024), "--sync"))
    return tuple(arguments)


def _validate_qualification(
    qualification: Any, contract: Mapping[str, Any], publication_sha256: str,
) -> tuple[tuple[str, Any], ...]:
    expected = {
        "source_sha256": contract.get("source_sha256"),
        "compiler_sha256": contract.get("compiler_sha256"),
        "compiler_id": contract.get("compiler_id"),
        "pmu_source_sha256": contract.get("pmu_source_sha256"),
        "upstream_commit": contract.get("upstream_commit"),
        "upstream_ref": contract.get("upstream_ref"),
        "upstream_repack_sha256": contract.get("upstream_repack_sha256"),
        "publication_manifest_sha256": publication_sha256,
    }
    exact = _exact_keys(qualification, set(expected), "bundle qualification")
    if dict(exact) != expected:
        raise ValueError("bundle qualification does not match publication artifacts")
    return tuple(sorted(expected.items()))


def _validate_manifest_shape(document: Mapping[str, Any]) -> None:
    _exact_keys(
        document,
        {
            "schema", "experiment", "phase_id", "qualification",
            "build_artifacts", "runs", "target_workload_executed",
        },
        "raw bundle manifest",
    )
    if document.get("schema") != BUNDLE_SCHEMA or \
            document.get("experiment") != "E055-Q1-HOT-COLD" or \
            document.get("target_workload_executed") is not True:
        raise ValueError("raw bundle manifest identity/status is invalid")
    phase_id = document.get("phase_id")
    if not isinstance(phase_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", phase_id) is None:
        raise ValueError("phase_id is not canonical")
    runs = document.get("runs")
    if not isinstance(runs, list) or not 1 <= len(runs) <= 1260:
        raise ValueError("raw bundle must contain between one and 1260 runs")


def _validate_cell(cell: Any) -> Mapping[str, Any]:
    exact = _exact_keys(
        cell,
        {
            "mode", "cache_state", "cpu", "target_working_set_bytes",
            "actual_working_set_bytes", "blocks", "pmu_group",
        },
        "run cell",
    )
    if exact.get("mode") not in ("packed_stream", "unpack_scale", "full_dotprod") or \
            exact.get("cache_state") not in ("hot_repeat", "cold_conditioned") or \
            not _is_int(exact.get("cpu")) or exact.get("cpu") not in (0, 6) or \
            exact.get("pmu_group") not in ("core", "cache", "memory"):
        raise ValueError("run cell mode/cache/CPU/PMU identity is invalid")
    target = _positive_int(exact.get("target_working_set_bytes"), "target working set")
    blocks = _positive_int(exact.get("blocks"), "native blocks")
    actual = _positive_int(exact.get("actual_working_set_bytes"), "actual working set")
    if blocks != (target + 207) // 208 or actual != blocks * 208:
        raise ValueError("run cell does not match the exact 208-byte carrier layout")
    return exact


def _validate_harness(
    raw: Mapping[str, Any], run: Mapping[str, Any], cell: Mapping[str, Any],
) -> tuple[int, int, int, str]:
    exact = _exact_keys(
        raw,
        {
            "schema", "mode", "cache_state", "cpu", "q1_layout", "golden_pass",
            "golden_cases", "target_working_set_bytes", "actual_working_set_bytes",
            "blocks", "iterations", "calls", "elapsed_ns", "first_call_ns",
            "calls_per_second", "logical_bytes_per_call", "checksum",
            "cold_conditioning", "sync", "qualification",
        },
        "E055 harness stdout",
    )
    cross = {
        "mode": cell["mode"], "cache_state": cell["cache_state"], "cpu": cell["cpu"],
        "target_working_set_bytes": cell["target_working_set_bytes"],
        "actual_working_set_bytes": cell["actual_working_set_bytes"],
        "blocks": cell["blocks"],
    }
    if any(exact.get(key) != value for key, value in cross.items()):
        raise ValueError("E055 stdout does not match its manifest cell")
    if exact.get("schema") != "e055-q1-hot-cold-harness/v1" or \
            exact.get("q1_layout") != "E039 stock native block_q1_0x4 4x4 DOTPROD" or \
            exact.get("golden_pass") is not True or exact.get("golden_cases") != 18 or \
            not _is_int(exact.get("golden_cases")) or \
            exact.get("qualification") != "unqualified_harness_output_requires_E049c_join":
        raise ValueError("E055 stdout lacks the exact build-bound 18-case golden")
    iterations = _positive_int(exact.get("iterations"), "harness iterations")
    calls = _positive_int(exact.get("calls"), "harness calls")
    elapsed = _positive_int(exact.get("elapsed_ns"), "harness elapsed_ns")
    _positive_int(exact.get("first_call_ns"), "harness first_call_ns")
    if iterations != calls:
        raise ValueError("harness iterations and calls differ")
    if cell["cache_state"] == "cold_conditioned" and calls != 1:
        raise ValueError("cold_conditioned raw capture must contain exactly one call")
    rate = _finite_float(exact.get("calls_per_second"), "harness calls_per_second")
    if not math.isclose(rate, calls * 1.0e9 / elapsed, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("harness calls_per_second does not match calls/elapsed_ns")
    checksum = exact.get("checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"0x[0-9a-f]+", checksum) is None or \
            int(checksum, 16) == 0:
        raise ValueError("harness checksum must be nonzero lowercase hexadecimal")
    logical = _exact_keys(
        exact.get("logical_bytes_per_call"),
        {"q1_packed_bytes", "q8_bytes", "total_input_bytes", "output_bytes", "dot_products"},
        "logical byte counts",
    )
    blocks = cell["blocks"]
    expected_logical = {
        "q1_packed_bytes": blocks * 72,
        "q8_bytes": blocks * 4 * 34,
        "total_input_bytes": blocks * 208,
        "output_bytes": 16,
        "dot_products": blocks * 512,
    }
    if dict(logical) != expected_logical:
        raise ValueError("harness logical byte/operation counts are inconsistent")
    conditioning = _exact_keys(
        exact.get("cold_conditioning"),
        {
            "strategy", "requested_bytes", "actual_bytes", "line_bytes",
            "lines_touched", "checksum", "verified_touched", "warmup_calls",
        },
        "cache conditioning",
    )
    for key in ("requested_bytes", "actual_bytes", "line_bytes", "lines_touched"):
        _positive_int(conditioning.get(key), f"conditioning {key}")
    if conditioning.get("verified_touched") is not True or conditioning.get("line_bytes") != 64:
        raise ValueError("cache conditioning is not exactly verified at 64-byte lines")
    condition_checksum = conditioning.get("checksum")
    if not isinstance(condition_checksum, str) or \
            re.fullmatch(r"0x[0-9a-f]+", condition_checksum) is None or \
            int(condition_checksum, 16) == 0:
        raise ValueError("conditioning checksum is invalid")
    if cell["cache_state"] == "hot_repeat":
        if conditioning.get("strategy") != "verified_kernel_warmup" or \
                not _is_int(conditioning.get("warmup_calls")) or \
                conditioning["warmup_calls"] <= 0 or \
                conditioning.get("actual_bytes") != cell["actual_working_set_bytes"] or \
                conditioning.get("requested_bytes") != cell["actual_working_set_bytes"] or \
                conditioning.get("lines_touched") != (cell["actual_working_set_bytes"] + 63) // 64:
            raise ValueError("hot conditioning does not cover the exact working set")
    else:
        if conditioning.get("strategy") != "verified_write_read_each_64B_line" or \
                conditioning.get("warmup_calls") != 0 or \
                conditioning.get("requested_bytes") != conditioning.get("actual_bytes") or \
                conditioning.get("actual_bytes") % 64 != 0 or \
                conditioning.get("lines_touched") * 64 != conditioning.get("actual_bytes"):
            raise ValueError("cold conditioning does not cover every requested cache line")
    sync = _exact_keys(
        exact.get("sync"), {"requested", "started", "acknowledged", "ended", "sequence"},
        "harness sync",
    )
    if sync != {
        "requested": True, "started": True, "acknowledged": True,
        "ended": True, "sequence": "S/A/E",
    }:
        raise ValueError("harness did not complete exact S/ACK/E synchronization")
    return iterations, calls, elapsed, checksum


def _validate_e049c(
    raw: Mapping[str, Any], cell: Mapping[str, Any], argv: tuple[str, ...],
    harness_elapsed_ns: int, contract: Mapping[str, Any],
) -> None:
    exact = _exact_keys(
        raw,
        {
            "schema_version", "status", "sample_valid", "event_source",
            "counter_semantics", "event_group", "software_group_size_limit",
            "event_group_size", "software_group_size_limit_semantics",
            "min_running_ratio", "pid", "process_group", "command", "sync",
            "measured_elapsed_ns", "thermal", "exit", "failure_reason", "events",
        },
        "E049c raw JSON",
    )
    if exact.get("schema_version") != "e049c-arm-pmu/v2" or \
            exact.get("status") != "ok" or exact.get("sample_valid") is not True or \
            exact.get("event_source") != "armv8_pmuv3_raw_config" or \
            exact.get("counter_semantics") != "event counts only; no DDR-byte conversion" or \
            exact.get("event_group") != cell["pmu_group"] or \
            exact.get("failure_reason") is not None:
        raise ValueError("E049c status/source/group/count semantics are invalid")
    if exact.get("command") != list(argv):
        raise ValueError("E049c command does not match exact runner argv")
    if type(exact.get("min_running_ratio")) is not float or \
            exact.get("min_running_ratio") != 0.95:
        raise ValueError("E049c launcher floor must remain the explicit float 0.95")
    if not _is_int(exact.get("software_group_size_limit")) or \
            exact.get("software_group_size_limit") != 4:
        raise ValueError("E049c software group-size limit is invalid")
    if not _is_int(exact.get("pid")) or exact.get("pid") <= 0 or \
            not _is_int(exact.get("process_group")) or exact.get("process_group") <= 0:
        raise ValueError("E049c PID/process group must be positive integers")
    sync = _exact_keys(
        exact.get("sync"), {"mode", "started", "acknowledged", "ended"}, "E049c sync"
    )
    if sync != {"mode": "start_ack_end", "started": True,
                "acknowledged": True, "ended": True}:
        raise ValueError("E049c did not observe exact S/ACK/E synchronization")
    measured = _positive_int(exact.get("measured_elapsed_ns"), "E049c measured elapsed")
    if measured < harness_elapsed_ns:
        raise ValueError("E049c timing window is shorter than raw harness timing")
    thermal = _exact_keys(
        exact.get("thermal"),
        {"guard_enabled", "checked", "readable", "limit_c", "max_observed_c", "tripped"},
        "E049c thermal",
    )
    if thermal.get("guard_enabled") is not True or thermal.get("checked") is not True or \
            thermal.get("readable") is not True or thermal.get("tripped") is not False:
        raise ValueError("E049c thermal gate did not pass")
    limit = _finite_float(thermal.get("limit_c"), "thermal limit")
    maximum = _finite_float(thermal.get("max_observed_c"), "thermal maximum")
    if maximum > limit:
        raise ValueError("thermal maximum exceeds the configured limit")
    exit_status = _exact_keys(
        exact.get("exit"), {"code", "raw_wait_status"}, "E049c exit"
    )
    _zero_int(exit_status.get("code"), "E049c exit code")
    _zero_int(exit_status.get("raw_wait_status"), "E049c raw wait status")
    configs = contract.get("pmu_group_configs", {}).get(cell["pmu_group"])
    events = exact.get("events")
    if not isinstance(configs, Mapping) or not isinstance(events, list) or \
            len(events) != len(configs) or exact.get("event_group_size") != len(configs) or \
            not _is_int(exact.get("event_group_size")):
        raise ValueError("E049c event group does not have the exact size")
    seen: set[str] = set()
    for event in events:
        item = _exact_keys(
            event,
            {
                "name", "config", "meaning", "support", "count_semantics",
                "sample_valid", "value", "time_enabled_ns", "time_running_ns",
                "running_ratio", "errno", "error",
            },
            "E049c event",
        )
        name = item.get("name")
        if name not in configs or name in seen or item.get("config") != configs.get(name):
            raise ValueError("E049c event name/config does not match the exact group")
        seen.add(str(name))
        if not isinstance(item.get("meaning"), str) or not item.get("meaning") or \
                item.get("support") != "supported" or \
                item.get("count_semantics") != "event_count_not_bytes" or \
                item.get("sample_valid") is not True or item.get("error") is not None:
            raise ValueError("E049c event support/count semantics are invalid")
        if not _is_int(item.get("value")) or item.get("value") < 0:
            raise ValueError("E049c event value must be a nonnegative integer count")
        enabled = _positive_int(item.get("time_enabled_ns"), "PMU time_enabled_ns")
        running = _positive_int(item.get("time_running_ns"), "PMU time_running_ns")
        if enabled > measured or running != enabled:
            raise ValueError("PMU enabled/running timing is inconsistent")
        ratio = item.get("running_ratio")
        if type(ratio) is not float or not math.isfinite(ratio) or ratio != 1.0:
            raise ValueError("qualified PMU running_ratio must be float exactly 1.0")
        _zero_int(item.get("errno"), "PMU errno")
    if seen != set(configs):
        raise ValueError("E049c event set is incomplete")


def _validate_runner(
    raw: Mapping[str, Any], run: Mapping[str, Any], cell: Mapping[str, Any],
    argv: tuple[str, ...], provenance: Mapping[str, Any], artifact_map: Mapping[str, SealedArtifact],
) -> None:
    exact = _exact_keys(
        raw,
        {
            "schema", "run_id", "pair_id", "pair_index", "pair_order", "order_index",
            "build_name", "argv", "environment", "affinity", "exit", "provenance",
            "artifact_sha256", "target_workload_executed",
        },
        "runner metadata",
    )
    cross = ("run_id", "pair_id", "pair_index", "pair_order", "order_index", "build_name")
    if exact.get("schema") != "e055-runner-capture/v1" or \
            exact.get("target_workload_executed") is not True or \
            any(exact.get(key) != run.get(key) for key in cross):
        raise ValueError("runner identity does not match the manifest run")
    if exact.get("argv") != list(argv):
        raise ValueError("runner argv is not the exact canonical command")
    environment = _exact_keys(
        exact.get("environment"), {"LC_ALL", "LANG", "E055_BUILD_NAME"},
        "runner non-secret environment",
    )
    if environment != {"LC_ALL": "C", "LANG": "C", "E055_BUILD_NAME": run["build_name"]}:
        raise ValueError("runner environment contains an unknown, secret, or noncanonical value")
    affinity = _exact_keys(
        exact.get("affinity"),
        {"requested_cpus", "effective_cpus", "cpu_start", "cpu_end", "migration_count"},
        "runner affinity",
    )
    cpu = cell["cpu"]
    if affinity.get("requested_cpus") != [cpu] or affinity.get("effective_cpus") != [cpu] or \
            not all(_is_int(affinity.get(key)) and affinity.get(key) == cpu
                    for key in ("cpu_start", "cpu_end")) or \
            not _is_int(affinity.get("migration_count")) or \
            affinity.get("migration_count") != 0:
        raise ValueError("runner affinity/migration does not prove one exact CPU")
    exit_status = _exact_keys(exact.get("exit"), {"code", "signal"}, "runner exit")
    _zero_int(exit_status.get("code"), "runner exit code")
    if exit_status.get("signal") is not None:
        raise ValueError("runner process terminated by a signal")
    if exact.get("provenance") != provenance:
        raise ValueError("runner provenance does not match publication artifacts/build")
    hashes = _exact_keys(
        exact.get("artifact_sha256"),
        {"harness_stdout", "harness_stderr", "e049c_json", "e049c_stderr"},
        "runner raw-artifact hashes",
    )
    if any(hashes.get(role) != artifact_map[role].sha256 for role in hashes):
        raise ValueError("runner raw-artifact SHA-256 cross-binding mismatch")


def load_sealed_bundle(manifest_path: str | Path) -> SealedBundle:
    """Load, parse, cross-check, and derive samples from one committed bundle."""

    files = seal_bundle_files(manifest_path)
    document = files.document
    _validate_manifest_shape(document)
    contract, publication_sha256 = _publication_contract()
    identity = _validate_qualification(
        document.get("qualification"), contract, publication_sha256
    )
    qualification = dict(identity)
    build_files: dict[str, SealedArtifact] = {}
    artifacts_by_path = {artifact.relative_path: artifact for artifact in files.artifacts}
    for build in document["build_artifacts"]:
        name = build["build_name"]
        declaration = build["artifact"]
        artifact = artifacts_by_path[declaration["path"]]
        allowed = contract.get("allowed_builds")
        if not isinstance(allowed, Mapping) or name not in allowed or \
                artifact.sha256 != allowed[name] or declaration.get("role") != "harness_executable":
            raise ValueError("sealed executable does not match an allowed publication build")
        build_files[name] = artifact

    samples: list[DerivedSample] = []
    for run in document["runs"]:
        exact_run = _exact_keys(
            run,
            {
                "run_id", "pair_id", "pair_index", "pair_order", "order_index",
                "build_name", "cell", "artifacts",
            },
            "manifest run",
        )
        for key in ("run_id", "pair_id"):
            value = exact_run.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value) is None:
                raise ValueError(f"{key} is not canonical")
        pair_index = _positive_int(exact_run.get("pair_index"), "pair_index")
        if pair_index > 5 or exact_run.get("pair_order") not in ("hot_then_cold", "cold_then_hot") or \
                not _is_int(exact_run.get("order_index")) or exact_run.get("order_index") not in (1, 2):
            raise ValueError("pair index/order metadata is invalid")
        expected_pair_order = "hot_then_cold" if pair_index % 2 else "cold_then_hot"
        cell = _validate_cell(exact_run.get("cell"))
        expected_order = {
            ("hot_then_cold", "hot_repeat"): 1,
            ("hot_then_cold", "cold_conditioned"): 2,
            ("cold_then_hot", "cold_conditioned"): 1,
            ("cold_then_hot", "hot_repeat"): 2,
        }.get((exact_run.get("pair_order"), cell["cache_state"]))
        if exact_run.get("pair_order") != expected_pair_order or \
                exact_run.get("order_index") != expected_order:
            raise ValueError("pair order must alternate exactly by pair_index")
        build_name = exact_run.get("build_name")
        if build_name not in build_files:
            raise ValueError("manifest run references an undeclared build")
        declarations = exact_run["artifacts"]
        artifact_map = {
            role: artifacts_by_path[declarations[role]["path"]] for role in RUN_ROLES
        }
        run_id = exact_run["run_id"]
        stdout_payload = _decode_stream(
            artifact_map["harness_stdout"], run_id=run_id,
            producer="e055_harness", stream="stdout",
        )
        if _decode_stream(
            artifact_map["harness_stderr"], run_id=run_id,
            producer="e055_harness", stream="stderr",
        ):
            raise ValueError("qualified E055 harness stderr must be empty")
        if _decode_stream(
            artifact_map["e049c_stderr"], run_id=run_id,
            producer="e049c_launcher", stream="stderr",
        ):
            raise ValueError("qualified E049c stderr must be empty")
        harness = _parse_json_object(stdout_payload, "E055 harness stdout payload")
        iterations, calls, elapsed_ns, checksum = _validate_harness(harness, exact_run, cell)
        executable = build_files[build_name]
        argv = canonical_harness_argv(cell, executable.relative_path, iterations)
        e049c = _parse_json_object(artifact_map["e049c_json"].payload, "E049c raw JSON")
        _validate_e049c(e049c, cell, argv, elapsed_ns, contract)
        provenance = {
            **qualification,
            "binary_sha256": contract["allowed_builds"][build_name],
        }
        runner = _parse_json_object(
            artifact_map["runner_metadata"].payload, "runner metadata"
        )
        _validate_runner(runner, exact_run, cell, argv, provenance, artifact_map)
        samples.append(DerivedSample(
            run_id=run_id,
            pair_id=exact_run["pair_id"],
            pair_index=pair_index,
            pair_order=exact_run["pair_order"],
            order_index=exact_run["order_index"],
            build_name=build_name,
            mode=cell["mode"],
            cache_state=cell["cache_state"],
            cpu=cell["cpu"],
            target_working_set_bytes=cell["target_working_set_bytes"],
            actual_working_set_bytes=cell["actual_working_set_bytes"],
            blocks=cell["blocks"],
            iterations=iterations,
            calls=calls,
            elapsed_ns=elapsed_ns,
            checksum=checksum,
            pmu_group=cell["pmu_group"],
            publication_identity=tuple(sorted(provenance.items())),
            commit=files.commit,
            tree=files.tree,
            manifest_path=files.manifest.relative_path,
            manifest_blob_oid=files.manifest.git_blob_oid,
            raw_artifacts=tuple(
                RawArtifactEvidence(
                    role=role,
                    relative_path=artifact_map[role].relative_path,
                    sha256=artifact_map[role].sha256,
                    size_bytes=artifact_map[role].size_bytes,
                    git_blob_oid=artifact_map[role].git_blob_oid,
                )
                for role in sorted(RUN_ROLES)
            ),
        ))
    return SealedBundle(files=files, samples=tuple(samples))
