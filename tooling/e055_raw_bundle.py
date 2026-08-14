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

from tooling.e055_transport_evidence import (
    canonical_deployment_layout,
    canonical_remote_output_path,
    validate_serialized_transport_evidence,
)


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
UINT64_MAX = (1 << 64) - 1
NATIVE_CARRIER_BYTES = 208
MAX_CPP_NATIVE_BLOCKS = ((1 << 31) - 1) // 128
COLD_THRASH_BYTES = 64 * 1024 * 1024
CACHE_LINE_BYTES = 64
COLD_THRASH_LINES = COLD_THRASH_BYTES // CACHE_LINE_BYTES
COLD_THRASH_CHECKSUM = "0x8d3ea13d15850279"
HOT_WARMUP_CALLS = 16
E049C_SYNC_TIMEOUT_MS = 5000
E049C_MAX_TEMP_C = 85.0
E049C_MIN_RUNNING_RATIO = 0.95
CLOCK_DOMAIN_TOLERANCE_NS = 1_000_000
CANONICAL_WORKING_SET_BYTES = (
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    25 * 1024 * 1024 // 2,
)


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
class PmuEventEvidence:
    name: str
    config: str
    value: int
    time_enabled_ns: int
    time_running_ns: int
    running_ratio: float


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
    measured_elapsed_ns: int
    thermal_limit_c: float
    max_temp_c: float
    pmu_events: tuple[PmuEventEvidence, ...]
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


def _git_entries(
    root: Path, source: str, phase_relative: str,
) -> dict[str, tuple[str, str]]:
    if source == "HEAD":
        output = _git(root, "ls-tree", "-r", "HEAD", "--", phase_relative)
    else:
        output = _git(root, "ls-files", "--stage", "--", phase_relative)
    assert isinstance(output, str)
    entries: dict[str, tuple[str, str]] = {}
    for line in (line for line in output.splitlines() if line):
        metadata, tab, path = line.partition("\t")
        fields = metadata.split()
        if tab != "\t" or path in entries:
            raise ValueError(f"Git returned a duplicate or aliased {source} path")
        if source == "HEAD":
            if len(fields) != 3 or fields[1] != "blob":
                raise ValueError(f"{path} is not a regular HEAD blob")
            mode, _, oid = fields
        else:
            if len(fields) != 3 or fields[2] != "0":
                raise ValueError(f"{path} has a non-stage-zero index entry")
            mode, oid, _ = fields
        if mode not in ("100644", "100755") or GIT_OID_RE.fullmatch(oid) is None:
            raise ValueError(f"{path} has an unsupported Git mode or object ID")
        entries[path] = (mode, oid)
    return entries


def _git_blob_oid(payload: bytes, object_format: str) -> str:
    if object_format not in ("sha1", "sha256"):
        raise ValueError(f"unsupported Git object format: {object_format}")
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.new(object_format, framed).hexdigest()


def _seal_file(
    root: Path,
    relative: str,
    role: str,
    maximum: int,
    declared: Mapping[str, Any] | None,
    head_entries: Mapping[str, tuple[str, str]],
    index_entries: Mapping[str, tuple[str, str]],
    object_format: str,
) -> tuple[SealedArtifact, tuple[int, int]]:
    path = root / relative
    payload, status = _read_regular_file(path, maximum)
    try:
        head_mode, head_oid = head_entries[relative]
        index_mode, index_oid = index_entries[relative]
    except KeyError as exc:
        raise ValueError(f"{relative} is not one exact tracked Git entry") from exc
    if (head_mode, head_oid) != (index_mode, index_oid):
        raise ValueError(f"index and HEAD differ for sealed artifact {relative}")
    if _git_blob_oid(payload, object_format) != head_oid:
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

    phase_text = phase_relative.as_posix()
    head_entries = _git_entries(root, "HEAD", phase_text)
    index_entries = _git_entries(root, "INDEX", phase_text)
    format_output = _git(root, "rev-parse", "--show-object-format")
    assert isinstance(format_output, str)
    object_format = format_output.strip()

    manifest_sealed, manifest_inode = _seal_file(
        root, manifest_relative, "bundle_manifest", MAX_MANIFEST_BYTES, None,
        head_entries, index_entries, object_format,
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
            root, relative, role, ROLE_SIZE_LIMITS[role], declaration,
            head_entries, index_entries, object_format,
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


def _positive_u64(value: Any, label: str) -> int:
    if not _is_int(value) or value <= 0 or value > UINT64_MAX:
        raise ValueError(f"{label} must be a positive uint64 integer")
    return value


def _nonnegative_u64(value: Any, label: str) -> int:
    if not _is_int(value) or value < 0 or value > UINT64_MAX:
        raise ValueError(f"{label} must be a nonnegative uint64 integer")
    return value


def _nonzero_u64_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or \
            re.fullmatch(r"0x[1-9a-f][0-9a-f]{0,15}", value) is None:
        raise ValueError(f"{label} must be canonical nonzero uint64 hexadecimal")
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
    if text[end:] != "\n" or not isinstance(value, Mapping):
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
    from tooling.e055_q1_hotcold import (
        load_publication_contract,
        runtime_qualification_sha256,
    )

    contract = load_publication_contract()
    return contract, runtime_qualification_sha256(contract)


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
        arguments.extend(("--warmup", str(HOT_WARMUP_CALLS)))
    arguments.extend(("--thrash-bytes", str(COLD_THRASH_BYTES), "--sync"))
    return tuple(arguments)


def canonical_e049c_launcher_argv(
    cell: Mapping[str, Any], output_path: str,
    harness_argv: tuple[str, ...], pmu_executable_path: str,
) -> tuple[str, ...]:
    """Return the exact E049c wrapper command accepted for E055."""

    if not isinstance(output_path, str) or not output_path:
        raise ValueError("E049c output path must be a nonempty string")
    output = PurePosixPath(output_path)
    executable = PurePosixPath(pmu_executable_path)
    if not output.is_absolute() or ".." in output.parts or output.name != "e049c.json" or \
            not executable.is_absolute() or ".." in executable.parts or \
            executable.name != "a733-pmu-exec-aarch64":
        raise ValueError("E049c executable/output paths must be canonical remote paths")
    child_stdout_path = (output.parent / "target-harness.stdout.raw").as_posix()
    child_stderr_path = (output.parent / "target-harness.stderr.raw").as_posix()
    return (
        pmu_executable_path, "-o", output_path,
        "--child-stdout", child_stdout_path,
        "--child-stderr", child_stderr_path,
        "--event-group", str(cell["pmu_group"]),
        "--min-running-ratio", "0.95",
        "--start-on-ready",
        "--sync-timeout-ms", str(E049C_SYNC_TIMEOUT_MS),
        "--max-temp-c", "85",
        "--", *harness_argv,
    )


def _validate_qualification(
    qualification: Any, contract: Mapping[str, Any], qualification_sha256: str,
) -> tuple[tuple[str, Any], ...]:
    expected = {
        "source_sha256": contract.get("source_sha256"),
        "compiler_sha256": contract.get("compiler_sha256"),
        "compiler_id": contract.get("compiler_id"),
        "pmu_source_sha256": contract.get("pmu_source_sha256"),
        "pmu_binary_sha256": contract.get("pmu_binary_sha256"),
        "pmu_binary_size_bytes": contract.get("pmu_binary_size_bytes"),
        "pmu_compiler_sha256": contract.get("pmu_compiler_sha256"),
        "pmu_compiler_id": contract.get("pmu_compiler_id"),
        "upstream_commit": contract.get("upstream_commit"),
        "upstream_ref": contract.get("upstream_ref"),
        "upstream_repack_sha256": contract.get("upstream_repack_sha256"),
        "runtime_qualification_sha256": qualification_sha256,
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
            "build_artifacts", "runs", "transport_evidence",
            "target_workload_executed",
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
    target = _positive_u64(exact.get("target_working_set_bytes"), "target working set")
    blocks = _positive_int(exact.get("blocks"), "native blocks")
    actual = _positive_int(exact.get("actual_working_set_bytes"), "actual working set")
    if target not in CANONICAL_WORKING_SET_BYTES:
        raise ValueError("target working set is not one canonical planned E055 size")
    if target > UINT64_MAX - (NATIVE_CARRIER_BYTES - 1):
        raise ValueError("target working set exceeds the C++ uint64 rounding bound")
    expected_blocks = (target + NATIVE_CARRIER_BYTES - 1) // NATIVE_CARRIER_BYTES
    if expected_blocks > MAX_CPP_NATIVE_BLOCKS or blocks > MAX_CPP_NATIVE_BLOCKS:
        raise ValueError("working set exceeds the C++ native-block allocation bound")
    if blocks != expected_blocks or actual != blocks * NATIVE_CARRIER_BYTES:
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
    for key in ("mode", "cache_state"):
        if not isinstance(exact.get(key), str):
            raise ValueError(f"E055 stdout {key} must be a string")
    numeric_identity = (
        "cpu", "target_working_set_bytes", "actual_working_set_bytes", "blocks",
    )
    for key in numeric_identity:
        if not _is_int(exact.get(key)):
            raise ValueError(f"E055 stdout {key} must be an integer, not Boolean/float")
    if any(exact.get(key) != value for key, value in cross.items()):
        raise ValueError("E055 stdout does not match its manifest cell")
    if exact.get("schema") != "e055-q1-hot-cold-harness/v1" or \
            exact.get("q1_layout") != "E039 stock native block_q1_0x4 4x4 DOTPROD" or \
            exact.get("golden_pass") is not True or exact.get("golden_cases") != 18 or \
            not _is_int(exact.get("golden_cases")) or \
            exact.get("qualification") != "unqualified_harness_output_requires_E049c_join":
        raise ValueError("E055 stdout lacks the exact build-bound 18-case golden")
    iterations = _positive_u64(exact.get("iterations"), "harness iterations")
    calls = _positive_u64(exact.get("calls"), "harness calls")
    elapsed = _positive_u64(exact.get("elapsed_ns"), "harness elapsed_ns")
    _positive_u64(exact.get("first_call_ns"), "harness first_call_ns")
    if iterations != calls:
        raise ValueError("harness iterations and calls differ")
    if cell["cache_state"] == "cold_conditioned" and calls != 1:
        raise ValueError("cold_conditioned raw capture must contain exactly one call")
    rate = _finite_float(exact.get("calls_per_second"), "harness calls_per_second")
    if not math.isclose(rate, calls * 1.0e9 / elapsed, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("harness calls_per_second does not match calls/elapsed_ns")
    checksum = _nonzero_u64_hex(exact.get("checksum"), "harness checksum")
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
    if any(not _is_int(logical.get(key)) for key in expected_logical) or \
            dict(logical) != expected_logical:
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
        _positive_u64(conditioning.get(key), f"conditioning {key}")
    if not _is_int(conditioning.get("warmup_calls")) or \
            conditioning.get("warmup_calls") < 0:
        raise ValueError("conditioning warmup_calls must be a nonnegative integer")
    if conditioning.get("verified_touched") is not True or \
            conditioning.get("line_bytes") != CACHE_LINE_BYTES:
        raise ValueError("cache conditioning is not exactly verified at 64-byte lines")
    condition_checksum = _nonzero_u64_hex(
        conditioning.get("checksum"), "conditioning checksum"
    )
    if cell["cache_state"] == "hot_repeat":
        if conditioning.get("strategy") != "verified_kernel_warmup" or \
                not _is_int(conditioning.get("warmup_calls")) or \
                conditioning["warmup_calls"] != HOT_WARMUP_CALLS or \
                conditioning.get("actual_bytes") != cell["actual_working_set_bytes"] or \
                conditioning.get("requested_bytes") != cell["actual_working_set_bytes"] or \
                conditioning.get("lines_touched") != \
                (cell["actual_working_set_bytes"] + CACHE_LINE_BYTES - 1) // CACHE_LINE_BYTES:
            raise ValueError("hot conditioning does not cover the exact working set")
    else:
        if conditioning.get("strategy") != "verified_write_read_each_64B_line" or \
                conditioning.get("warmup_calls") != 0 or \
                conditioning.get("requested_bytes") != COLD_THRASH_BYTES or \
                conditioning.get("actual_bytes") != COLD_THRASH_BYTES or \
                conditioning.get("lines_touched") != COLD_THRASH_LINES or \
                condition_checksum != COLD_THRASH_CHECKSUM:
            raise ValueError("cold conditioning does not cover every requested cache line")
    sync = _exact_keys(
        exact.get("sync"), {"requested", "started", "acknowledged", "ended", "sequence"},
        "harness sync",
    )
    expected_sync = {
        "requested": True, "started": True, "acknowledged": True,
        "ended": True, "sequence": "S/A/E",
    }
    if any(type(sync.get(key)) is not bool for key in
           ("requested", "started", "acknowledged", "ended")) or \
            sync != expected_sync:
        raise ValueError("harness did not complete exact S/ACK/E synchronization")
    return iterations, calls, elapsed, checksum


def _validate_e049c(
    raw: Mapping[str, Any], cell: Mapping[str, Any], argv: tuple[str, ...],
    harness_elapsed_ns: int, contract: Mapping[str, Any],
) -> tuple[int, float, float, tuple[PmuEventEvidence, ...]]:
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
    if not isinstance(exact.get("command"), list) or \
            any(not isinstance(item, str) for item in exact["command"]) or \
            exact.get("command") != list(argv):
        raise ValueError("E049c command does not match exact runner argv")
    if type(exact.get("min_running_ratio")) is not float or \
            exact.get("min_running_ratio") != E049C_MIN_RUNNING_RATIO:
        raise ValueError("E049c launcher floor must remain the explicit float 0.95")
    if not _is_int(exact.get("software_group_size_limit")) or \
            exact.get("software_group_size_limit") != 4:
        raise ValueError("E049c software group-size limit is invalid")
    if not _is_int(exact.get("pid")) or exact.get("pid") <= 0 or \
            not _is_int(exact.get("process_group")) or exact.get("process_group") <= 0 or \
            exact.get("pid") != exact.get("process_group"):
        raise ValueError("E049c PID must equal its positive integer process group")
    sync = _exact_keys(
        exact.get("sync"), {"mode", "started", "acknowledged", "ended"}, "E049c sync"
    )
    expected_sync = {"mode": "start_ack_end", "started": True,
                     "acknowledged": True, "ended": True}
    if any(type(sync.get(key)) is not bool for key in
           ("started", "acknowledged", "ended")) or sync != expected_sync:
        raise ValueError("E049c did not observe exact S/ACK/E synchronization")
    measured = _positive_u64(
        exact.get("measured_elapsed_ns"), "E049c measured elapsed"
    )
    if measured + CLOCK_DOMAIN_TOLERANCE_NS < harness_elapsed_ns:
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
    if limit != E049C_MAX_TEMP_C or maximum > limit:
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
    qualified_events: list[PmuEventEvidence] = []
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
        value = _nonnegative_u64(item.get("value"), "E049c event value")
        enabled = _positive_u64(item.get("time_enabled_ns"), "PMU time_enabled_ns")
        running = _positive_u64(item.get("time_running_ns"), "PMU time_running_ns")
        if running != enabled or enabled + CLOCK_DOMAIN_TOLERANCE_NS < harness_elapsed_ns:
            raise ValueError(
                "PMU enabled/running time is shorter than the harness window"
            )
        ratio = item.get("running_ratio")
        if type(ratio) is not float or not math.isfinite(ratio) or ratio != 1.0:
            raise ValueError("qualified PMU running_ratio must be float exactly 1.0")
        _zero_int(item.get("errno"), "PMU errno")
        qualified_events.append(PmuEventEvidence(
            name=str(name),
            config=str(item["config"]),
            value=value,
            time_enabled_ns=enabled,
            time_running_ns=running,
            running_ratio=ratio,
        ))
    if seen != set(configs):
        raise ValueError("E049c event set is incomplete")
    return measured, limit, maximum, tuple(qualified_events)


def _validate_runner(
    raw: Mapping[str, Any], run: Mapping[str, Any], cell: Mapping[str, Any],
    argv: tuple[str, ...], e049c_argv: tuple[str, ...],
    provenance: Mapping[str, Any], artifact_map: Mapping[str, SealedArtifact],
    transport_evidence: Mapping[str, Any],
) -> None:
    exact = _exact_keys(
        raw,
        {
            "schema", "run_id", "pair_id", "pair_index", "pair_order", "order_index",
            "build_name", "argv", "e049c_argv", "environment", "affinity",
            "exit", "provenance", "transport_evidence",
            "artifact_sha256", "target_workload_executed",
        },
        "runner metadata",
    )
    cross = ("run_id", "pair_id", "pair_index", "pair_order", "order_index", "build_name")
    if exact.get("schema") != "e055-runner-capture/v1" or \
            exact.get("target_workload_executed") is not True or \
            any(exact.get(key) != run.get(key) for key in cross):
        raise ValueError("runner identity does not match the manifest run")
    for key in ("run_id", "pair_id", "pair_order", "build_name"):
        if not isinstance(exact.get(key), str):
            raise ValueError(f"runner {key} must be a string")
    if not _is_int(exact.get("pair_index")) or not _is_int(exact.get("order_index")):
        raise ValueError("runner pair/order indexes must be integers, not Boolean/float")
    if not isinstance(exact.get("argv"), list) or \
            any(not isinstance(item, str) for item in exact["argv"]) or \
            exact.get("argv") != list(argv):
        raise ValueError("runner argv is not the exact canonical command")
    if not isinstance(exact.get("e049c_argv"), list) or \
            any(not isinstance(item, str) for item in exact["e049c_argv"]) or \
            exact.get("e049c_argv") != list(e049c_argv):
        raise ValueError("runner E049c argv is not the exact canonical launcher command")
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
    requested = affinity.get("requested_cpus")
    effective = affinity.get("effective_cpus")
    if not isinstance(requested, list) or len(requested) != 1 or \
            not _is_int(requested[0]) or not isinstance(effective, list) or \
            len(effective) != 1 or not _is_int(effective[0]) or \
            requested != [cpu] or effective != [cpu] or \
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
    if exact.get("transport_evidence") != transport_evidence:
        raise ValueError("runner transport evidence does not match its sealed bundle")
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
    contract, qualification_sha256 = _publication_contract()
    identity = _validate_qualification(
        document.get("qualification"), contract, qualification_sha256
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

    if len(build_files) != 1:
        raise ValueError("one raw phase must deploy exactly one qualified harness build")
    deployed_build = next(iter(build_files.values()))
    layout = canonical_deployment_layout(
        str(document["phase_id"]), deployed_build.sha256,
        str(contract["pmu_binary_sha256"]),
    )
    transport_evidence = validate_serialized_transport_evidence(
        document.get("transport_evidence"), layout,
        (
            {
                "role": "harness_executable",
                "target_path": layout.harness_path,
                "sha256": deployed_build.sha256,
                "size_bytes": deployed_build.size_bytes,
                "mode": 0o755,
            },
            {
                "role": "pmu_executable",
                "target_path": layout.pmu_path,
                "sha256": contract["pmu_binary_sha256"],
                "size_bytes": contract["pmu_binary_size_bytes"],
                "mode": 0o755,
            },
        ),
    )

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
        argv = canonical_harness_argv(cell, layout.harness_path, iterations)
        e049c = _parse_json_object(artifact_map["e049c_json"].payload, "E049c raw JSON")
        measured_ns, thermal_limit, max_temp, pmu_events = _validate_e049c(
            e049c, cell, argv, elapsed_ns, contract
        )
        provenance = {
            **qualification,
            "binary_sha256": contract["allowed_builds"][build_name],
        }
        runner = _parse_json_object(
            artifact_map["runner_metadata"].payload, "runner metadata"
        )
        e049c_argv = canonical_e049c_launcher_argv(
            cell, canonical_remote_output_path(layout, run_id), argv,
            layout.pmu_path,
        )
        _validate_runner(
            runner, exact_run, cell, argv, e049c_argv, provenance, artifact_map,
            transport_evidence,
        )
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
            measured_elapsed_ns=measured_ns,
            thermal_limit_c=thermal_limit,
            max_temp_c=max_temp,
            pmu_events=pmu_events,
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


def derived_sample_document(sample: DerivedSample) -> dict[str, Any]:
    """Serialize an internally derived row for publication, never ingestion."""

    if not isinstance(sample, DerivedSample):
        raise TypeError("only an analyzer-derived E055 sample can be published")
    return {
        "schema": "e055-derived-sample/v1",
        "run_id": sample.run_id,
        "pair_id": sample.pair_id,
        "pair_index": sample.pair_index,
        "pair_order": sample.pair_order,
        "order_index": sample.order_index,
        "build_name": sample.build_name,
        "mode": sample.mode,
        "cache_state": sample.cache_state,
        "cpu": sample.cpu,
        "target_working_set_bytes": sample.target_working_set_bytes,
        "actual_working_set_bytes": sample.actual_working_set_bytes,
        "blocks": sample.blocks,
        "iterations": sample.iterations,
        "calls": sample.calls,
        "elapsed_ns": sample.elapsed_ns,
        "checksum": sample.checksum,
        "pmu_group": sample.pmu_group,
        "measured_elapsed_ns": sample.measured_elapsed_ns,
        "thermal_limit_c": sample.thermal_limit_c,
        "max_temp_c": sample.max_temp_c,
        "pmu_events": [
            {
                "name": event.name,
                "config": event.config,
                "value": event.value,
                "time_enabled_ns": event.time_enabled_ns,
                "time_running_ns": event.time_running_ns,
                "running_ratio": event.running_ratio,
            }
            for event in sample.pmu_events
        ],
        "golden_pass": True,
        "golden_cases": 18,
        "evidence": {
            "commit": sample.commit,
            "tree": sample.tree,
            "manifest_path": sample.manifest_path,
            "manifest_blob_oid": sample.manifest_blob_oid,
            "publication_identity": dict(sample.publication_identity),
            "raw_artifacts": [
                {
                    "role": artifact.role,
                    "relative_path": artifact.relative_path,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "git_blob_oid": artifact.git_blob_oid,
                }
                for artifact in sample.raw_artifacts
            ],
        },
    }
