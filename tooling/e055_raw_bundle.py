"""Git-sealed raw artifact ingestion for E055.

This module treats the current Git commit as a byte-immutability boundary. It
does not claim that Git proves device truth. Statistical qualification is built
only after raw files pass this sealing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
