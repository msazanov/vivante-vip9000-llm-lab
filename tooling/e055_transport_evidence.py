"""Fail-closed transport evidence for the bounded E055 target phase.

The caller supplies distinct random request IDs/nonces at two API boundaries,
and each response must echo its exact request. This rejects cached/replayed
responses at the live boundary, but it is not cryptographic device attestation;
the transport implementation and its endpoint remain in the trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_TOKEN_RE = SHA256_RE
PHASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
RUN_ID_RE = PHASE_ID_RE
HOST_KEY_RE = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
BOARD_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PYTHON_REALPATH_RE = re.compile(r"^/usr/bin/python3(?:\.[0-9]+)?$")
PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]{0,32})?$")
OPENSSH_VERSION_RE = re.compile(r"^OpenSSH_[A-Za-z0-9._,+-]{1,96}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
DEPLOYMENT_BASE = "/tmp/e055-q1-hot-cold"


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _read_regular_file(path: Path, maximum: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or \
                before.st_size < 1 or before.st_size > maximum:
            raise ValueError("local provenance file is not one bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("local provenance file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("local provenance file grew while reading")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mode
        ):
            raise ValueError("local provenance file identity changed")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=root, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("Git provenance query failed")
    return result.stdout.strip()


def _expected_pins_document(value: Any) -> dict[str, Any]:
    if type(value) is not ExpectedTransportPins or \
            value.schema != "e055-expected-transport-pins/v1" or \
            value.trust_mode != "pinned_host_key":
        raise ValueError("independent expected transport pins are required")
    if not isinstance(value.endpoint_label, str) or \
            not 1 <= len(value.endpoint_label) <= 128 or \
            any(ord(character) < 32 for character in value.endpoint_label) or \
            not isinstance(value.board_identity, str) or \
            BOARD_IDENTITY_RE.fullmatch(value.board_identity) is None or \
            not isinstance(value.host_key_fingerprint, str) or \
            HOST_KEY_RE.fullmatch(value.host_key_fingerprint) is None:
        raise ValueError("expected endpoint pins are invalid")
    for digest in (
        value.known_hosts_sha256, value.target_endpoint_sha256,
        value.helper_source_sha256, value.openssh_sha256,
    ):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or \
                digest == "0" * 64:
            raise ValueError("expected transport digest is invalid")
    if not _strict_int(value.helper_source_size_bytes, minimum=1) or \
            value.helper_source_size_bytes > 1024 * 1024 or \
            not isinstance(value.helper_git_blob_oid, str) or \
            GIT_OID_RE.fullmatch(value.helper_git_blob_oid) is None or \
            not _strict_int(value.openssh_size_bytes, minimum=1) or \
            value.openssh_size_bytes > 32 * 1024 * 1024:
        raise ValueError("expected local implementation pins are invalid")
    return {
        "schema": value.schema, "trust_mode": value.trust_mode,
        "endpoint_label": value.endpoint_label,
        "board_identity": value.board_identity,
        "host_key_fingerprint": value.host_key_fingerprint,
        "known_hosts_sha256": value.known_hosts_sha256,
        "target_endpoint_sha256": value.target_endpoint_sha256,
        "helper_source_sha256": value.helper_source_sha256,
        "helper_source_size_bytes": value.helper_source_size_bytes,
        "helper_git_blob_oid": value.helper_git_blob_oid,
        "openssh_sha256": value.openssh_sha256,
        "openssh_size_bytes": value.openssh_size_bytes,
    }


def _verify_local_implementation(
    value: TransportImplementationEvidence,
    pins: ExpectedTransportPins,
    repository_root: Path,
) -> None:
    if not isinstance(repository_root, Path) or not repository_root.is_absolute():
        raise ValueError("repository root must be one absolute Path")
    helper = repository_root / value.helper_source_path
    try:
        helper.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("helper source escaped the repository") from exc
    helper_payload, helper_status = _read_regular_file(helper, 1024 * 1024)
    openssh_payload, openssh_status = _read_regular_file(
        Path(value.openssh_path), 32 * 1024 * 1024,
    )
    if hashlib.sha256(helper_payload).hexdigest() != pins.helper_source_sha256 or \
            helper_status.st_size != pins.helper_source_size_bytes or \
            hashlib.sha256(openssh_payload).hexdigest() != pins.openssh_sha256 or \
            openssh_status.st_size != pins.openssh_size_bytes:
        raise ValueError("actual local helper/OpenSSH bytes do not match expected pins")
    if _git(repository_root, "status", "--porcelain=v2", "--", value.helper_source_path):
        raise ValueError("helper source is dirty, staged, deleted, or untracked")
    head = _git(repository_root, "ls-tree", "HEAD", "--", value.helper_source_path)
    index = _git(repository_root, "ls-files", "--stage", "--", value.helper_source_path)
    head_fields = head.split()
    index_fields = index.split()
    if len(head_fields) < 3 or len(index_fields) < 2 or \
            head_fields[0] not in ("100644", "100755") or head_fields[1] != "blob" or \
            index_fields[0] != head_fields[0] or index_fields[1] != head_fields[2] or \
            head_fields[2] != pins.helper_git_blob_oid:
        raise ValueError("helper source is not bound to one identical HEAD/index blob")


@dataclass(frozen=True)
class DeploymentLayout:
    deployment_root: str
    lock_path: str
    harness_path: str
    pmu_path: str


@dataclass(frozen=True)
class EndpointIdentity:
    """Operational endpoint identity reported by the transport."""

    trust_mode: str
    endpoint_label: str
    board_identity: str
    host_key_fingerprint: str | None


@dataclass(frozen=True)
class TargetArtifactObservation:
    role: str
    target_path: str
    sha256: str
    size_bytes: int
    mode: int
    device: int
    inode: int
    operation_sequence: int
    request_id: str
    request_nonce: str


@dataclass(frozen=True)
class TransportImplementationEvidence:
    """Exact local OpenSSH/helper bytes and reviewed non-secret configuration."""

    schema: str
    implementation: str
    helper_source_path: str
    helper_source_sha256: str
    helper_source_size_bytes: int
    helper_git_blob_oid: str
    openssh_path: str
    openssh_sha256: str
    openssh_size_bytes: int
    openssh_mode: int
    openssh_version: str
    config_file: str
    client_config: tuple[str, ...]
    client_config_sha256: str
    known_hosts_sha256: str
    target_endpoint_sha256: str
    credential_mode: str


@dataclass(frozen=True)
class ExpectedTransportPins:
    """Independent caller policy; never derived from a transport response."""

    schema: str
    trust_mode: str
    endpoint_label: str
    board_identity: str
    host_key_fingerprint: str
    known_hosts_sha256: str
    target_endpoint_sha256: str
    helper_source_sha256: str
    helper_source_size_bytes: int
    helper_git_blob_oid: str
    openssh_sha256: str
    openssh_size_bytes: int


@dataclass(frozen=True)
class RemoteRuntimeObservation:
    """Fresh helper and interpreter identity returned by one remote operation."""

    schema: str
    operation_sequence: int
    request_id: str
    request_nonce: str
    helper_path: str
    helper_sha256: str
    helper_size_bytes: int
    helper_mode: int
    helper_device: int
    helper_inode: int
    helper_protocol: str
    python_requested_path: str
    python_realpath: str
    python_version: str
    python_sha256: str
    python_size_bytes: int
    python_mode: int
    python_device: int
    python_inode: int


@dataclass(frozen=True)
class ExclusiveDeploymentReceipt:
    schema: str
    operation_sequence: int
    request_id: str
    request_nonce: str
    deployment_root: str
    lock_path: str
    lock_acquired_exclusively: bool
    deployment_created_exclusively: bool
    endpoint_identity: EndpointIdentity
    artifacts: tuple[TargetArtifactObservation, ...]
    runtime: RemoteRuntimeObservation


@dataclass(frozen=True)
class FreshReadbackProof:
    schema: str
    operation_sequence: int
    request_id: str
    request_nonce: str
    deployment_root: str
    endpoint_identity: EndpointIdentity
    artifacts: tuple[TargetArtifactObservation, ...]
    runtime: RemoteRuntimeObservation


def canonical_deployment_layout(
    phase_id: str, harness_sha256: str, pmu_sha256: str,
) -> DeploymentLayout:
    """Derive unique, content-addressed remote paths for one phase."""

    if not isinstance(phase_id, str) or PHASE_ID_RE.fullmatch(phase_id) is None:
        raise ValueError("phase_id must be canonical and bounded")
    for label, digest in (("harness", harness_sha256), ("PMU", pmu_sha256)):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label} SHA-256 must be canonical")
    deployment_key = hashlib.sha256(
        (
            "e055-deployment/v1\0" + phase_id + "\0" + harness_sha256
            + "\0" + pmu_sha256
        ).encode("ascii")
    ).hexdigest()
    root = f"{DEPLOYMENT_BASE}/deployments/{deployment_key}"
    return DeploymentLayout(
        deployment_root=root,
        lock_path=f"{DEPLOYMENT_BASE}/locks/a733-target.lock",
        harness_path=f"{root}/artifacts/{harness_sha256}/e055-O3-aarch64",
        pmu_path=f"{root}/artifacts/{pmu_sha256}/a733-pmu-exec-aarch64",
    )


def canonical_remote_output_path(layout: DeploymentLayout, run_id: str) -> str:
    """Return one phase-unique path reserved by the remote E049c launcher."""

    if type(layout) is not DeploymentLayout or not isinstance(run_id, str) or \
            RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("deployment layout and run_id must be canonical")
    return f"{layout.deployment_root}/captures/{run_id}/e049c.json"


def _validate_identity(identity: Any) -> EndpointIdentity:
    if type(identity) is not EndpointIdentity:
        raise ValueError("transport endpoint identity has a noncanonical type")
    if identity.trust_mode not in (
        "test_fixture", "pinned_host_key", "operationally_trusted",
    ):
        raise ValueError("transport endpoint trust mode is invalid")
    for value in (identity.endpoint_label, identity.board_identity):
        if not isinstance(value, str) or not 1 <= len(value) <= 256 or \
                any(ord(character) < 32 for character in value):
            raise ValueError("transport endpoint identity is not bounded text")
    fingerprint = identity.host_key_fingerprint
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or HOST_KEY_RE.fullmatch(fingerprint) is None
    ):
        raise ValueError("transport host-key fingerprint is not canonical")
    if identity.trust_mode == "pinned_host_key" and fingerprint is None:
        raise ValueError("pinned_host_key mode requires an exact fingerprint")
    if identity.trust_mode == "pinned_host_key" and (
        BOARD_IDENTITY_RE.fullmatch(identity.board_identity) is None
    ):
        raise ValueError("pinned_host_key mode requires a board identity digest")
    if identity.trust_mode == "test_fixture" and fingerprint is not None:
        raise ValueError("test fixtures cannot claim a real host-key pin")
    return identity


def _identity_document(identity: EndpointIdentity) -> dict[str, Any]:
    return {
        "trust_mode": identity.trust_mode,
        "endpoint_label": identity.endpoint_label,
        "board_identity": identity.board_identity,
        "host_key_fingerprint": identity.host_key_fingerprint,
    }


def _observation_document(
    observation: Any, expected: Mapping[str, Any], *,
    operation_sequence: int, request_id: str, request_nonce: str,
) -> dict[str, Any]:
    if type(observation) is not TargetArtifactObservation:
        raise ValueError("transport artifact observation has a noncanonical type")
    if observation.role != expected.get("role") or \
            observation.target_path != expected.get("target_path") or \
            observation.sha256 != expected.get("sha256") or \
            observation.size_bytes != expected.get("size_bytes") or \
            observation.mode != expected.get("mode") or \
            observation.operation_sequence != operation_sequence or \
            not _strict_int(observation.operation_sequence, minimum=1) or \
            observation.request_id != request_id or \
            observation.request_nonce != request_nonce:
        raise ValueError("transport proof does not match exact deployed bytes/path/mode")
    if not isinstance(observation.target_path, str) or \
            not observation.target_path.startswith(DEPLOYMENT_BASE + "/") or \
            not isinstance(observation.sha256, str) or \
            SHA256_RE.fullmatch(observation.sha256) is None:
        raise ValueError("transport artifact path/hash is invalid")
    for value, label, minimum in (
        (observation.size_bytes, "size", 1),
        (observation.mode, "mode", 1),
        (observation.device, "device", 0),
        (observation.inode, "inode", 1),
    ):
        if not _strict_int(value, minimum=minimum):
            raise ValueError(f"transport artifact {label} is not a strict integer")
    return {
        "role": observation.role,
        "target_path": observation.target_path,
        "sha256": observation.sha256,
        "size_bytes": observation.size_bytes,
        "mode": observation.mode,
        "device": observation.device,
        "inode": observation.inode,
        "operation_sequence": observation.operation_sequence,
        "request_id": observation.request_id,
        "request_nonce": observation.request_nonce,
    }


def _implementation_document(value: Any) -> dict[str, Any]:
    if type(value) is not TransportImplementationEvidence:
        raise ValueError("transport implementation evidence has a noncanonical type")
    if value.schema != "e055-openssh-implementation/v1" or \
            value.implementation != "openssh_fixed_helper" or \
            value.credential_mode != "external_agent_or_identity":
        raise ValueError("transport implementation identity is invalid")
    if value.helper_source_path != "tooling/e055_remote_helper.py" or \
            not isinstance(value.helper_source_sha256, str) or \
            SHA256_RE.fullmatch(value.helper_source_sha256) is None or \
            value.helper_source_sha256 == "0" * 64 or \
            not _strict_int(value.helper_source_size_bytes, minimum=1) or \
            value.helper_source_size_bytes > 1024 * 1024 or \
            not isinstance(value.helper_git_blob_oid, str) or \
            GIT_OID_RE.fullmatch(value.helper_git_blob_oid) is None:
        raise ValueError("transport helper source identity is invalid")
    if not isinstance(value.openssh_path, str) or \
            not value.openssh_path.startswith("/") or \
            ".." in value.openssh_path.split("/") or \
            not value.openssh_path.endswith("/ssh") or \
            not isinstance(value.openssh_sha256, str) or \
            SHA256_RE.fullmatch(value.openssh_sha256) is None or \
            value.openssh_sha256 == "0" * 64 or \
            not _strict_int(value.openssh_size_bytes, minimum=1) or \
            value.openssh_size_bytes > 32 * 1024 * 1024 or \
            value.openssh_mode != 0o755 or isinstance(value.openssh_mode, bool) or \
            not isinstance(value.openssh_version, str) or \
            OPENSSH_VERSION_RE.fullmatch(value.openssh_version) is None:
        raise ValueError("local OpenSSH executable identity is invalid")
    if value.config_file != "/dev/null" or \
            not isinstance(value.known_hosts_sha256, str) or \
            SHA256_RE.fullmatch(value.known_hosts_sha256) is None or \
            value.known_hosts_sha256 == "0" * 64 or \
            not isinstance(value.target_endpoint_sha256, str) or \
            SHA256_RE.fullmatch(value.target_endpoint_sha256) is None or \
            value.target_endpoint_sha256 == "0" * 64:
        raise ValueError("OpenSSH execution boundary pins are invalid")
    if not isinstance(value.client_config, tuple) or not value.client_config or \
            len(set(value.client_config)) != len(value.client_config):
        raise ValueError("OpenSSH client config must be one unique tuple")
    exact_keys = {
        "BatchMode", "StrictHostKeyChecking", "UserKnownHostsFile",
        "GlobalKnownHostsFile", "CheckHostIP", "PasswordAuthentication",
        "KbdInteractiveAuthentication", "NumberOfPasswordPrompts",
        "ForwardAgent", "ClearAllForwardings", "PermitLocalCommand",
        "RequestTTY", "ConnectTimeout", "ConnectionAttempts",
        "ServerAliveInterval", "ServerAliveCountMax", "LogLevel",
        "IdentitiesOnly", "ProxyCommand", "ProxyJump", "CanonicalizeHostname",
    }
    parsed: dict[str, str] = {}
    for option in value.client_config:
        if not isinstance(option, str) or not 3 <= len(option) <= 128 or \
                any(ord(character) < 32 or ord(character) > 126 for character in option):
            raise ValueError("OpenSSH client config contains invalid text")
        key, separator, setting = option.partition("=")
        lowered = option.lower()
        if separator != "=" or key not in exact_keys or key in parsed or not setting or any(
            secret in lowered for secret in (
                "sshpass", "password=", "token=", "secret=", "identityfile=",
            )
        ):
            raise ValueError("OpenSSH client config is unreviewed or sensitive")
        parsed[key] = setting
    fixed = {
        "BatchMode": "yes", "StrictHostKeyChecking": "yes",
        "UserKnownHostsFile": "external-pinned-file",
        "GlobalKnownHostsFile": "/dev/null", "CheckHostIP": "no",
        "PasswordAuthentication": "no", "KbdInteractiveAuthentication": "no",
        "NumberOfPasswordPrompts": "0", "ForwardAgent": "no",
        "ClearAllForwardings": "yes", "PermitLocalCommand": "no",
        "RequestTTY": "no", "ConnectionAttempts": "1", "LogLevel": "ERROR",
        "IdentitiesOnly": "yes", "ProxyCommand": "none", "ProxyJump": "none",
        "CanonicalizeHostname": "no",
    }
    if set(parsed) != exact_keys or any(parsed.get(key) != setting for key, setting in fixed.items()):
        raise ValueError("OpenSSH client config does not enforce exact safe semantics")
    for key, minimum, maximum in (
        ("ConnectTimeout", 1, 60), ("ServerAliveInterval", 1, 30),
        ("ServerAliveCountMax", 1, 6),
    ):
        setting = parsed.get(key, "")
        if not setting.isascii() or not setting.isdigit() or \
                not minimum <= int(setting) <= maximum:
            raise ValueError("OpenSSH timeout configuration is outside safe bounds")
    config_bytes = ("\n".join(value.client_config) + "\n").encode("ascii")
    if not isinstance(value.client_config_sha256, str) or \
            value.client_config_sha256 != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("OpenSSH client config digest is invalid")
    return {
        "schema": value.schema,
        "implementation": value.implementation,
        "helper_source_path": value.helper_source_path,
        "helper_source_sha256": value.helper_source_sha256,
        "helper_source_size_bytes": value.helper_source_size_bytes,
        "helper_git_blob_oid": value.helper_git_blob_oid,
        "openssh_path": value.openssh_path,
        "openssh_sha256": value.openssh_sha256,
        "openssh_size_bytes": value.openssh_size_bytes,
        "openssh_mode": value.openssh_mode,
        "openssh_version": value.openssh_version,
        "config_file": value.config_file,
        "client_config": list(value.client_config),
        "client_config_sha256": value.client_config_sha256,
        "known_hosts_sha256": value.known_hosts_sha256,
        "target_endpoint_sha256": value.target_endpoint_sha256,
        "credential_mode": value.credential_mode,
    }


def _runtime_document(
    runtime: Any, implementation: TransportImplementationEvidence, *,
    operation_sequence: int, request_id: str, request_nonce: str,
) -> dict[str, Any]:
    if type(runtime) is not RemoteRuntimeObservation or \
            runtime.schema != "e055-remote-runtime-observation/v1" or \
            runtime.operation_sequence != operation_sequence or \
            not _strict_int(runtime.operation_sequence, minimum=1) or \
            runtime.request_id != request_id or runtime.request_nonce != request_nonce:
        raise ValueError("remote runtime request binding is invalid")
    expected_helper = (
        f"{DEPLOYMENT_BASE}/helpers/{implementation.helper_source_sha256}/helper.py"
    )
    if runtime.helper_path != expected_helper or \
            runtime.helper_sha256 != implementation.helper_source_sha256 or \
            runtime.helper_size_bytes != implementation.helper_source_size_bytes or \
            runtime.helper_mode != 0o700 or isinstance(runtime.helper_mode, bool) or \
            runtime.helper_protocol != "e055-remote-helper/v1":
        raise ValueError("remote helper identity does not match local source")
    for value, label, minimum in (
        (runtime.helper_size_bytes, "helper size", 1),
        (runtime.helper_device, "helper device", 0),
        (runtime.helper_inode, "helper inode", 1),
        (runtime.python_size_bytes, "Python size", 1),
        (runtime.python_device, "Python device", 0),
        (runtime.python_inode, "Python inode", 1),
    ):
        if not _strict_int(value, minimum=minimum):
            raise ValueError(f"remote runtime {label} is not a strict integer")
    if runtime.python_size_bytes > 128 * 1024 * 1024 or \
            runtime.python_requested_path != "/usr/bin/python3" or \
            not isinstance(runtime.python_realpath, str) or \
            PYTHON_REALPATH_RE.fullmatch(runtime.python_realpath) is None or \
            not isinstance(runtime.python_version, str) or \
            PYTHON_VERSION_RE.fullmatch(runtime.python_version) is None or \
            not isinstance(runtime.python_sha256, str) or \
            SHA256_RE.fullmatch(runtime.python_sha256) is None or \
            runtime.python_sha256 == "0" * 64 or \
            runtime.python_mode != 0o755 or isinstance(runtime.python_mode, bool):
        raise ValueError("remote Python interpreter identity is invalid")
    if (runtime.helper_device, runtime.helper_inode) == (
        runtime.python_device, runtime.python_inode,
    ):
        raise ValueError("helper and Python interpreter reuse one inode")
    return {
        "schema": runtime.schema,
        "operation_sequence": runtime.operation_sequence,
        "request_id": runtime.request_id,
        "request_nonce": runtime.request_nonce,
        "helper_path": runtime.helper_path,
        "helper_sha256": runtime.helper_sha256,
        "helper_size_bytes": runtime.helper_size_bytes,
        "helper_mode": runtime.helper_mode,
        "helper_device": runtime.helper_device,
        "helper_inode": runtime.helper_inode,
        "helper_protocol": runtime.helper_protocol,
        "python_requested_path": runtime.python_requested_path,
        "python_realpath": runtime.python_realpath,
        "python_version": runtime.python_version,
        "python_sha256": runtime.python_sha256,
        "python_size_bytes": runtime.python_size_bytes,
        "python_mode": runtime.python_mode,
        "python_device": runtime.python_device,
        "python_inode": runtime.python_inode,
    }


def validate_transport_evidence(
    receipt: Any,
    readback: Any,
    layout: DeploymentLayout,
    expected_artifacts: Sequence[Mapping[str, Any]],
    *,
    implementation_evidence: TransportImplementationEvidence,
    expected_pins: ExpectedTransportPins,
    repository_root: Path,
    exclusive_request_id: str,
    exclusive_request_nonce: str,
    readback_request_id: str,
    readback_request_nonce: str,
) -> dict[str, Any]:
    """Cross-check independent deploy/readback results and serialize evidence."""

    if type(layout) is not DeploymentLayout:
        raise ValueError("deployment layout is noncanonical")
    if type(receipt) is not ExclusiveDeploymentReceipt or \
            type(readback) is not FreshReadbackProof:
        raise ValueError("transport must return exact deploy receipt and readback proof")
    request_tokens = (
        exclusive_request_id, exclusive_request_nonce,
        readback_request_id, readback_request_nonce,
    )
    if any(not isinstance(token, str) or REQUEST_TOKEN_RE.fullmatch(token) is None
           for token in request_tokens) or len(set(request_tokens)) != 4:
        raise ValueError("transport request IDs and nonces must be canonical and distinct")
    if receipt.schema != "e055-exclusive-deployment-receipt/v1" or \
            receipt.operation_sequence != 1 or \
            not _strict_int(receipt.operation_sequence, minimum=1) or \
            receipt.request_id != exclusive_request_id or \
            receipt.request_nonce != exclusive_request_nonce or \
            receipt.deployment_root != layout.deployment_root or \
            receipt.lock_path != layout.lock_path or \
            type(receipt.lock_acquired_exclusively) is not bool or \
            receipt.lock_acquired_exclusively is not True or \
            type(receipt.deployment_created_exclusively) is not bool or \
            receipt.deployment_created_exclusively is not True:
        raise ValueError("exclusive deployment receipt is invalid")
    if readback.schema != "e055-fresh-readback-proof/v1" or \
            readback.operation_sequence != 2 or \
            not _strict_int(readback.operation_sequence, minimum=1) or \
            readback.request_id != readback_request_id or \
            readback.request_nonce != readback_request_nonce or \
            readback.deployment_root != layout.deployment_root:
        raise ValueError("fresh readback proof is invalid")
    pins_document = _expected_pins_document(expected_pins)
    receipt_identity = _validate_identity(receipt.endpoint_identity)
    readback_identity = _validate_identity(readback.endpoint_identity)
    if receipt_identity != readback_identity:
        raise ValueError("deploy and readback endpoint identities differ")
    if receipt_identity.trust_mode != expected_pins.trust_mode or \
            receipt_identity.endpoint_label != expected_pins.endpoint_label or \
            receipt_identity.board_identity != expected_pins.board_identity or \
            receipt_identity.host_key_fingerprint != expected_pins.host_key_fingerprint:
        raise ValueError("transport endpoint does not match independent expected pins")
    implementation_document = _implementation_document(implementation_evidence)
    if implementation_evidence.known_hosts_sha256 != expected_pins.known_hosts_sha256 or \
            implementation_evidence.target_endpoint_sha256 != expected_pins.target_endpoint_sha256 or \
            implementation_evidence.helper_source_sha256 != expected_pins.helper_source_sha256 or \
            implementation_evidence.helper_source_size_bytes != expected_pins.helper_source_size_bytes or \
            implementation_evidence.helper_git_blob_oid != expected_pins.helper_git_blob_oid or \
            implementation_evidence.openssh_sha256 != expected_pins.openssh_sha256 or \
            implementation_evidence.openssh_size_bytes != expected_pins.openssh_size_bytes:
        raise ValueError("implementation evidence does not match independent expected pins")
    _verify_local_implementation(
        implementation_evidence, expected_pins, repository_root,
    )
    receipt_runtime = _runtime_document(
        receipt.runtime, implementation_evidence, operation_sequence=1,
        request_id=exclusive_request_id, request_nonce=exclusive_request_nonce,
    )
    readback_runtime = _runtime_document(
        readback.runtime, implementation_evidence, operation_sequence=2,
        request_id=readback_request_id, request_nonce=readback_request_nonce,
    )
    if receipt.runtime is readback.runtime:
        raise ValueError("runtime readback must be a fresh observation")
    runtime_binding_fields = {"operation_sequence", "request_id", "request_nonce"}
    if {
        key: value for key, value in receipt_runtime.items()
        if key not in runtime_binding_fields
    } != {
        key: value for key, value in readback_runtime.items()
        if key not in runtime_binding_fields
    }:
        raise ValueError("deploy and readback runtime identities differ")
    expected_by_role: dict[str, Mapping[str, Any]] = {}
    for expected in expected_artifacts:
        role = expected.get("role")
        if role not in ("harness_executable", "pmu_executable") or \
                role in expected_by_role:
            raise ValueError("expected transport artifact roles are invalid")
        expected_by_role[str(role)] = expected
    if set(expected_by_role) != {"harness_executable", "pmu_executable"} or \
            not isinstance(receipt.artifacts, tuple) or \
            not isinstance(readback.artifacts, tuple) or \
            len(receipt.artifacts) != 2 or len(readback.artifacts) != 2 or \
            receipt.artifacts is readback.artifacts:
        raise ValueError("transport proof must contain two independently observed artifacts")
    receipt_documents: dict[str, dict[str, Any]] = {}
    readback_documents: dict[str, dict[str, Any]] = {}
    for observation in receipt.artifacts:
        if observation.role in receipt_documents or observation.role not in expected_by_role:
            raise ValueError("exclusive receipt has duplicate or unknown artifact roles")
        receipt_documents[observation.role] = _observation_document(
            observation, expected_by_role[observation.role], operation_sequence=1,
            request_id=exclusive_request_id, request_nonce=exclusive_request_nonce,
        )
    for observation in readback.artifacts:
        if observation.role in readback_documents or observation.role not in expected_by_role:
            raise ValueError("readback has duplicate or unknown artifact roles")
        if any(observation is prior for prior in receipt.artifacts):
            raise ValueError("readback must be a fresh transport observation")
        readback_documents[observation.role] = _observation_document(
            observation, expected_by_role[observation.role], operation_sequence=2,
            request_id=readback_request_id, request_nonce=readback_request_nonce,
        )
    observation_binding_fields = {
        "operation_sequence", "request_id", "request_nonce",
    }
    receipt_payloads = {
        role: {key: value for key, value in document.items()
               if key not in observation_binding_fields}
        for role, document in receipt_documents.items()
    }
    readback_payloads = {
        role: {key: value for key, value in document.items()
               if key not in observation_binding_fields}
        for role, document in readback_documents.items()
    }
    if receipt_payloads != readback_payloads:
        raise ValueError("exclusive receipt and fresh readback disagree")
    identities = {
        (item["device"], item["inode"]) for item in receipt_payloads.values()
    }
    if len(identities) != 2:
        raise ValueError("remote artifact inode identity is reused")
    ordered_roles = ("harness_executable", "pmu_executable")
    return {
        "schema": "e055-transport-evidence/v2",
        "trust_statement": (
            "replay-resistant transport evidence only; no cryptographic device attestation; "
            "transport and endpoint remain operationally trusted"
        ),
        "endpoint_identity": _identity_document(receipt_identity),
        "expected_pins": pins_document,
        "implementation_evidence": implementation_document,
        "deployment_root": layout.deployment_root,
        "lock_path": layout.lock_path,
        "requests": {
            "exclusive_deploy": {
                "operation_sequence": 1,
                "request_id": exclusive_request_id,
                "request_nonce": exclusive_request_nonce,
            },
            "fresh_readback": {
                "operation_sequence": 2,
                "request_id": readback_request_id,
                "request_nonce": readback_request_nonce,
            },
        },
        "exclusive_receipt": {
            "schema": receipt.schema,
            "operation_sequence": receipt.operation_sequence,
            "request_id": receipt.request_id,
            "request_nonce": receipt.request_nonce,
            "lock_acquired_exclusively": receipt.lock_acquired_exclusively,
            "deployment_created_exclusively": receipt.deployment_created_exclusively,
            "endpoint_identity": _identity_document(receipt_identity),
            "runtime": receipt_runtime,
            "artifacts": [receipt_documents[role] for role in ordered_roles],
        },
        "fresh_readback": {
            "schema": readback.schema,
            "operation_sequence": readback.operation_sequence,
            "request_id": readback.request_id,
            "request_nonce": readback.request_nonce,
            "endpoint_identity": _identity_document(readback_identity),
            "runtime": readback_runtime,
            "artifacts": [readback_documents[role] for role in ordered_roles],
        },
    }


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain the exact documented fields")
    return value


def validate_serialized_transport_evidence(
    value: Any,
    layout: DeploymentLayout,
    expected_artifacts: Sequence[Mapping[str, Any]],
    *,
    expected_pins: ExpectedTransportPins,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate committed JSON evidence through the same canonical model."""

    exact = _exact_mapping(
        value,
        {
            "schema", "trust_statement", "endpoint_identity", "deployment_root",
            "lock_path", "requests", "exclusive_receipt", "fresh_readback",
            "implementation_evidence", "expected_pins",
        },
        "transport evidence",
    )
    identity_raw = _exact_mapping(
        exact.get("endpoint_identity"),
        {"trust_mode", "endpoint_label", "board_identity", "host_key_fingerprint"},
        "transport endpoint identity",
    )
    def identity_from(raw: Any, label: str) -> EndpointIdentity:
        document = _exact_mapping(
            raw,
            {"trust_mode", "endpoint_label", "board_identity", "host_key_fingerprint"},
            label,
        )
        return EndpointIdentity(
            document.get("trust_mode"), document.get("endpoint_label"),
            document.get("board_identity"), document.get("host_key_fingerprint"),
        )

    identity = identity_from(identity_raw, "transport endpoint identity")

    pins_raw = _exact_mapping(
        exact.get("expected_pins"),
        {
            "schema", "trust_mode", "endpoint_label", "board_identity",
            "host_key_fingerprint", "known_hosts_sha256",
            "target_endpoint_sha256", "helper_source_sha256",
            "helper_source_size_bytes", "helper_git_blob_oid", "openssh_sha256",
            "openssh_size_bytes",
        },
        "expected transport pins",
    )
    serialized_pins = ExpectedTransportPins(
        pins_raw.get("schema"), pins_raw.get("trust_mode"),
        pins_raw.get("endpoint_label"), pins_raw.get("board_identity"),
        pins_raw.get("host_key_fingerprint"), pins_raw.get("known_hosts_sha256"),
        pins_raw.get("target_endpoint_sha256"),
        pins_raw.get("helper_source_sha256"),
        pins_raw.get("helper_source_size_bytes"), pins_raw.get("helper_git_blob_oid"),
        pins_raw.get("openssh_sha256"), pins_raw.get("openssh_size_bytes"),
    )
    if serialized_pins != expected_pins:
        raise ValueError("serialized expected pins differ from caller policy")

    implementation_raw = _exact_mapping(
        exact.get("implementation_evidence"),
        {
            "schema", "implementation", "helper_source_path",
            "helper_source_sha256", "helper_source_size_bytes", "helper_git_blob_oid",
            "openssh_path",
            "openssh_sha256", "openssh_size_bytes", "openssh_mode",
            "openssh_version", "config_file", "client_config",
            "client_config_sha256", "known_hosts_sha256",
            "target_endpoint_sha256", "credential_mode",
        },
        "transport implementation evidence",
    )
    client_config = implementation_raw.get("client_config")
    if not isinstance(client_config, list):
        raise ValueError("serialized OpenSSH client config must be one list")
    implementation = TransportImplementationEvidence(
        implementation_raw.get("schema"), implementation_raw.get("implementation"),
        implementation_raw.get("helper_source_path"),
        implementation_raw.get("helper_source_sha256"),
        implementation_raw.get("helper_source_size_bytes"),
        implementation_raw.get("helper_git_blob_oid"),
        implementation_raw.get("openssh_path"),
        implementation_raw.get("openssh_sha256"),
        implementation_raw.get("openssh_size_bytes"),
        implementation_raw.get("openssh_mode"),
        implementation_raw.get("openssh_version"),
        implementation_raw.get("config_file"), tuple(client_config),
        implementation_raw.get("client_config_sha256"),
        implementation_raw.get("known_hosts_sha256"),
        implementation_raw.get("target_endpoint_sha256"),
        implementation_raw.get("credential_mode"),
    )

    requests_raw = _exact_mapping(
        exact.get("requests"), {"exclusive_deploy", "fresh_readback"},
        "transport requests",
    )
    exclusive_request = _exact_mapping(
        requests_raw.get("exclusive_deploy"),
        {"operation_sequence", "request_id", "request_nonce"},
        "exclusive deployment request",
    )
    readback_request = _exact_mapping(
        requests_raw.get("fresh_readback"),
        {"operation_sequence", "request_id", "request_nonce"},
        "fresh readback request",
    )
    if not _strict_int(exclusive_request.get("operation_sequence"), minimum=1) or \
            exclusive_request.get("operation_sequence") != 1 or \
            not _strict_int(readback_request.get("operation_sequence"), minimum=1) or \
            readback_request.get("operation_sequence") != 2:
        raise ValueError("transport request operation sequence is invalid")

    def observations(raw: Any, label: str) -> tuple[TargetArtifactObservation, ...]:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{label} must contain exactly two observations")
        result: list[TargetArtifactObservation] = []
        for item in raw:
            document = _exact_mapping(
                item,
                {
                    "role", "target_path", "sha256", "size_bytes", "mode",
                    "device", "inode", "operation_sequence", "request_id",
                    "request_nonce",
                },
                f"{label} artifact",
            )
            result.append(TargetArtifactObservation(
                document.get("role"), document.get("target_path"),
                document.get("sha256"), document.get("size_bytes"),
                document.get("mode"), document.get("device"), document.get("inode"),
                document.get("operation_sequence"), document.get("request_id"),
                document.get("request_nonce"),
            ))
        return tuple(result)

    def runtime_observation(raw: Any, label: str) -> RemoteRuntimeObservation:
        document = _exact_mapping(
            raw,
            {
                "schema", "operation_sequence", "request_id", "request_nonce",
                "helper_path", "helper_sha256", "helper_size_bytes", "helper_mode",
                "helper_device", "helper_inode", "helper_protocol",
                "python_requested_path", "python_realpath", "python_version",
                "python_sha256", "python_size_bytes", "python_mode",
                "python_device", "python_inode",
            },
            label,
        )
        return RemoteRuntimeObservation(
            document.get("schema"), document.get("operation_sequence"),
            document.get("request_id"), document.get("request_nonce"),
            document.get("helper_path"), document.get("helper_sha256"),
            document.get("helper_size_bytes"), document.get("helper_mode"),
            document.get("helper_device"), document.get("helper_inode"),
            document.get("helper_protocol"), document.get("python_requested_path"),
            document.get("python_realpath"), document.get("python_version"),
            document.get("python_sha256"), document.get("python_size_bytes"),
            document.get("python_mode"), document.get("python_device"),
            document.get("python_inode"),
        )

    receipt_raw = _exact_mapping(
        exact.get("exclusive_receipt"),
        {
            "schema", "operation_sequence", "lock_acquired_exclusively",
            "request_id", "request_nonce", "deployment_created_exclusively",
            "endpoint_identity", "artifacts", "runtime",
        },
        "exclusive deployment receipt",
    )
    readback_raw = _exact_mapping(
        exact.get("fresh_readback"),
        {
            "schema", "operation_sequence", "request_id", "request_nonce",
            "endpoint_identity", "artifacts", "runtime",
        },
        "fresh readback proof",
    )
    receipt = ExclusiveDeploymentReceipt(
        receipt_raw.get("schema"), receipt_raw.get("operation_sequence"),
        receipt_raw.get("request_id"), receipt_raw.get("request_nonce"),
        exact.get("deployment_root"), exact.get("lock_path"),
        receipt_raw.get("lock_acquired_exclusively"),
        receipt_raw.get("deployment_created_exclusively"),
        identity_from(receipt_raw.get("endpoint_identity"), "receipt endpoint identity"),
        observations(receipt_raw.get("artifacts"), "exclusive receipt"),
        runtime_observation(receipt_raw.get("runtime"), "exclusive runtime"),
    )
    # Construct independent objects deliberately: serialized proof has two
    # separately named observations even when every validated value agrees.
    readback_identity = identity_from(
        readback_raw.get("endpoint_identity"), "readback endpoint identity"
    )
    readback = FreshReadbackProof(
        readback_raw.get("schema"), readback_raw.get("operation_sequence"),
        readback_raw.get("request_id"), readback_raw.get("request_nonce"),
        exact.get("deployment_root"), readback_identity,
        observations(readback_raw.get("artifacts"), "fresh readback"),
        runtime_observation(readback_raw.get("runtime"), "readback runtime"),
    )
    canonical = validate_transport_evidence(
        receipt, readback, layout, expected_artifacts,
        implementation_evidence=implementation,
        expected_pins=expected_pins,
        repository_root=repository_root,
        exclusive_request_id=exclusive_request.get("request_id"),
        exclusive_request_nonce=exclusive_request.get("request_nonce"),
        readback_request_id=readback_request.get("request_id"),
        readback_request_nonce=readback_request.get("request_nonce"),
    )
    if dict(exact) != canonical:
        raise ValueError("transport evidence is not the exact canonical document")
    return canonical
