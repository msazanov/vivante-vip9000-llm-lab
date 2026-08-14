"""Fail-closed transport evidence for the bounded E055 target phase.

The receipt and readback records prove what an injected transport reported at
two separate API boundaries. They are not cryptographic device attestation;
the transport implementation and its endpoint remain in the trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PHASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
RUN_ID_RE = PHASE_ID_RE
HOST_KEY_RE = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
DEPLOYMENT_BASE = "/tmp/e055-q1-hot-cold"


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


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


@dataclass(frozen=True)
class ExclusiveDeploymentReceipt:
    schema: str
    operation_sequence: int
    deployment_root: str
    lock_path: str
    lock_acquired_exclusively: bool
    deployment_created_exclusively: bool
    endpoint_identity: EndpointIdentity
    artifacts: tuple[TargetArtifactObservation, ...]


@dataclass(frozen=True)
class FreshReadbackProof:
    schema: str
    operation_sequence: int
    deployment_root: str
    endpoint_identity: EndpointIdentity
    artifacts: tuple[TargetArtifactObservation, ...]


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
    observation: Any, expected: Mapping[str, Any],
) -> dict[str, Any]:
    if type(observation) is not TargetArtifactObservation:
        raise ValueError("transport artifact observation has a noncanonical type")
    if observation.role != expected.get("role") or \
            observation.target_path != expected.get("target_path") or \
            observation.sha256 != expected.get("sha256") or \
            observation.size_bytes != expected.get("size_bytes") or \
            observation.mode != expected.get("mode"):
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
    }


def validate_transport_evidence(
    receipt: Any,
    readback: Any,
    layout: DeploymentLayout,
    expected_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check independent deploy/readback results and serialize evidence."""

    if type(layout) is not DeploymentLayout:
        raise ValueError("deployment layout is noncanonical")
    if type(receipt) is not ExclusiveDeploymentReceipt or \
            type(readback) is not FreshReadbackProof:
        raise ValueError("transport must return exact deploy receipt and readback proof")
    if receipt.schema != "e055-exclusive-deployment-receipt/v1" or \
            receipt.operation_sequence != 1 or \
            not _strict_int(receipt.operation_sequence, minimum=1) or \
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
            readback.deployment_root != layout.deployment_root:
        raise ValueError("fresh readback proof is invalid")
    receipt_identity = _validate_identity(receipt.endpoint_identity)
    readback_identity = _validate_identity(readback.endpoint_identity)
    if receipt_identity != readback_identity:
        raise ValueError("deploy and readback endpoint identities differ")
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
            observation, expected_by_role[observation.role]
        )
    for observation in readback.artifacts:
        if observation.role in readback_documents or observation.role not in expected_by_role:
            raise ValueError("readback has duplicate or unknown artifact roles")
        if any(observation is prior for prior in receipt.artifacts):
            raise ValueError("readback must be a fresh transport observation")
        readback_documents[observation.role] = _observation_document(
            observation, expected_by_role[observation.role]
        )
    if receipt_documents != readback_documents:
        raise ValueError("exclusive receipt and fresh readback disagree")
    identities = {
        (item["device"], item["inode"]) for item in receipt_documents.values()
    }
    if len(identities) != 2:
        raise ValueError("remote artifact inode identity is reused")
    ordered_roles = ("harness_executable", "pmu_executable")
    return {
        "schema": "e055-transport-evidence/v1",
        "trust_statement": (
            "transport evidence only; no cryptographic device attestation; "
            "transport and endpoint remain operationally trusted"
        ),
        "endpoint_identity": _identity_document(receipt_identity),
        "deployment_root": layout.deployment_root,
        "lock_path": layout.lock_path,
        "exclusive_receipt": {
            "schema": receipt.schema,
            "operation_sequence": receipt.operation_sequence,
            "lock_acquired_exclusively": receipt.lock_acquired_exclusively,
            "deployment_created_exclusively": receipt.deployment_created_exclusively,
            "endpoint_identity": _identity_document(receipt_identity),
            "artifacts": [receipt_documents[role] for role in ordered_roles],
        },
        "fresh_readback": {
            "schema": readback.schema,
            "operation_sequence": readback.operation_sequence,
            "endpoint_identity": _identity_document(readback_identity),
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
) -> dict[str, Any]:
    """Validate committed JSON evidence through the same canonical model."""

    exact = _exact_mapping(
        value,
        {
            "schema", "trust_statement", "endpoint_identity", "deployment_root",
            "lock_path", "exclusive_receipt", "fresh_readback",
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

    def observations(raw: Any, label: str) -> tuple[TargetArtifactObservation, ...]:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{label} must contain exactly two observations")
        result: list[TargetArtifactObservation] = []
        for item in raw:
            document = _exact_mapping(
                item,
                {"role", "target_path", "sha256", "size_bytes", "mode", "device", "inode"},
                f"{label} artifact",
            )
            result.append(TargetArtifactObservation(
                document.get("role"), document.get("target_path"),
                document.get("sha256"), document.get("size_bytes"),
                document.get("mode"), document.get("device"), document.get("inode"),
            ))
        return tuple(result)

    receipt_raw = _exact_mapping(
        exact.get("exclusive_receipt"),
        {
            "schema", "operation_sequence", "lock_acquired_exclusively",
            "deployment_created_exclusively", "endpoint_identity", "artifacts",
        },
        "exclusive deployment receipt",
    )
    readback_raw = _exact_mapping(
        exact.get("fresh_readback"),
        {"schema", "operation_sequence", "endpoint_identity", "artifacts"},
        "fresh readback proof",
    )
    receipt = ExclusiveDeploymentReceipt(
        receipt_raw.get("schema"), receipt_raw.get("operation_sequence"),
        exact.get("deployment_root"), exact.get("lock_path"),
        receipt_raw.get("lock_acquired_exclusively"),
        receipt_raw.get("deployment_created_exclusively"),
        identity_from(receipt_raw.get("endpoint_identity"), "receipt endpoint identity"),
        observations(receipt_raw.get("artifacts"), "exclusive receipt"),
    )
    # Construct independent objects deliberately: serialized proof has two
    # separately named observations even when every validated value agrees.
    readback_identity = identity_from(
        readback_raw.get("endpoint_identity"), "readback endpoint identity"
    )
    readback = FreshReadbackProof(
        readback_raw.get("schema"), readback_raw.get("operation_sequence"),
        exact.get("deployment_root"), readback_identity,
        observations(readback_raw.get("artifacts"), "fresh readback"),
    )
    canonical = validate_transport_evidence(
        receipt, readback, layout, expected_artifacts
    )
    if dict(exact) != canonical:
        raise ValueError("transport evidence is not the exact canonical document")
    return canonical
