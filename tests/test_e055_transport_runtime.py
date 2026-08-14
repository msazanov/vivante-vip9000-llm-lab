"""Adversarial contracts for E055 OpenSSH/helper runtime provenance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from tooling.e055_transport_evidence import (
    EndpointIdentity,
    ExclusiveDeploymentReceipt,
    FreshReadbackProof,
    RemoteRuntimeObservation,
    TargetArtifactObservation,
    TransportImplementationEvidence,
    canonical_deployment_layout,
    validate_transport_evidence,
)


class E055TransportRuntimeEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.exclusive_id = "1" * 64
        self.exclusive_nonce = "2" * 64
        self.readback_id = "3" * 64
        self.readback_nonce = "4" * 64
        self.layout = canonical_deployment_layout(
            "phase-openssh", "a" * 64, "b" * 64
        )
        self.expected = (
            {
                "role": "harness_executable",
                "target_path": self.layout.harness_path,
                "sha256": "a" * 64,
                "size_bytes": 75240,
                "mode": 0o755,
            },
            {
                "role": "pmu_executable",
                "target_path": self.layout.pmu_path,
                "sha256": "b" * 64,
                "size_bytes": 74752,
                "mode": 0o755,
            },
        )
        self.identity = EndpointIdentity(
            "pinned_host_key",
            "orangepi-zero3w-lab",
            "sha256:" + "c" * 64,
            "SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )
        self.implementation = TransportImplementationEvidence(
            schema="e055-openssh-implementation/v1",
            implementation="openssh_fixed_helper",
            helper_source_path="tooling/e055_remote_helper.py",
            helper_source_sha256="d" * 64,
            helper_source_size_bytes=32768,
            openssh_path="/usr/bin/ssh",
            openssh_sha256="e" * 64,
            openssh_size_bytes=112233,
            openssh_mode=0o755,
            openssh_version="OpenSSH_9.9p2",
            client_config=(
                "BatchMode=yes",
                "StrictHostKeyChecking=yes",
                "UserKnownHostsFile=external-pinned-file",
            ),
            client_config_sha256=hashlib.sha256(
                b"BatchMode=yes\nStrictHostKeyChecking=yes\n"
                b"UserKnownHostsFile=external-pinned-file\n"
            ).hexdigest(),
            credential_mode="external_agent_or_identity",
        )

    def runtime(
        self, sequence: int, request_id: str, nonce: str,
    ) -> RemoteRuntimeObservation:
        return RemoteRuntimeObservation(
            schema="e055-remote-runtime-observation/v1",
            operation_sequence=sequence,
            request_id=request_id,
            request_nonce=nonce,
            helper_path=(
                "/tmp/e055-q1-hot-cold/helpers/" + "d" * 64 + "/helper.py"
            ),
            helper_sha256="d" * 64,
            helper_size_bytes=32768,
            helper_mode=0o700,
            helper_device=7,
            helper_inode=9001,
            helper_protocol="e055-remote-helper/v1",
            python_requested_path="/usr/bin/python3",
            python_realpath="/usr/bin/python3.13",
            python_version="3.13.7",
            python_sha256="f" * 64,
            python_size_bytes=6839896,
            python_mode=0o755,
            python_device=7,
            python_inode=42,
        )

    def artifact_observations(
        self, sequence: int, request_id: str, nonce: str,
    ) -> tuple[TargetArtifactObservation, ...]:
        return tuple(
            TargetArtifactObservation(
                role=item["role"], target_path=item["target_path"],
                sha256=item["sha256"], size_bytes=item["size_bytes"],
                mode=item["mode"], device=7, inode=100 + index,
                operation_sequence=sequence, request_id=request_id,
                request_nonce=nonce,
            )
            for index, item in enumerate(self.expected, 1)
        )

    def evidence(self):
        receipt = ExclusiveDeploymentReceipt(
            "e055-exclusive-deployment-receipt/v1", 1,
            self.exclusive_id, self.exclusive_nonce,
            self.layout.deployment_root, self.layout.lock_path,
            True, True, self.identity,
            self.artifact_observations(1, self.exclusive_id, self.exclusive_nonce),
            self.runtime(1, self.exclusive_id, self.exclusive_nonce),
        )
        readback = FreshReadbackProof(
            "e055-fresh-readback-proof/v1", 2,
            self.readback_id, self.readback_nonce,
            self.layout.deployment_root, self.identity,
            self.artifact_observations(2, self.readback_id, self.readback_nonce),
            self.runtime(2, self.readback_id, self.readback_nonce),
        )
        return receipt, readback

    def validate(self, receipt, readback):
        return validate_transport_evidence(
            receipt, readback, self.layout, self.expected,
            implementation_evidence=self.implementation,
            exclusive_request_id=self.exclusive_id,
            exclusive_request_nonce=self.exclusive_nonce,
            readback_request_id=self.readback_id,
            readback_request_nonce=self.readback_nonce,
        )

    def test_valid_runtime_is_bound_into_v2_transport_evidence(self) -> None:
        receipt, readback = self.evidence()
        document = self.validate(receipt, readback)
        self.assertEqual(document["schema"], "e055-transport-evidence/v2")
        self.assertEqual(
            document["implementation_evidence"]["helper_source_sha256"],
            "d" * 64,
        )
        self.assertEqual(
            document["fresh_readback"]["runtime"]["python_realpath"],
            "/usr/bin/python3.13",
        )

    def test_runtime_readback_mismatch_and_forged_python_fail_closed(self) -> None:
        receipt, readback = self.evidence()
        mutations = (
            replace(readback.runtime, helper_inode=receipt.runtime.helper_inode + 1),
            replace(readback.runtime, python_sha256="0" * 64),
            replace(readback.runtime, python_realpath="/tmp/python3"),
            replace(readback.runtime, python_mode=True),
        )
        for runtime in mutations:
            with self.subTest(runtime=runtime):
                with self.assertRaises(ValueError):
                    self.validate(receipt, replace(readback, runtime=runtime))

    def test_implementation_rejects_secret_or_unreviewed_client_config(self) -> None:
        receipt, readback = self.evidence()
        mutations = (
            replace(
                self.implementation,
                client_config=self.implementation.client_config + ("SSHPASS=hunter2",),
            ),
            replace(self.implementation, credential_mode="embedded_password"),
            replace(self.implementation, openssh_sha256="0" * 64),
        )
        for implementation in mutations:
            with self.subTest(implementation=implementation):
                with self.assertRaises(ValueError):
                    validate_transport_evidence(
                        receipt, readback, self.layout, self.expected,
                        implementation_evidence=implementation,
                        exclusive_request_id=self.exclusive_id,
                        exclusive_request_nonce=self.exclusive_nonce,
                        readback_request_id=self.readback_id,
                        readback_request_nonce=self.readback_nonce,
                    )


if __name__ == "__main__":
    unittest.main()
