"""Adversarial contracts for E055 OpenSSH/helper runtime provenance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tooling.e055_transport_evidence import (
    EndpointIdentity,
    ExpectedTransportPins,
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
        source_root = Path(__file__).resolve().parents[1]
        temporary = tempfile.TemporaryDirectory(prefix="e055-runtime-")
        self.addCleanup(temporary.cleanup)
        self.repository_root = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repository_root, check=True)
        subprocess.run(["git", "config", "user.name", "E055 Runtime Test"],
                       cwd=self.repository_root, check=True)
        subprocess.run(["git", "config", "user.email", "e055@example.invalid"],
                       cwd=self.repository_root, check=True)
        helper = self.repository_root / "tooling/e055_remote_helper.py"
        helper.parent.mkdir()
        shutil.copyfile(source_root / "tooling/e055_remote_helper.py", helper)
        subprocess.run(["git", "add", "tooling/e055_remote_helper.py"],
                       cwd=self.repository_root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "pin helper fixture"],
                       cwd=self.repository_root, check=True)
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
        helper = self.repository_root / "tooling/e055_remote_helper.py"
        helper_payload = helper.read_bytes()
        helper_blob = subprocess.run(
            ["git", "rev-parse", "HEAD:tooling/e055_remote_helper.py"],
            cwd=self.repository_root, check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        openssh = Path("/usr/bin/ssh")
        openssh_payload = openssh.read_bytes()
        sftp = Path("/usr/bin/sftp")
        sftp_payload = sftp.read_bytes()
        self.client_config = (
            "BatchMode=yes", "StrictHostKeyChecking=yes",
            "UserKnownHostsFile=external-pinned-file",
            "GlobalKnownHostsFile=/dev/null", "CheckHostIP=no",
            "PasswordAuthentication=no", "KbdInteractiveAuthentication=no",
            "NumberOfPasswordPrompts=0", "ForwardAgent=no",
            "ClearAllForwardings=yes", "PermitLocalCommand=no", "RequestTTY=no",
            "ConnectTimeout=10", "ConnectionAttempts=1", "ServerAliveInterval=5",
            "ServerAliveCountMax=2", "LogLevel=ERROR", "IdentitiesOnly=yes",
            "ProxyCommand=none", "ProxyJump=none", "CanonicalizeHostname=no",
        )
        self.known_hosts_sha256 = "1" * 64
        self.target_endpoint_sha256 = "2" * 64
        self.implementation = TransportImplementationEvidence(
            schema="e055-openssh-implementation/v1",
            implementation="openssh_fixed_helper",
            client_execution_mode="operational_system_clients",
            helper_source_path="tooling/e055_remote_helper.py",
            helper_source_sha256=hashlib.sha256(helper_payload).hexdigest(),
            helper_source_size_bytes=len(helper_payload),
            helper_git_blob_oid=helper_blob,
            openssh_path="/usr/bin/ssh",
            openssh_sha256=hashlib.sha256(openssh_payload).hexdigest(),
            openssh_size_bytes=len(openssh_payload),
            openssh_mode=0o755,
            openssh_version="OpenSSH_9.9p2",
            sftp_path="/usr/bin/sftp",
            sftp_sha256=hashlib.sha256(sftp_payload).hexdigest(),
            sftp_size_bytes=len(sftp_payload),
            sftp_mode=0o755,
            config_file="/dev/null",
            client_config=self.client_config,
            client_config_sha256=hashlib.sha256(
                ("\n".join(self.client_config) + "\n").encode("ascii")
            ).hexdigest(),
            known_hosts_sha256=self.known_hosts_sha256,
            target_endpoint_sha256=self.target_endpoint_sha256,
            credential_mode="external_agent_or_identity",
        )
        self.expected_pins = ExpectedTransportPins(
            "e055-expected-transport-pins/v1", "pinned_host_key",
            self.identity.endpoint_label, self.identity.board_identity,
            self.identity.host_key_fingerprint, self.known_hosts_sha256,
            self.target_endpoint_sha256, self.implementation.helper_source_sha256,
            self.implementation.helper_source_size_bytes, helper_blob,
            self.implementation.openssh_path, self.implementation.openssh_sha256,
            self.implementation.openssh_size_bytes, self.implementation.openssh_mode,
            self.implementation.sftp_path, self.implementation.sftp_sha256,
            self.implementation.sftp_size_bytes, self.implementation.sftp_mode,
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
                "/tmp/e055-q1-hot-cold/helpers/"
                + self.implementation.helper_source_sha256 + "/helper.py"
            ),
            helper_sha256=self.implementation.helper_source_sha256,
            helper_size_bytes=self.implementation.helper_source_size_bytes,
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
            expected_pins=self.expected_pins,
            repository_root=self.repository_root,
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
            self.implementation.helper_source_sha256,
        )
        self.assertEqual(
            document["implementation_evidence"]["sftp_sha256"],
            self.implementation.sftp_sha256,
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
                client_config=self.implementation.client_config + ("UnreviewedOption=value",),
            ),
            replace(self.implementation, credential_mode="embedded_password"),
            replace(self.implementation, openssh_sha256="0" * 64),
            replace(self.implementation, sftp_path="/tmp/fake-sftp"),
            replace(self.implementation, sftp_sha256="0" * 64),
            replace(self.implementation, sftp_mode=True),
        )
        for implementation in mutations:
            with self.subTest(implementation=implementation):
                with self.assertRaises(ValueError):
                    validate_transport_evidence(
                        receipt, readback, self.layout, self.expected,
                        implementation_evidence=implementation,
                        expected_pins=self.expected_pins,
                        repository_root=self.repository_root,
                        exclusive_request_id=self.exclusive_id,
                        exclusive_request_nonce=self.exclusive_nonce,
                        readback_request_id=self.readback_id,
                        readback_request_nonce=self.readback_nonce,
                    )

    def test_insecure_known_host_proxy_and_timeout_semantics_fail_closed(self) -> None:
        receipt, readback = self.evidence()
        def changed(key: str, value: str) -> tuple[str, ...]:
            return tuple(
                f"{key}={value}" if option.startswith(key + "=") else option
                for option in self.client_config
            )
        insecure_sets = (
            changed("StrictHostKeyChecking", "accept-new"),
            changed("ProxyCommand", "nc attacker 22"),
            changed("ConnectTimeout", "0"),
        )
        for options in insecure_sets:
            implementation = replace(
                self.implementation, client_config=options,
                client_config_sha256=hashlib.sha256(
                    ("\n".join(options) + "\n").encode("ascii")
                ).hexdigest(),
            )
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    validate_transport_evidence(
                        receipt, readback, self.layout, self.expected,
                        implementation_evidence=implementation,
                        expected_pins=self.expected_pins,
                        repository_root=self.repository_root,
                        exclusive_request_id=self.exclusive_id,
                        exclusive_request_nonce=self.exclusive_nonce,
                        readback_request_id=self.readback_id,
                        readback_request_nonce=self.readback_nonce,
                    )

    def test_operational_or_self_asserted_endpoint_identity_is_not_a_pin(self) -> None:
        receipt, readback = self.evidence()
        forged = EndpointIdentity(
            "operationally_trusted", "attacker-selected", "attacker-selected", None
        )
        with self.assertRaises(ValueError):
            self.validate(
                replace(receipt, endpoint_identity=forged),
                replace(readback, endpoint_identity=forged),
            )

    def test_recomputed_claimed_helper_and_openssh_hashes_are_not_expected_pins(self) -> None:
        receipt, readback = self.evidence()
        forged_implementation = replace(
            self.implementation, helper_source_sha256="6" * 64,
            openssh_sha256="5" * 64,
        )
        forged_receipt = replace(
            receipt, runtime=replace(
                receipt.runtime, helper_path=(
                    "/tmp/e055-q1-hot-cold/helpers/" + "6" * 64 + "/helper.py"
                ), helper_sha256="6" * 64,
            ),
        )
        forged_readback = replace(
            readback, runtime=replace(
                readback.runtime, helper_path=(
                    "/tmp/e055-q1-hot-cold/helpers/" + "6" * 64 + "/helper.py"
                ), helper_sha256="6" * 64,
            ),
        )
        with self.assertRaises(ValueError):
            validate_transport_evidence(
                forged_receipt, forged_readback, self.layout, self.expected,
                implementation_evidence=forged_implementation,
                expected_pins=self.expected_pins,
                repository_root=self.repository_root,
                exclusive_request_id=self.exclusive_id,
                exclusive_request_nonce=self.exclusive_nonce,
                readback_request_id=self.readback_id,
                readback_request_nonce=self.readback_nonce,
            )

    def test_outer_raw_and_runner_schemas_explicitly_migrate_to_v2(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw_schema = json.loads((
            root / "experiments/E055-q1-hot-cold/data/raw-bundle.schema.json"
        ).read_text(encoding="utf-8"))
        runner_schema = json.loads((
            root / "experiments/E055-q1-hot-cold/data/runner-capture.schema.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual("e055-raw-bundle/v2", raw_schema["properties"]["schema"]["const"])
        self.assertEqual("e055-runner-capture/v2",
                         runner_schema["properties"]["schema"]["const"])


if __name__ == "__main__":
    unittest.main()
