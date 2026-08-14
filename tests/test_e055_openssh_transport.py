"""Fake-boundary tests for the disabled E055 OpenSSH transport."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tooling.e055_openssh_transport import (
    E055OpenSSHConfig,
    E055OpenSSHTransport,
    OpenSSHTransportError,
)
from tooling.e055_target_executor import TargetArtifact, limited_o3_microgate_plan
from tooling.e055_capture_scaffold import safe_environment
from tooling.e055_raw_bundle import (
    canonical_e049c_launcher_argv,
    canonical_harness_argv,
)
from tooling.e055_transport_evidence import (
    canonical_deployment_layout,
    ExpectedTransportPins,
    validate_transport_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tooling/e055_remote_helper.py"


FAKE_CLIENT = r'''#!/usr/bin/python3
import base64, hashlib, json, os, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
state = json.loads((here / "state.json").read_text(encoding="ascii"))
log = here / "calls.jsonl"
def record(kind, stdin):
    with log.open("a", encoding="ascii") as stream:
        stream.write(json.dumps({"kind": kind, "argv": sys.argv[1:],
                                 "stdin_base64": base64.b64encode(stdin).decode("ascii")},
                                sort_keys=True, separators=(",", ":")) + "\n")
if pathlib.Path(sys.argv[0]).name == "sftp":
    payload = sys.stdin.buffer.read()
    record("sftp", payload)
    calls = [json.loads(line) for line in log.read_text(encoding="ascii").splitlines()]
    sftp_index = sum(item["kind"] == "sftp" for item in calls)
    if state.get("sftp_exit_on_call") == sftp_index:
        raise SystemExit(1)
    raise SystemExit(state.get("sftp_exit", 0))
if sys.argv[1:] == ["-V"]:
    sys.stderr.write("OpenSSH_9.9p2, OpenSSL test\n")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
record("ssh", payload)
if state.get("ssh_exit", 0):
    sys.stderr.write("untrusted remote diagnostic with credential-material\n")
    raise SystemExit(state["ssh_exit"])
request = json.loads(payload.decode("ascii"))
sequence = request["operation_sequence"]
request_id = request["request_id"]
nonce = request["request_nonce"]
board = ("sha256:" + "f" * 64) if state.get("wrong_board") else state["board"]
runtime = {
    "schema": "e055-remote-runtime-observation/v1",
    "operation_sequence": sequence, "request_id": request_id,
    "request_nonce": nonce, "helper_path": state["helper_remote"],
    "helper_sha256": state["helper_sha256"],
    "helper_size_bytes": state["helper_size"], "helper_mode": 448,
    "helper_device": 7, "helper_inode": 9001,
    "helper_protocol": "e055-remote-helper/v1",
    "python_requested_path": "/usr/bin/python3",
    "python_realpath": "/usr/bin/python3.13", "python_version": "3.13.7",
    "python_sha256": "7" * 64, "python_size_bytes": 6839896,
    "python_mode": 493, "python_device": 7, "python_inode": 42,
}
endpoint = {"trust_mode": "pinned_host_key", "endpoint_label": "fixture-a733",
            "board_identity": board, "host_key_fingerprint": None}
def observations():
    result = []
    for index, artifact in enumerate(request["artifacts"], 1):
        digest = artifact["sha256"]
        if state.get("forged_readback") and request["operation"] == "fresh_readback":
            digest = "e" * 64
        result.append({"role": artifact["role"], "target_path": artifact["target_path"],
                       "sha256": digest, "size_bytes": artifact["size_bytes"],
                       "mode": artifact["mode"], "device": 7, "inode": 1000 + index,
                       "operation_sequence": sequence, "request_id": request_id,
                       "request_nonce": nonce})
    return result
operation = request["operation"]
if operation == "exclusive_deploy":
    response = {"schema": "e055-exclusive-deployment-receipt/v1",
                "operation_sequence": sequence, "request_id": request_id,
                "request_nonce": nonce, "deployment_root": request["deployment_root"],
                "lock_path": request["lock_path"], "lock_acquired_exclusively": True,
                "deployment_created_exclusively": True, "endpoint_identity": endpoint,
                "artifacts": observations(), "runtime": runtime}
elif operation == "fresh_readback":
    response = {"schema": "e055-fresh-readback-proof/v1",
                "operation_sequence": sequence, "request_id": request_id,
                "request_nonce": nonce, "deployment_root": request["deployment_root"],
                "endpoint_identity": endpoint, "artifacts": observations(),
                "runtime": runtime}
elif operation == "capture":
    streams = {name: base64.b64encode(value).decode("ascii") for name, value in {
        "child_stdout_base64": b"{}\n", "child_stderr_base64": b"",
        "e049c_json_base64": b"{}\n", "wrapper_stdout_base64": b"",
        "wrapper_stderr_base64": b""}.items()}
    response = {"schema": "e055-capture-result/v1", "operation_sequence": sequence,
                "request_id": request_id, "request_nonce": nonce,
                "run_id": request["run_id"],
                "exit": {"code": state.get("capture_exit", 0),
                         "signal": state.get("capture_signal")},
                "streams": streams, "affinity": {"effective_cpus": [request["cpu"]],
                "cpu_start": request["cpu"], "cpu_end": request["cpu"],
                "migration_count": 0}, "runtime": runtime}
elif operation == "restore":
    response = {"schema": "e055-restore-result/v1", "operation_sequence": sequence,
                "request_id": request_id, "request_nonce": nonce, "restored": True,
                "deployment_removed": True, "lock_released": True, "runtime": runtime}
elif operation == "finalize_helper":
    response = {"schema": "e055-helper-finalization/v1",
                "operation_sequence": sequence, "request_id": request_id,
                "request_nonce": nonce, "helper_removed": True,
                "helper_sha256": state["helper_sha256"],
                "helper_device": 8 if state.get("wrong_helper_identity") else 7,
                "helper_inode": 9002 if state.get("wrong_helper_identity") else 9001}
else:
    raise SystemExit(64)
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
'''


class FakeOpenSSHBoundary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        self.ssh = self.bin / "ssh"
        self.sftp = self.bin / "sftp"
        self.ssh.write_text(FAKE_CLIENT, encoding="ascii")
        self.sftp.write_text(FAKE_CLIENT, encoding="ascii")
        self.ssh.chmod(0o755)
        self.sftp.chmod(0o755)
        host_key = root / "fixture_host_key"
        subprocess.run(
            ["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f",
             host_key.as_posix()], check=True,
        )
        public_fields = (root / "fixture_host_key.pub").read_text(
            encoding="ascii"
        ).split()
        fingerprint_output = subprocess.run(
            ["/usr/bin/ssh-keygen", "-lf", (root / "fixture_host_key.pub").as_posix(),
             "-E", "sha256"], check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.split()
        self.fingerprint = fingerprint_output[1]
        self.known_hosts = root / "known_hosts"
        self.known_hosts.write_text(
            f"[board.test]:2222 {public_fields[0]} {public_fields[1]}\n",
            encoding="ascii",
        )
        self.identity = root / "ephemeral_test_identity"
        self.identity.write_bytes(b"not-a-real-private-key\n")
        self.identity.chmod(0o600)
        self.helper_payload = HELPER.read_bytes()
        self.helper_hash = hashlib.sha256(self.helper_payload).hexdigest()
        self.board = "sha256:" + "a" * 64
        self.update_state()

    def update_state(self, **changes) -> None:
        state_path = self.bin / "state.json"
        state = {
            "helper_sha256": self.helper_hash,
            "helper_size": len(self.helper_payload),
            "helper_remote": f"/tmp/e055-q1-hot-cold/helpers/{self.helper_hash}/helper.py",
            "board": self.board,
        }
        if state_path.exists():
            state.update(json.loads(state_path.read_text(encoding="ascii")))
        state.update(changes)
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="ascii")

    def config_values(self, **changes) -> dict:
        values = dict(
            host="board.test", port=2222, username="fixture-user",
            endpoint_label="fixture-a733", known_hosts_file=self.known_hosts,
            expected_host_key_fingerprint=self.fingerprint,
            expected_board_identity=self.board, ssh_path=self.ssh,
            sftp_path=self.sftp, helper_source_path=HELPER,
            identity_file=self.identity, connect_timeout_seconds=3,
        )
        values.update(changes)
        return values

    def config(self, **changes) -> E055OpenSSHConfig:
        return E055OpenSSHConfig.for_test(**self.config_values(**changes))

    def expected_pins(self) -> ExpectedTransportPins:
        helper_blob = subprocess.run(
            ["git", "rev-parse", "HEAD:tooling/e055_remote_helper.py"], cwd=ROOT,
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        target = hashlib.sha256(
            b"e055-target-endpoint/v1\0board.test\0" + b"2222"
        ).hexdigest()
        ssh_payload = self.ssh.read_bytes()
        sftp_payload = self.sftp.read_bytes()
        return ExpectedTransportPins(
            "e055-expected-transport-pins/v1", "pinned_host_key",
            "fixture-a733", self.board, self.fingerprint,
            hashlib.sha256(self.known_hosts.read_bytes()).hexdigest(), target,
            self.helper_hash, len(self.helper_payload), helper_blob,
            self.ssh.as_posix(), hashlib.sha256(ssh_payload).hexdigest(),
            len(ssh_payload), 0o755,
            self.sftp.as_posix(), hashlib.sha256(sftp_payload).hexdigest(),
            len(sftp_payload), 0o755,
        )

    def calls(self) -> list[dict]:
        path = self.bin / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


class E055OpenSSHTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.boundary = FakeOpenSSHBoundary(Path(self.temporary.name))
        self.transport = E055OpenSSHTransport(self.boundary.config())
        harness = b"harness\n"
        pmu = b"pmu\n"
        self.layout = canonical_deployment_layout(
            "phase-fixture", hashlib.sha256(harness).hexdigest(),
            hashlib.sha256(pmu).hexdigest(),
        )
        self.artifacts = (
            TargetArtifact("harness_executable", self.layout.harness_path, harness,
                           hashlib.sha256(harness).hexdigest(), 0o755),
            TargetArtifact("pmu_executable", self.layout.pmu_path, pmu,
                           hashlib.sha256(pmu).hexdigest(), 0o755),
        )

    def prepare(self):
        return self.transport.prepare(
            self.artifacts, deployment_root=self.layout.deployment_root,
            lock_path=self.layout.lock_path, request_id="1" * 64,
            request_nonce="2" * 64,
        )

    def test_constructor_and_uninjected_executor_path_have_no_subprocess_side_effect(self) -> None:
        E055OpenSSHTransport(self.boundary.config())
        self.assertEqual([], self.boundary.calls())

    def test_operational_config_rejects_injected_ssh_and_sftp_executables(self) -> None:
        with self.assertRaises(ValueError):
            E055OpenSSHConfig(**self.boundary.config_values())

    def test_prepare_and_readback_use_fixed_argv_json_and_pinned_identity(self) -> None:
        receipt = self.prepare()
        proof = self.transport.readback(
            self.artifacts, deployment_root=self.layout.deployment_root,
            request_id="3" * 64, request_nonce="4" * 64,
        )
        self.assertEqual(self.boundary.fingerprint,
                         receipt.endpoint_identity.host_key_fingerprint)
        self.assertEqual(self.boundary.board, proof.endpoint_identity.board_identity)
        with self.assertRaisesRegex(ValueError, "implementation|system"):
            validate_transport_evidence(
                receipt, proof, self.layout,
                tuple({
                    "role": artifact.role, "target_path": artifact.target_path,
                    "sha256": artifact.sha256, "size_bytes": len(artifact.payload),
                    "mode": artifact.mode,
                } for artifact in self.artifacts),
                implementation_evidence=self.transport.implementation_evidence(),
                expected_pins=self.boundary.expected_pins(), repository_root=ROOT,
                exclusive_request_id="1" * 64, exclusive_request_nonce="2" * 64,
                readback_request_id="3" * 64, readback_request_nonce="4" * 64,
            )
        calls = self.boundary.calls()
        self.assertEqual(
            ["sftp", "sftp", "sftp", "sftp", "ssh", "ssh"],
            [item["kind"] for item in calls],
        )
        batch = b"".join(
            base64.b64decode(item["stdin_base64"])
            for item in calls if item["kind"] == "sftp"
        )
        self.assertIn(self.boundary.helper_hash.encode("ascii"), batch)
        self.assertNotIn(b"fixture-user", batch)
        for call in (item for item in calls if item["kind"] == "ssh"):
            argv = call["argv"]
            self.assertIn("-F", argv)
            self.assertIn("/dev/null", argv)
            self.assertEqual(1, sum("exec /usr/bin/python3 -- " in item for item in argv))
            request = json.loads(base64.b64decode(call["stdin_base64"]))
            self.assertNotIn("fixture-user", json.dumps(request))

    def test_metacharacters_newlines_and_path_traversal_reject_before_subprocess(self) -> None:
        for changes in (
            {"host": "board.test;touch /tmp/pwn"}, {"username": "name\n-oProxyCommand=x"},
            {"helper_source_path": ROOT / "tooling/../tooling/e055_remote_helper.py"},
        ):
            with self.assertRaises(ValueError):
                E055OpenSSHTransport(self.boundary.config(**changes))
        self.assertEqual([], self.boundary.calls())

    def test_wrong_host_key_or_board_identity_fail_closed(self) -> None:
        wrong = "SHA256:" + "A" * 43
        transport = E055OpenSSHTransport(
            self.boundary.config(expected_host_key_fingerprint=wrong)
        )
        with self.assertRaises(OpenSSHTransportError):
            transport.prepare(
                self.artifacts, deployment_root=self.layout.deployment_root,
                lock_path=self.layout.lock_path, request_id="1" * 64,
                request_nonce="2" * 64,
            )
        self.assertEqual([], self.boundary.calls())

        self.boundary.update_state(wrong_board=True)
        with self.assertRaises(OpenSSHTransportError):
            self.prepare()

    def test_partial_transfer_disconnect_and_forged_readback_are_sanitized(self) -> None:
        self.boundary.update_state(sftp_exit=1)
        with self.assertRaisesRegex(OpenSSHTransportError, "sftp_bootstrap_failed") as raised:
            self.prepare()
        self.assertNotIn("fixture-user", str(raised.exception))
        self.assertNotIn("credential-material", str(raised.exception))

        self.boundary.update_state(sftp_exit=0, ssh_exit=255)
        self.transport = E055OpenSSHTransport(self.boundary.config())
        with self.assertRaisesRegex(OpenSSHTransportError, "ssh_protocol_failed") as raised:
            self.prepare()
        self.assertNotIn("credential-material", str(raised.exception))

        self.boundary.update_state(ssh_exit=0, forged_readback=True)
        self.transport = E055OpenSSHTransport(self.boundary.config())
        self.prepare()
        with self.assertRaises(OpenSSHTransportError):
            self.transport.readback(
                self.artifacts, deployment_root=self.layout.deployment_root,
                request_id="3" * 64, request_nonce="4" * 64,
            )

    def test_bootstrap_has_separate_owned_stage_and_cleanup_path(self) -> None:
        self.boundary.update_state(sftp_exit_on_call=3)
        with self.assertRaises(OpenSSHTransportError):
            self.prepare()
        batches = [
            base64.b64decode(item["stdin_base64"])
            for item in self.boundary.calls() if item["kind"] == "sftp"
        ]
        stage_mkdirs = [
            batch for batch in batches
            if batch.startswith(b"mkdir ") and batch.count(b"\n") == 1
        ]
        self.assertEqual(1, len(stage_mkdirs))
        self.assertIn(b"rmdir ", batches[-1])

    def test_initial_bootstrap_failure_enters_bounded_cleanup(self) -> None:
        self.boundary.update_state(sftp_exit_on_call=1)
        with self.assertRaises(OpenSSHTransportError):
            self.prepare()
        self.assertTrue(self.transport.bootstrap_cleanup_attempted)

    def test_stage_collision_never_removes_unowned_path(self) -> None:
        self.boundary.update_state(sftp_exit_on_call=2)
        with self.assertRaises(OpenSSHTransportError):
            self.prepare()
        batches = [
            base64.b64decode(item["stdin_base64"])
            for item in self.boundary.calls() if item["kind"] == "sftp"
        ]
        self.assertEqual(2, len(batches))
        self.assertFalse(any(b"-rm " in batch or b"-rmdir " in batch
                             for batch in batches))

    def test_restore_response_is_retained_before_separate_finalization(self) -> None:
        self.prepare()
        self.transport.readback(
            self.artifacts, deployment_root=self.layout.deployment_root,
            request_id="3" * 64, request_nonce="4" * 64,
        )
        self.transport.restore()
        self.assertEqual("e055-restore-result/v1",
                         self.transport.retained_restore_evidence["schema"])
        operations = []
        for call in self.boundary.calls():
            if call["kind"] == "ssh":
                operations.append(json.loads(base64.b64decode(call["stdin_base64"]))["operation"])
        self.assertEqual(
            ["exclusive_deploy", "fresh_readback", "restore", "finalize_helper"],
            operations,
        )

    def test_finalization_rejects_wrong_helper_device_and_inode(self) -> None:
        self.prepare()
        self.transport.readback(
            self.artifacts, deployment_root=self.layout.deployment_root,
            request_id="3" * 64, request_nonce="4" * 64,
        )
        self.boundary.update_state(wrong_helper_identity=True)
        with self.assertRaisesRegex(OpenSSHTransportError, "finalization"):
            self.transport.restore()

    def test_implementation_evidence_is_exact_and_contains_no_login_or_key_path(self) -> None:
        evidence = self.transport.implementation_evidence()
        serialized = json.dumps(evidence.__dict__, default=list, sort_keys=True)
        self.assertEqual("openssh_fixed_helper", evidence.implementation)
        self.assertEqual(self.boundary.helper_hash, evidence.helper_source_sha256)
        self.assertEqual("OpenSSH_9.9p2", evidence.openssh_version)
        self.assertEqual(self.boundary.sftp.as_posix(), evidence.sftp_path)
        self.assertEqual(
            hashlib.sha256(self.boundary.sftp.read_bytes()).hexdigest(),
            evidence.sftp_sha256,
        )
        self.assertNotIn("fixture-user", serialized)
        self.assertNotIn(self.boundary.identity.as_posix(), serialized)
        self.assertNotIn("fixture-user", repr(self.boundary.config()))
        self.assertNotIn(self.boundary.identity.as_posix(), repr(self.boundary.config()))

    def test_capture_transports_exact_argv_and_preserves_remote_signal(self) -> None:
        self.prepare()
        run = limited_o3_microgate_plan()[0]
        harness = canonical_harness_argv(run.cell(), self.layout.harness_path, run.iterations)
        e049c = canonical_e049c_launcher_argv(
            run.cell(),
            f"{self.layout.deployment_root}/captures/{run.run_id}/e049c.json",
            harness, self.layout.pmu_path,
        )
        self.boundary.update_state(capture_signal=15)
        capture = self.transport.capture(
            run=run, e049c_argv=e049c, environment=safe_environment("O3")
        )
        self.assertEqual(15, capture.signal)
        self.assertEqual((run.cpu,), capture.effective_cpus)
        request = json.loads(base64.b64decode([
            item for item in self.boundary.calls() if item["kind"] == "ssh"
        ][-1]["stdin_base64"]))
        self.assertEqual(list(e049c), request["argv"])
        self.assertEqual({"E055_BUILD_NAME": "O3", "LANG": "C", "LC_ALL": "C"},
                         request["environment"])


if __name__ == "__main__":
    unittest.main()
