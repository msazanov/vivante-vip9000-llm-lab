"""Adversarial protocol tests for the fixed E055 remote helper."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from tooling.e055_remote_helper import HelperContext, ProtocolError, handle_request


HELPER_SOURCE = Path(__file__).resolve().parents[1] / "tooling/e055_remote_helper.py"
BASE = "/tmp/e055-q1-hot-cold"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class E055RemoteHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.sandbox = Path(self.temporary.name)
        self.helper_payload = HELPER_SOURCE.read_bytes()
        self.helper_hash = sha256(self.helper_payload)
        self.helper_remote = f"{BASE}/helpers/{self.helper_hash}/helper.py"
        self.context = HelperContext.for_test(
            filesystem_root=self.sandbox,
            helper_source=HELPER_SOURCE,
            helper_remote_path=self.helper_remote,
            board_identity_payloads=(b"orange-pi-zero-3w\n", b"a733\n"),
        )
        helper_local = self.context.local_path(self.helper_remote)
        helper_local.parent.mkdir(parents=True)
        helper_local.write_bytes(self.helper_payload)
        os.chmod(helper_local, 0o700)
        self.board_digest = self.context.board_identity_digest
        self.phase_hash = "1" * 64
        self.deployment_root = f"{BASE}/deployments/{self.phase_hash}"
        self.lock_path = f"{BASE}/locks/a733-target.lock"
        self.owner_id = "2" * 64
        self.artifact_payloads = {
            "harness_executable": b"harness-executable\n",
            "pmu_executable": b"pmu-executable\n",
        }

    def artifact_declarations(self, *, include_payload: bool) -> list[dict]:
        result = []
        names = {
            "harness_executable": "e055-O3-aarch64",
            "pmu_executable": "a733-pmu-exec-aarch64",
        }
        for role, payload in self.artifact_payloads.items():
            digest = sha256(payload)
            artifact = {
                "role": role,
                "target_path": (
                    f"{self.deployment_root}/artifacts/{digest}/{names[role]}"
                ),
                "sha256": digest,
                "size_bytes": len(payload),
                "mode": 0o755,
            }
            if include_payload:
                artifact["payload_base64"] = base64.b64encode(payload).decode("ascii")
            result.append(artifact)
        return result

    def request(self, operation: str, sequence: int, nonce_digit: str) -> dict:
        request = {
            "schema": "e055-remote-request/v1",
            "operation": operation,
            "operation_sequence": sequence,
            "request_id": nonce_digit * 64,
            "request_nonce": chr(ord(nonce_digit) + 1) * 64,
            "expected_board_identity": self.board_digest,
            "owner_id": self.owner_id,
            "deployment_root": self.deployment_root,
        }
        if operation == "exclusive_deploy":
            request.update({
                "lock_path": self.lock_path,
                "artifacts": self.artifact_declarations(include_payload=True),
            })
        elif operation == "fresh_readback":
            request["artifacts"] = self.artifact_declarations(include_payload=False)
        elif operation in ("restore", "finalize_helper"):
            request["lock_path"] = self.lock_path
        return request

    def capture_request(self, *, cpu: int = 0, run_id: str = "cpu0-pair1-hot") -> dict:
        declarations = self.artifact_declarations(include_payload=False)
        paths = {item["role"]: item["target_path"] for item in declarations}
        output = f"{self.deployment_root}/captures/{run_id}/e049c.json"
        harness = (
            "taskset", "-c", str(cpu), paths["harness_executable"],
            "--mode", "full_dotprod", "--cache-state", "hot_repeat",
            "--working-set-bytes", "65536", "--cpu", str(cpu),
            "--iterations", "250", "--warmup", "16", "--thrash-bytes",
            "67108864", "--sync",
        )
        argv = (
            paths["pmu_executable"], "-o", output,
            "--child-stdout", output.replace("e049c.json", "target-harness.stdout.raw"),
            "--child-stderr", output.replace("e049c.json", "target-harness.stderr.raw"),
            "--event-group", "core", "--min-running-ratio", "0.95",
            "--start-on-ready", "--sync-timeout-ms", "5000",
            "--max-temp-c", "85", "--", *harness,
        )
        request = {
            "schema": "e055-remote-request/v1", "operation": "capture",
            "operation_sequence": 3, "request_id": "7" * 64,
            "request_nonce": "8" * 64,
            "expected_board_identity": self.board_digest,
            "owner_id": self.owner_id, "deployment_root": self.deployment_root,
            "run_id": run_id, "cpu": cpu, "argv": list(argv),
            "environment": {"E055_BUILD_NAME": "O3", "LANG": "C", "LC_ALL": "C"},
            "timeout_ms": 3000,
        }
        return request

    def deploy_fake_capture_programs(self, harness_body: str) -> None:
        harness = (
            "#!/usr/bin/python3\nimport json, time\n" + harness_body + "\n"
        ).encode("ascii")
        wrapper = b"""#!/usr/bin/python3
import json, subprocess, sys
args = sys.argv[1:]
output = args[args.index('-o') + 1]
child_out = args[args.index('--child-stdout') + 1]
child_err = args[args.index('--child-stderr') + 1]
child = args[args.index('--') + 1:]
with open(child_out, 'xb') as stdout, open(child_err, 'xb') as stderr:
    process = subprocess.Popen(child, stdout=stdout, stderr=stderr)
    status = process.wait()
with open(output, 'x', encoding='ascii') as stream:
    json.dump({'schema_version': 'test-e049c', 'pid': process.pid}, stream)
    stream.write('\\n')
raise SystemExit(status)
"""
        self.artifact_payloads = {
            "harness_executable": harness,
            "pmu_executable": wrapper,
        }
        handle_request(self.request("exclusive_deploy", 1, "3"), self.context)

    def test_exclusive_deploy_and_fresh_readback_are_exact_and_distinct(self) -> None:
        receipt = handle_request(self.request("exclusive_deploy", 1, "3"), self.context)
        proof = handle_request(self.request("fresh_readback", 2, "5"), self.context)

        self.assertEqual("e055-exclusive-deployment-receipt/v1", receipt["schema"])
        self.assertEqual("e055-fresh-readback-proof/v1", proof["schema"])
        self.assertEqual(1, receipt["operation_sequence"])
        self.assertEqual(2, proof["operation_sequence"])
        self.assertNotEqual(receipt["request_nonce"], proof["request_nonce"])
        self.assertEqual(self.board_digest, receipt["endpoint_identity"]["board_identity"])
        self.assertEqual(self.helper_hash, receipt["runtime"]["helper_sha256"])
        self.assertEqual("/usr/bin/python3", receipt["runtime"]["python_requested_path"])
        for first, second in zip(receipt["artifacts"], proof["artifacts"], strict=True):
            for key in ("role", "target_path", "sha256", "size_bytes", "mode",
                        "device", "inode"):
                self.assertEqual(first[key], second[key])
            self.assertEqual(1, first["operation_sequence"])
            self.assertEqual(2, second["operation_sequence"])

    def test_invalid_paths_types_and_unknown_keys_have_no_side_effects(self) -> None:
        invalid = self.request("exclusive_deploy", 1, "3")
        invalid["deployment_root"] = f"{BASE}/deployments/../escape"
        invalid["unexpected"] = "$(touch /tmp/not-allowed)\n"
        invalid["operation_sequence"] = True
        with self.assertRaises(ProtocolError):
            handle_request(invalid, self.context)
        self.assertFalse(self.context.local_path(self.deployment_root).exists())
        self.assertFalse(self.context.local_path(self.lock_path).exists())

    def test_lock_and_deployment_collisions_preserve_preexisting_data(self) -> None:
        lock = self.context.local_path(self.lock_path)
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"preexisting\n")
        with self.assertRaises(FileExistsError):
            handle_request(self.request("exclusive_deploy", 1, "3"), self.context)
        self.assertEqual(b"preexisting\n", lock.read_bytes())
        self.assertFalse(self.context.local_path(self.deployment_root).exists())

        lock.unlink()
        deployment = self.context.local_path(self.deployment_root)
        deployment.mkdir(parents=True)
        marker = deployment / "keep"
        marker.write_bytes(b"keep\n")
        with self.assertRaises(FileExistsError):
            handle_request(self.request("exclusive_deploy", 1, "3"), self.context)
        self.assertEqual(b"keep\n", marker.read_bytes())
        self.assertFalse(lock.exists())

    def test_mutation_stale_owner_and_forged_board_fail_closed(self) -> None:
        handle_request(self.request("exclusive_deploy", 1, "3"), self.context)
        artifact = self.artifact_declarations(include_payload=False)[0]
        target = self.context.local_path(artifact["target_path"])
        target.write_bytes(b"mutated\n")
        with self.assertRaises(ProtocolError):
            handle_request(self.request("fresh_readback", 2, "5"), self.context)

        forged = self.request("fresh_readback", 2, "5")
        forged["expected_board_identity"] = "sha256:" + "f" * 64
        with self.assertRaises(ProtocolError):
            handle_request(forged, self.context)

        stale = self.request("restore", 3, "7")
        stale["owner_id"] = "8" * 64
        with self.assertRaises(ProtocolError):
            handle_request(stale, self.context)
        self.assertTrue(self.context.local_path(self.deployment_root).exists())

    def test_restore_precedes_separate_helper_finalization(self) -> None:
        handle_request(self.request("exclusive_deploy", 1, "3"), self.context)
        helper_local = self.context.local_path(self.helper_remote)
        restored = handle_request(self.request("restore", 3, "7"), self.context)
        self.assertEqual("e055-restore-result/v1", restored["schema"])
        self.assertTrue(restored["restored"])
        self.assertTrue(helper_local.exists())

        finalized = handle_request(
            self.request("finalize_helper", 4, "a"), self.context
        )
        self.assertEqual("e055-helper-finalization/v1", finalized["schema"])
        self.assertTrue(finalized["helper_removed"])
        self.assertFalse(helper_local.exists())

    def test_capture_executes_only_exact_argv_and_observes_affinity(self) -> None:
        self.deploy_fake_capture_programs(
            "time.sleep(0.2)\nprint(json.dumps({'schema': 'test-harness'}))"
        )
        response = handle_request(self.capture_request(), self.context)
        self.assertEqual("e055-capture-result/v1", response["schema"])
        self.assertEqual(0, response["exit"]["code"])
        self.assertIsNone(response["exit"]["signal"])
        self.assertEqual([0], response["affinity"]["effective_cpus"])
        self.assertEqual(0, response["affinity"]["migration_count"])
        child_stdout = base64.b64decode(response["streams"]["child_stdout_base64"])
        self.assertIn(b'"schema": "test-harness"', child_stdout)

    def test_capture_rejects_argv_injection_before_run_directory_creation(self) -> None:
        self.deploy_fake_capture_programs("time.sleep(0.2)")
        request = self.capture_request(run_id="cpu0-pair2-hot")
        request["argv"][1] = ";touch /tmp/e055-injection"
        with self.assertRaises(ProtocolError):
            handle_request(request, self.context)
        run_path = self.context.local_path(
            f"{self.deployment_root}/captures/cpu0-pair2-hot"
        )
        self.assertFalse(run_path.exists())

    def test_capture_nonzero_is_reported_and_outputs_are_preserved(self) -> None:
        self.deploy_fake_capture_programs(
            "print(json.dumps({'schema': 'test-harness'}))\nraise SystemExit(7)"
        )
        response = handle_request(self.capture_request(), self.context)
        self.assertEqual(7, response["exit"]["code"])
        self.assertIsNone(response["exit"]["signal"])
        self.assertTrue(response["streams"]["e049c_json_base64"])

    def test_canonical_stdio_is_one_bounded_json_document(self) -> None:
        request = self.request("exclusive_deploy", 1, "3")
        encoded = (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
        decoded = json.loads(encoded)
        receipt = handle_request(decoded, self.context)
        self.assertEqual("3" * 64, receipt["request_id"])


if __name__ == "__main__":
    unittest.main()
