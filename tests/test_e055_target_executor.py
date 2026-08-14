"""Safety and exact-scope tests for the disabled E055 target executor."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import tooling.e055_target_executor as target_executor
from tooling.e055_target_executor import (
    E055TargetExecutor,
    TargetCapture,
    TargetRunFailure,
    limited_o3_microgate_plan,
)
from tooling.e055_raw_bundle import load_sealed_bundle
from tooling.e055_transport_evidence import (
    EndpointIdentity,
    ExpectedTransportPins,
    ExclusiveDeploymentReceipt,
    FreshReadbackProof,
    RemoteRuntimeObservation,
    TargetArtifactObservation,
    TransportImplementationEvidence,
)


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tooling/e055_remote_helper.py"
HELPER_PAYLOAD = HELPER_PATH.read_bytes()
HELPER_SHA256 = hashlib.sha256(HELPER_PAYLOAD).hexdigest()
HELPER_BLOB = subprocess.run(
    ["git", "rev-parse", "HEAD:tooling/e055_remote_helper.py"], cwd=ROOT,
    check=True, text=True, stdout=subprocess.PIPE,
).stdout.strip()
OPENSSH_PATH = Path("/usr/bin/ssh")
OPENSSH_PAYLOAD = OPENSSH_PATH.read_bytes()
OPENSSH_SHA256 = hashlib.sha256(OPENSSH_PAYLOAD).hexdigest()
KNOWN_HOSTS_SHA256 = "1" * 64
TARGET_ENDPOINT_SHA256 = "2" * 64
FAKE_BOARD = "sha256:" + "3" * 64
FAKE_FINGERPRINT = "SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
CLIENT_CONFIG = (
    "BatchMode=yes", "StrictHostKeyChecking=yes",
    "UserKnownHostsFile=external-pinned-file", "GlobalKnownHostsFile=/dev/null",
    "CheckHostIP=no", "PasswordAuthentication=no",
    "KbdInteractiveAuthentication=no", "NumberOfPasswordPrompts=0",
    "ForwardAgent=no", "ClearAllForwardings=yes", "PermitLocalCommand=no",
    "RequestTTY=no", "ConnectTimeout=10", "ConnectionAttempts=1",
    "ServerAliveInterval=5", "ServerAliveCountMax=2", "LogLevel=ERROR",
    "IdentitiesOnly=yes", "ProxyCommand=none", "ProxyJump=none",
    "CanonicalizeHostname=no",
)


def fixture_expected_pins() -> ExpectedTransportPins:
    return ExpectedTransportPins(
        "e055-expected-transport-pins/v1", "pinned_host_key", "fake-transport",
        FAKE_BOARD, FAKE_FINGERPRINT, KNOWN_HOSTS_SHA256,
        TARGET_ENDPOINT_SHA256, HELPER_SHA256, len(HELPER_PAYLOAD), HELPER_BLOB,
        OPENSSH_SHA256, len(OPENSSH_PAYLOAD),
    )


class FakeTransport:
    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.prepared = []
        self.deployed = {}
        self.captures = []
        self.restored = 0
        self.actions = []
        self.requests = []
        self.helper_sha256 = HELPER_SHA256

    def implementation_evidence(self) -> TransportImplementationEvidence:
        return TransportImplementationEvidence(
            "e055-openssh-implementation/v1", "openssh_fixed_helper",
            "tooling/e055_remote_helper.py", self.helper_sha256,
            len(HELPER_PAYLOAD), HELPER_BLOB, "/usr/bin/ssh", OPENSSH_SHA256,
            len(OPENSSH_PAYLOAD), 0o755, "OpenSSH_9.9p2", "/dev/null",
            CLIENT_CONFIG,
            hashlib.sha256(("\n".join(CLIENT_CONFIG) + "\n").encode("ascii")).hexdigest(),
            KNOWN_HOSTS_SHA256, TARGET_ENDPOINT_SHA256,
            "external_agent_or_identity",
        )

    def runtime_observation(
        self, sequence: int, request_id: str, request_nonce: str,
    ) -> RemoteRuntimeObservation:
        return RemoteRuntimeObservation(
            "e055-remote-runtime-observation/v1", sequence,
            request_id, request_nonce,
            "/tmp/e055-q1-hot-cold/helpers/" + self.helper_sha256 + "/helper.py",
            self.helper_sha256, len(HELPER_PAYLOAD), 0o700, 7, 9001,
            "e055-remote-helper/v1", "/usr/bin/python3",
            "/usr/bin/python3.13", "3.13.7", "7" * 64,
            6839896, 0o755, 7, 42,
        )

    def _observations(
        self, artifacts: tuple, *, operation_sequence: int,
        request_id: str, request_nonce: str,
    ) -> tuple[TargetArtifactObservation, ...]:
        observations = []
        for index, artifact in enumerate(artifacts, 1):
            payload, mode = self.deployed[artifact.target_path]
            observations.append(TargetArtifactObservation(
                role=artifact.role,
                target_path=artifact.target_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                mode=mode,
                device=7,
                inode=1000 + index,
                operation_sequence=operation_sequence,
                request_id=request_id,
                request_nonce=request_nonce,
            ))
        return tuple(observations)

    def prepare(
        self, artifacts: tuple, *, deployment_root: str, lock_path: str,
        request_id: str, request_nonce: str,
    ) -> ExclusiveDeploymentReceipt:
        self.actions.append("prepare")
        self.requests.append(("exclusive_deploy", request_id, request_nonce))
        self.prepared.extend(artifacts)
        for artifact in artifacts:
            if artifact.target_path in self.deployed:
                raise FileExistsError(artifact.target_path)
            self.assert_artifact(artifact)
            self.deployed[artifact.target_path] = (bytes(artifact.payload), artifact.mode)
        return ExclusiveDeploymentReceipt(
            schema="e055-exclusive-deployment-receipt/v1",
            operation_sequence=1,
            request_id=request_id,
            request_nonce=request_nonce,
            deployment_root=deployment_root,
            lock_path=lock_path,
            lock_acquired_exclusively=True,
            deployment_created_exclusively=True,
            endpoint_identity=EndpointIdentity(
                "pinned_host_key", "fake-transport", FAKE_BOARD, FAKE_FINGERPRINT
            ),
            artifacts=self._observations(
                artifacts, operation_sequence=1,
                request_id=request_id, request_nonce=request_nonce,
            ),
            runtime=self.runtime_observation(1, request_id, request_nonce),
        )

    def assert_artifact(self, artifact) -> None:
        if hashlib.sha256(artifact.payload).hexdigest() != artifact.sha256:
            raise ValueError("test transport received inconsistent bytes")

    def readback(
        self, artifacts: tuple, *, deployment_root: str,
        request_id: str, request_nonce: str,
    ) -> FreshReadbackProof:
        self.actions.append("readback")
        self.requests.append(("fresh_readback", request_id, request_nonce))
        return FreshReadbackProof(
            schema="e055-fresh-readback-proof/v1",
            operation_sequence=2,
            request_id=request_id,
            request_nonce=request_nonce,
            deployment_root=deployment_root,
            endpoint_identity=EndpointIdentity(
                "pinned_host_key", "fake-transport", FAKE_BOARD, FAKE_FINGERPRINT
            ),
            artifacts=self._observations(
                artifacts, operation_sequence=2,
                request_id=request_id, request_nonce=request_nonce,
            ),
            runtime=self.runtime_observation(2, request_id, request_nonce),
        )

    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.actions.append("capture")
        self.captures.append((run, e049c_argv, dict(environment)))
        ordinal = len(self.captures)
        failed = self.fail_at == ordinal
        calls = run.iterations
        elapsed_ns = calls * 1_000_000
        if run.cache_state == "cold_conditioned":
            conditioning = {
                "strategy": "verified_write_read_each_64B_line",
                "requested_bytes": 67_108_864,
                "actual_bytes": 67_108_864,
                "line_bytes": 64,
                "lines_touched": 1_048_576,
                "checksum": "0x8d3ea13d15850279",
                "verified_touched": True,
                "warmup_calls": 0,
            }
        else:
            conditioning = {
                "strategy": "verified_kernel_warmup",
                "requested_bytes": run.actual_working_set_bytes,
                "actual_bytes": run.actual_working_set_bytes,
                "line_bytes": 64,
                "lines_touched": (run.actual_working_set_bytes + 63) // 64,
                "checksum": "0xe055c002",
                "verified_touched": True,
                "warmup_calls": 16,
            }
        harness = {
            "schema": "e055-q1-hot-cold-harness/v1",
            "mode": run.mode,
            "cache_state": run.cache_state,
            "cpu": run.cpu,
            "q1_layout": "E039 stock native block_q1_0x4 4x4 DOTPROD",
            "golden_pass": True,
            "golden_cases": 18,
            "target_working_set_bytes": run.target_working_set_bytes,
            "actual_working_set_bytes": run.actual_working_set_bytes,
            "blocks": run.blocks,
            "iterations": calls,
            "calls": calls,
            "elapsed_ns": elapsed_ns,
            "first_call_ns": 1_000_000,
            "calls_per_second": 1000.0,
            "logical_bytes_per_call": {
                "q1_packed_bytes": run.blocks * 72,
                "q8_bytes": run.blocks * 4 * 34,
                "total_input_bytes": run.blocks * 208,
                "output_bytes": 16,
                "dot_products": run.blocks * 512,
            },
            "checksum": "0xe055d001",
            "cold_conditioning": conditioning,
            "sync": {
                "requested": True, "started": True, "acknowledged": True,
                "ended": True, "sequence": "S/A/E",
            },
            "qualification": "unqualified_harness_output_requires_E049c_join",
        }
        harness_argv = list(e049c_argv[e049c_argv.index("--") + 1:])
        configs = (("cpu_cycles", "0x11"), ("instructions", "0x8"),
                   ("stall_backend", "0x24"))
        e049c = {
            "schema_version": "e049c-arm-pmu/v2",
            "status": "ok",
            "sample_valid": True,
            "event_source": "armv8_pmuv3_raw_config",
            "counter_semantics": "event counts only; no DDR-byte conversion",
            "event_group": "core",
            "software_group_size_limit": 4,
            "event_group_size": 3,
            "software_group_size_limit_semantics": (
                "conservative launcher policy; not measured hardware PMU capacity"
            ),
            "min_running_ratio": 0.95,
            "pid": 1000 + ordinal,
            "process_group": 1000 + ordinal,
            "command": harness_argv,
            "sync": {
                "mode": "start_ack_end", "started": True,
                "acknowledged": True, "ended": True,
            },
            "measured_elapsed_ns": elapsed_ns + 1000,
            "thermal": {
                "guard_enabled": True, "checked": True, "readable": True,
                "limit_c": 85.0, "max_observed_c": 55.0, "tripped": False,
            },
            "exit": {"code": 0, "raw_wait_status": 0},
            "failure_reason": None,
            "events": [{
                "name": name, "config": config,
                "meaning": f"ARMv8 PMUv3 {name} event count",
                "support": "supported", "count_semantics": "event_count_not_bytes",
                "sample_valid": True, "value": 100 + index,
                "time_enabled_ns": elapsed_ns, "time_running_ns": elapsed_ns,
                "running_ratio": 1.0, "errno": 0, "error": None,
            } for index, (name, config) in enumerate(configs)],
        }
        return TargetCapture(
            child_stdout=(json.dumps(harness, sort_keys=True,
                                     separators=(",", ":")) + "\n").encode("ascii"),
            child_stderr=b"exec failed\n" if failed else b"",
            e049c_json=(json.dumps(e049c, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("ascii"),
            wrapper_stdout=b"",
            wrapper_stderr=b"",
            exit_code=127 if failed else 0,
            signal=None,
            effective_cpus=(run.cpu,),
            cpu_start=run.cpu,
            cpu_end=run.cpu,
            migration_count=0,
        )

    def restore(self) -> None:
        self.actions.append("restore")
        self.restored += 1


class RaisingTransport(FakeTransport):
    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.captures.append((run, e049c_argv, dict(environment)))
        raise OSError("injected transport failure")


class RaisingAndRestoreFailingTransport(RaisingTransport):
    def restore(self) -> None:
        self.actions.append("restore")
        self.restored += 1
        raise RuntimeError("injected restore failure")


class MalformedTransport(FakeTransport):
    def capture(self, *, run, e049c_argv, environment) -> TargetCapture:
        self.captures.append((run, e049c_argv, dict(environment)))
        return TargetCapture(
            child_stdout="not bytes",  # type: ignore[arg-type]
            child_stderr=b"", e049c_json=b"{}\n", wrapper_stdout=b"",
            wrapper_stderr=b"", exit_code=0, signal=None,
            effective_cpus=(run.cpu,), cpu_start=run.cpu, cpu_end=run.cpu,
            migration_count=0,
        )


class NoProofTransport(FakeTransport):
    def prepare(
        self, artifacts: tuple, *, deployment_root: str, lock_path: str,
        request_id: str, request_nonce: str,
    ):
        self.prepared.extend(artifacts)
        return None

    def readback(
        self, artifacts: tuple, *, deployment_root: str,
        request_id: str, request_nonce: str,
    ):
        return None


class MutatingProofTransport(FakeTransport):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.mutation = mutation
        self.receipt = None

    def prepare(
        self, artifacts: tuple, *, deployment_root: str, lock_path: str,
        request_id: str, request_nonce: str,
    ):
        receipt = super().prepare(
            artifacts, deployment_root=deployment_root, lock_path=lock_path,
            request_id=request_id, request_nonce=request_nonce,
        )
        if self.mutation == "fabricated_mapping":
            return {"schema": receipt.schema}
        if self.mutation == "exclusive_false":
            receipt = replace(receipt, lock_acquired_exclusively=False)
        elif self.mutation == "forged_receipt_hash":
            forged = replace(receipt.artifacts[0], sha256="c" * 64)
            receipt = replace(receipt, artifacts=(forged, receipt.artifacts[1]))
        elif self.mutation == "forged_mode_type":
            forged = replace(receipt.artifacts[0], mode=True)
            receipt = replace(receipt, artifacts=(forged, receipt.artifacts[1]))
        elif self.mutation == "receipt_request_id_mismatch":
            receipt = replace(receipt, request_id="f" * 64)
        elif self.mutation == "receipt_nonce_mismatch":
            receipt = replace(receipt, request_nonce="e" * 64)
        self.receipt = receipt
        return receipt

    def readback(
        self, artifacts: tuple, *, deployment_root: str,
        request_id: str, request_nonce: str,
    ):
        proof = super().readback(
            artifacts, deployment_root=deployment_root,
            request_id=request_id, request_nonce=request_nonce,
        )
        if self.mutation == "readback_hash_mismatch":
            forged = replace(proof.artifacts[0], sha256="d" * 64)
            proof = replace(proof, artifacts=(forged, proof.artifacts[1]))
        elif self.mutation == "identity_mismatch":
            proof = replace(
                proof,
                endpoint_identity=EndpointIdentity(
                    "test_fixture", "different-endpoint", "fake-a733", None
                ),
            )
        elif self.mutation == "reused_observations" and self.receipt is not None:
            proof = replace(proof, artifacts=self.receipt.artifacts)
        elif self.mutation == "readback_request_id_mismatch":
            proof = replace(proof, request_id="d" * 64)
        elif self.mutation == "replayed_receipt_nonce" and self.receipt is not None:
            proof = replace(proof, request_nonce=self.receipt.request_nonce)
        return proof

class E055TargetExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e055-executor-")
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "E055 Executor Test"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "e055@example.invalid"],
                       cwd=self.root, check=True)
        helper = self.root / "tooling/e055_remote_helper.py"
        helper.parent.mkdir()
        shutil.copyfile(HELPER_PATH, helper)
        experiment = self.root / "experiments/E055-q1-hot-cold"
        experiment.mkdir(parents=True)
        subprocess.run(["git", "add", "tooling/e055_remote_helper.py"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "pin helper fixture"],
                       cwd=self.root, check=True)
        self.expected_pins = fixture_expected_pins()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def executor(self, transport) -> E055TargetExecutor:
        return E055TargetExecutor(
            self.root, transport=transport,
            expected_transport_pins=self.expected_pins,
        )

    def test_plan_is_exact_bounded_twenty_run_o3_microgate(self) -> None:
        plan = limited_o3_microgate_plan()
        self.assertEqual(len(plan), 20)
        self.assertEqual({item.cpu for item in plan}, {0, 6})
        self.assertEqual({item.build_name for item in plan}, {"O3"})
        self.assertEqual({item.mode for item in plan}, {"full_dotprod"})
        self.assertEqual({item.pmu_group for item in plan}, {"core"})
        self.assertEqual({item.target_working_set_bytes for item in plan}, {65536})
        for cpu in (0, 6):
            scoped = [item for item in plan if item.cpu == cpu]
            self.assertEqual(len(scoped), 10)
            for pair_index in range(1, 6):
                pair = [item for item in scoped if item.pair_index == pair_index]
                expected_states = (
                    ["hot_repeat", "cold_conditioned"] if pair_index % 2
                    else ["cold_conditioned", "hot_repeat"]
                )
                self.assertEqual([item.cache_state for item in pair], expected_states)
                self.assertEqual([item.order_index for item in pair], [1, 2])

    def test_no_injected_transport_means_no_files_or_target_side_effect(self) -> None:
        executor = E055TargetExecutor(self.root)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            executor.execute("phase-a")
        self.assertFalse((self.root / "experiments/E055-q1-hot-cold/raw").exists())

    def test_deployment_layout_is_phase_unique_and_content_addressed(self) -> None:
        layout_function = getattr(target_executor, "canonical_deployment_layout", None)
        self.assertTrue(callable(layout_function))
        first = layout_function("phase-a", "a" * 64, "b" * 64)
        repeated = layout_function("phase-a", "a" * 64, "b" * 64)
        different_phase = layout_function("phase-b", "a" * 64, "b" * 64)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.deployment_root, different_phase.deployment_root)
        self.assertIn("a" * 64, first.harness_path)
        self.assertIn("b" * 64, first.pmu_path)
        self.assertTrue(first.harness_path.startswith(first.deployment_root + "/"))
        self.assertTrue(first.pmu_path.startswith(first.deployment_root + "/"))
        self.assertNotEqual(first.lock_path, first.deployment_root)

    def test_missing_deploy_and_readback_proof_refuses_before_capture(self) -> None:
        transport = NoProofTransport()
        with self.assertRaisesRegex(ValueError, "proof|receipt|prepare"):
            self.executor(transport).execute("phase-no-proof")
        self.assertEqual(transport.captures, [])
        self.assertEqual(transport.restored, 1)

    def test_forged_or_nonindependent_transport_proof_refuses_before_capture(self) -> None:
        mutations = (
            "fabricated_mapping", "exclusive_false", "forged_receipt_hash",
            "forged_mode_type", "readback_hash_mismatch", "identity_mismatch",
            "reused_observations", "receipt_request_id_mismatch",
            "receipt_nonce_mismatch", "readback_request_id_mismatch",
            "replayed_receipt_nonce",
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                transport = MutatingProofTransport(mutation)
                with self.assertRaises(ValueError):
                    self.executor(transport).execute(
                        f"phase-forged-{index}"
                    )
                self.assertEqual(transport.captures, [])
                self.assertEqual(transport.restored, 1)

    def test_fake_transport_populates_exact_streams_runner_and_manifest(self) -> None:
        transport = FakeTransport()
        result = self.executor(transport).execute("phase-a")
        self.assertEqual(result.run_count, 20)
        self.assertEqual(len(transport.prepared), 2)
        for artifact in transport.prepared:
            self.assertTrue(artifact.target_path.startswith(
                "/tmp/e055-q1-hot-cold/deployments/"
            ))
            self.assertIn(artifact.sha256, artifact.target_path)
        self.assertEqual(len(transport.captures), 20)
        self.assertEqual(transport.restored, 1)
        self.assertEqual(transport.actions[:3], ["prepare", "readback", "capture"])
        self.assertEqual(transport.actions[-1], "restore")
        self.assertEqual(
            [operation for operation, _, _ in transport.requests],
            ["exclusive_deploy", "fresh_readback"],
        )
        request_tokens = [
            token for _, request_id, request_nonce in transport.requests
            for token in (request_id, request_nonce)
        ]
        self.assertEqual(len(set(request_tokens)), 4)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", token)
                            for token in request_tokens))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["runs"]), 20)
        self.assertTrue(manifest["target_workload_executed"])
        receipt_artifacts = manifest["transport_evidence"]["exclusive_receipt"][
            "artifacts"
        ]
        readback_artifacts = manifest["transport_evidence"]["fresh_readback"][
            "artifacts"
        ]
        self.assertEqual(
            [{key: value for key, value in item.items()
              if key not in {"operation_sequence", "request_id", "request_nonce"}}
             for item in receipt_artifacts],
            [{key: value for key, value in item.items()
              if key not in {"operation_sequence", "request_id", "request_nonce"}}
             for item in readback_artifacts],
        )
        self.assertEqual({item["operation_sequence"] for item in receipt_artifacts}, {1})
        self.assertEqual({item["operation_sequence"] for item in readback_artifacts}, {2})
        self.assertNotEqual(receipt_artifacts, readback_artifacts)
        self.assertEqual(
            manifest["transport_evidence"]["endpoint_identity"],
            manifest["transport_evidence"]["exclusive_receipt"]["endpoint_identity"],
        )
        self.assertEqual(
            manifest["transport_evidence"]["endpoint_identity"],
            manifest["transport_evidence"]["fresh_readback"]["endpoint_identity"],
        )
        first = manifest["runs"][0]
        run_dir = result.phase_dir / "runs" / first["run_id"]
        runner = json.loads((run_dir / "runner.json").read_text(encoding="utf-8"))
        launcher = runner["e049c_argv"]
        remote_output = launcher[launcher.index("-o") + 1]
        self.assertTrue(remote_output.startswith(
            manifest["transport_evidence"]["deployment_root"] + "/captures/"
        ))
        self.assertEqual(launcher[launcher.index("--child-stdout") + 1],
                         remote_output.replace(
                             "e049c.json", "target-harness.stdout.raw"
                         ))
        self.assertEqual(launcher[launcher.index("--child-stderr") + 1],
                         remote_output.replace(
                             "e049c.json", "target-harness.stderr.raw"
                         ))
        self.assertEqual(
            runner["transport_evidence"], manifest["transport_evidence"]
        )
        self.assertEqual(
            manifest["transport_evidence"]["requests"]["exclusive_deploy"],
            {
                "operation_sequence": 1,
                "request_id": transport.requests[0][1],
                "request_nonce": transport.requests[0][2],
            },
        )
        self.assertEqual(
            manifest["transport_evidence"]["requests"]["fresh_readback"],
            {
                "operation_sequence": 2,
                "request_id": transport.requests[1][1],
                "request_nonce": transport.requests[1][2],
            },
        )
        stdout_envelope = json.loads(
            (run_dir / "harness.stdout.capture.json").read_text(encoding="utf-8")
        )
        decoded_stdout = json.loads(
            base64.b64decode(stdout_envelope["payload_base64"]).decode("ascii")
        )
        self.assertEqual(decoded_stdout["mode"], "full_dotprod")
        executable = result.phase_dir / "artifacts/harness-O3.bin"
        published = ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64"
        self.assertEqual(hashlib.sha256(executable.read_bytes()).hexdigest(),
                         hashlib.sha256(published.read_bytes()).hexdigest())
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "sealed fake phase"],
                       cwd=self.root, check=True)
        sealed = load_sealed_bundle(
            result.manifest_path, expected_transport_pins=self.expected_pins,
        )
        self.assertEqual(len(sealed.samples), 20)

    def test_failed_target_run_preserves_captured_bytes_and_stops(self) -> None:
        transport = FakeTransport(fail_at=3)
        executor = self.executor(transport)
        with self.assertRaisesRegex(TargetRunFailure, "cpu0-pair2-cold"):
            executor.execute("phase-fail")
        self.assertEqual(len(transport.captures), 3)
        self.assertEqual(transport.restored, 1)
        phase = self.root / "experiments/E055-q1-hot-cold/raw/phase-fail"
        self.assertEqual((phase / "bundle.json").stat().st_size, 0)
        failed = phase / "runs/cpu0-pair2-cold"
        stderr_envelope = json.loads(
            (failed / "harness.stderr.capture.json").read_text(encoding="utf-8")
        )
        self.assertEqual(base64.b64decode(stderr_envelope["payload_base64"]),
                         b"exec failed\n")
        failure = json.loads((failed / "runner.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["schema"], "e055-runner-failure/v1")
        self.assertEqual(failure["exit"], {"code": 127, "signal": None})

    def test_transport_exception_is_recorded_without_leaking_exception_text(self) -> None:
        transport = RaisingTransport()
        with self.assertRaisesRegex(TargetRunFailure, "partial phase preserved"):
            self.executor(transport).execute("phase-error")
        self.assertEqual(transport.restored, 1)
        runner = self.root / (
            "experiments/E055-q1-hot-cold/raw/phase-error/"
            "runs/cpu0-pair1-hot/runner.json"
        )
        document = json.loads(runner.read_text(encoding="utf-8"))
        self.assertEqual(document["failure_kind"], "OSError")
        self.assertNotIn("injected transport failure", runner.read_text(encoding="utf-8"))

    def test_restore_failure_preserves_primary_target_failure(self) -> None:
        transport = RaisingAndRestoreFailingTransport()
        with self.assertRaises(BaseExceptionGroup) as caught:
            self.executor(transport).execute("phase-double-fail")
        failures = caught.exception.exceptions
        self.assertEqual(len(failures), 2)
        self.assertIsInstance(failures[0], TargetRunFailure)
        self.assertIsInstance(failures[0].__cause__, OSError)
        self.assertIsInstance(failures[1], RuntimeError)
        self.assertEqual(str(failures[1]), "injected restore failure")
        self.assertEqual(transport.restored, 1)
        runner = self.root / (
            "experiments/E055-q1-hot-cold/raw/phase-double-fail/"
            "runs/cpu0-pair1-hot/runner.json"
        )
        self.assertEqual(
            json.loads(runner.read_text(encoding="utf-8"))["failure_kind"],
            "OSError",
        )

    def test_existing_phase_refuses_before_transport_prepare(self) -> None:
        raw = self.root / "experiments/E055-q1-hot-cold/raw"
        raw.mkdir()
        (raw / "phase-existing").mkdir()
        transport = FakeTransport()
        with self.assertRaises(FileExistsError):
            self.executor(transport).execute("phase-existing")
        self.assertEqual(transport.prepared, [])
        self.assertEqual(transport.captures, [])
        self.assertEqual(transport.restored, 0)

    def test_malformed_transport_value_fails_closed_and_records_failure(self) -> None:
        transport = MalformedTransport()
        with self.assertRaises(TargetRunFailure):
            self.executor(transport).execute("phase-malformed")
        runner = self.root / (
            "experiments/E055-q1-hot-cold/raw/phase-malformed/"
            "runs/cpu0-pair1-hot/runner.json"
        )
        document = json.loads(runner.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "e055-runner-failure/v1")
        self.assertEqual(transport.restored, 1)


if __name__ == "__main__":
    unittest.main()
