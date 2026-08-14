"""Disabled-by-default executor for the first bounded E055 target microgate.

The module has no SSH or board implementation. Target I/O exists only through
an explicitly injected transport, which keeps accidental execution impossible
before independent review. Captures are reservations-first and are not
measurement evidence until committed and reloaded by ``load_sealed_bundle``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any, Mapping, Protocol

from tooling.e055_capture_scaffold import (
    ReservedOutput,
    RunPlan,
    populate_reserved,
    reserve_phase,
    safe_environment,
)
from tooling.e055_q1_hotcold import (
    PMU_ARTIFACT,
    ROOT as PUBLICATION_ROOT,
    load_publication_contract,
    runtime_qualification_sha256,
)
from tooling.e055_raw_bundle import (
    canonical_e049c_launcher_argv,
    canonical_harness_argv,
)
from tooling.e055_transport_evidence import (
    ExclusiveDeploymentReceipt,
    ExpectedTransportPins,
    FreshReadbackProof,
    TransportImplementationEvidence,
    canonical_deployment_layout,
    canonical_remote_output_path,
    validate_transport_evidence,
)


PHASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
RAW_RELATIVE = Path("experiments/E055-q1-hot-cold/raw")
O3_ARTIFACT = PUBLICATION_ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64"


@dataclass(frozen=True)
class LimitedRun:
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
    pmu_group: str
    iterations: int

    def cell(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cache_state": self.cache_state,
            "cpu": self.cpu,
            "target_working_set_bytes": self.target_working_set_bytes,
            "actual_working_set_bytes": self.actual_working_set_bytes,
            "blocks": self.blocks,
            "pmu_group": self.pmu_group,
        }


@dataclass(frozen=True)
class TargetArtifact:
    role: str
    target_path: str
    payload: bytes
    sha256: str
    mode: int


@dataclass(frozen=True)
class TargetCapture:
    """Exact bytes and affinity observations returned by an injected transport."""

    child_stdout: bytes
    child_stderr: bytes
    e049c_json: bytes
    wrapper_stdout: bytes
    wrapper_stderr: bytes
    exit_code: int
    signal: int | None
    effective_cpus: tuple[int, ...]
    cpu_start: int
    cpu_end: int
    migration_count: int


class TargetTransport(Protocol):
    def implementation_evidence(self) -> TransportImplementationEvidence: ...

    def prepare(
        self, artifacts: tuple[TargetArtifact, ...], *,
        deployment_root: str, lock_path: str,
        request_id: str, request_nonce: str,
    ) -> ExclusiveDeploymentReceipt: ...

    def readback(
        self, artifacts: tuple[TargetArtifact, ...], *, deployment_root: str,
        request_id: str, request_nonce: str,
    ) -> FreshReadbackProof: ...

    def capture(
        self, *, run: LimitedRun, e049c_argv: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> TargetCapture: ...

    def restore(self) -> None: ...


@dataclass(frozen=True)
class PhaseExecutionResult:
    phase_dir: Path
    manifest_path: Path
    run_count: int


class TargetRunFailure(RuntimeError):
    """A target attempt failed after its available raw bytes were preserved."""


def limited_o3_microgate_plan() -> tuple[LimitedRun, ...]:
    """Return the immutable 20-run CPU0/CPU6, 64 KiB, O3/core plan."""

    target = 64 * 1024
    blocks = (target + 207) // 208
    actual = blocks * 208
    runs: list[LimitedRun] = []
    for cpu in (0, 6):
        for pair_index in range(1, 6):
            pair_order = "hot_then_cold" if pair_index % 2 else "cold_then_hot"
            states = (
                ("hot_repeat", "cold_conditioned") if pair_index % 2
                else ("cold_conditioned", "hot_repeat")
            )
            for order_index, cache_state in enumerate(states, 1):
                state_label = "hot" if cache_state == "hot_repeat" else "cold"
                runs.append(LimitedRun(
                    run_id=f"cpu{cpu}-pair{pair_index}-{state_label}",
                    pair_id=f"cpu{cpu}-pair{pair_index}",
                    pair_index=pair_index,
                    pair_order=pair_order,
                    order_index=order_index,
                    build_name="O3",
                    mode="full_dotprod",
                    cache_state=cache_state,
                    cpu=cpu,
                    target_working_set_bytes=target,
                    actual_working_set_bytes=actual,
                    blocks=blocks,
                    pmu_group="core",
                    iterations=250 if cache_state == "hot_repeat" else 1,
                ))
    if len(runs) != 20:
        raise AssertionError("limited E055 plan must contain exactly 20 runs")
    return tuple(runs)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _fresh_transport_requests() -> tuple[tuple[str, str], tuple[str, str]]:
    """Create caller-side 256-bit IDs/nonces for two distinct observations."""

    tokens: list[str] = []
    while len(tokens) < 4:
        candidate = secrets.token_hex(32)
        if candidate not in tokens:
            tokens.append(candidate)
    return (tokens[0], tokens[1]), (tokens[2], tokens[3])


def _stream_envelope(
    run_id: str, producer: str, stream: str, payload: bytes,
) -> bytes:
    return _json_bytes({
        "schema": "e055-stream-capture/v1",
        "run_id": run_id,
        "producer": producer,
        "stream": stream,
        "encoding": "base64",
        "payload_size_bytes": len(payload),
        "payload_sha256": _sha256(payload),
        "payload_base64": base64.b64encode(payload).decode("ascii"),
    })


def _git_object_format(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-object-format"], cwd=root, check=False,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or value not in ("sha1", "sha256"):
        raise ValueError("repository has no supported Git object format")
    return value


def _artifact_declaration(
    root: Path, reservation: ReservedOutput, role: str, object_format: str,
) -> dict[str, Any]:
    status = reservation.path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or \
            (status.st_dev, status.st_ino) != (reservation.device, reservation.inode):
        raise OSError("reserved raw artifact identity changed")
    payload = reservation.path.read_bytes()
    if not payload:
        raise ValueError(f"raw role was not populated: {role}")
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return {
        "role": role,
        "path": reservation.path.relative_to(root).as_posix(),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "git_blob_oid": hashlib.new(object_format, framed).hexdigest(),
    }


def _strict_capture(capture: Any, run: LimitedRun) -> str | None:
    if not isinstance(capture, TargetCapture):
        return "transport returned a noncanonical capture type"
    byte_fields = (
        capture.child_stdout, capture.child_stderr, capture.e049c_json,
        capture.wrapper_stdout, capture.wrapper_stderr,
    )
    if any(not isinstance(value, bytes) for value in byte_fields):
        return "transport returned a non-bytes stream"
    if not isinstance(capture.effective_cpus, tuple):
        return "transport returned a noncanonical CPU tuple"
    if not isinstance(capture.exit_code, int) or isinstance(capture.exit_code, bool) or \
            capture.exit_code < 0:
        return "wrapper exit code is invalid"
    if capture.signal is not None and (
        not isinstance(capture.signal, int) or isinstance(capture.signal, bool)
        or capture.signal <= 0
    ):
        return "wrapper signal is invalid"
    integers = (
        *capture.effective_cpus, capture.cpu_start, capture.cpu_end,
        capture.migration_count,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
        return "affinity observation is not strictly integer"
    if capture.exit_code != 0 or capture.signal is not None:
        return "wrapper did not exit successfully"
    if capture.wrapper_stdout or capture.wrapper_stderr or capture.child_stderr:
        return "a qualified success requires empty wrapper stdout/stderr and child stderr"
    if not capture.child_stdout or not capture.e049c_json:
        return "qualified child stdout and E049c JSON must be nonempty"
    if capture.effective_cpus != (run.cpu,) or capture.cpu_start != run.cpu or \
            capture.cpu_end != run.cpu or capture.migration_count != 0:
        return "affinity/migration observation does not prove the requested CPU"
    for payload in (capture.child_stdout, capture.e049c_json):
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return "successful JSON capture is malformed"
        if not isinstance(document, dict):
            return "successful JSON capture must contain one object"
    return None


def _sanitized_failure_capture(capture: TargetCapture, run: LimitedRun) -> TargetCapture:
    """Keep every well-typed byte stream while making failure JSON serializable."""

    def exact_bytes(value: Any) -> bytes:
        return value if isinstance(value, bytes) else b""

    exit_code = capture.exit_code
    if not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code < 0:
        exit_code = 255
    signal_value = capture.signal
    if signal_value is not None and (
        not isinstance(signal_value, int) or isinstance(signal_value, bool)
        or signal_value <= 0
    ):
        signal_value = None
    effective = capture.effective_cpus
    if not isinstance(effective, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in effective
    ):
        effective = ()
    strict_values = (capture.cpu_start, capture.cpu_end, capture.migration_count)
    cpu_start, cpu_end, migration = strict_values
    if any(not isinstance(value, int) or isinstance(value, bool) for value in strict_values):
        cpu_start, cpu_end, migration = run.cpu, run.cpu, -1
    return TargetCapture(
        exact_bytes(capture.child_stdout), exact_bytes(capture.child_stderr),
        exact_bytes(capture.e049c_json), exact_bytes(capture.wrapper_stdout),
        exact_bytes(capture.wrapper_stderr), exit_code, signal_value, effective,
        cpu_start, cpu_end, migration,
    )


def _ensure_raw_parent(root: Path) -> Path:
    experiment = root / "experiments/E055-q1-hot-cold"
    status = experiment.lstat()
    if not stat.S_ISDIR(status.st_mode) or experiment.resolve() != experiment.absolute():
        raise ValueError("E055 experiment root must be one non-symlink directory")
    raw = root / RAW_RELATIVE
    try:
        os.mkdir(raw, 0o700)
    except FileExistsError:
        raw_status = raw.lstat()
        if not stat.S_ISDIR(raw_status.st_mode) or raw.resolve() != raw.absolute():
            raise ValueError("E055 raw root must be one non-symlink directory")
    return raw


class E055TargetExecutor:
    """Package one exact microgate through a caller-injected target transport."""

    def __init__(
        self, repository_root: Path, transport: TargetTransport | None = None,
        expected_transport_pins: ExpectedTransportPins | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.transport = transport
        self.expected_transport_pins = expected_transport_pins

    def execute(self, phase_id: str) -> PhaseExecutionResult:
        if self.transport is None:
            raise RuntimeError("E055 target execution is disabled without an injected transport")
        if type(self.expected_transport_pins) is not ExpectedTransportPins:
            raise RuntimeError("E055 target execution requires independent expected transport pins")
        if not isinstance(phase_id, str) or PHASE_ID_RE.fullmatch(phase_id) is None:
            raise ValueError("phase_id must be canonical and bounded")
        plan = limited_o3_microgate_plan()
        raw_parent = _ensure_raw_parent(self.repository_root)
        phase_dir = raw_parent / phase_id
        reservations = reserve_phase(
            phase_dir, tuple(RunPlan(run.run_id, run.build_name) for run in plan)
        )
        by_path = {reservation.path: reservation for reservation in reservations}
        bundle_reservation = by_path[phase_dir / "bundle.json"]
        harness_reservation = by_path[phase_dir / "artifacts/harness-O3.bin"]
        contract = load_publication_contract()
        harness_payload = O3_ARTIFACT.read_bytes()
        pmu_payload = PMU_ARTIFACT.read_bytes()
        if _sha256(harness_payload) != contract["allowed_builds"]["O3"] or \
                _sha256(pmu_payload) != contract["pmu_binary_sha256"]:
            raise ValueError("target artifacts do not match publication qualification")
        populate_reserved(harness_reservation, harness_payload)
        harness_relative = harness_reservation.path.relative_to(
            self.repository_root
        ).as_posix()
        layout = canonical_deployment_layout(
            phase_id, _sha256(harness_payload), _sha256(pmu_payload)
        )
        prepared = (
            TargetArtifact(
                "harness_executable", layout.harness_path, harness_payload,
                _sha256(harness_payload), 0o755,
            ),
            TargetArtifact(
                "pmu_executable", layout.pmu_path, pmu_payload,
                _sha256(pmu_payload), 0o755,
            ),
        )
        qualification_sha256 = runtime_qualification_sha256(contract)
        qualification = {
            "source_sha256": contract["source_sha256"],
            "compiler_sha256": contract["compiler_sha256"],
            "compiler_id": contract["compiler_id"],
            "pmu_source_sha256": contract["pmu_source_sha256"],
            "pmu_binary_sha256": contract["pmu_binary_sha256"],
            "pmu_binary_size_bytes": contract["pmu_binary_size_bytes"],
            "pmu_compiler_sha256": contract["pmu_compiler_sha256"],
            "pmu_compiler_id": contract["pmu_compiler_id"],
            "upstream_commit": contract["upstream_commit"],
            "upstream_ref": contract["upstream_ref"],
            "upstream_repack_sha256": contract["upstream_repack_sha256"],
            "runtime_qualification_sha256": qualification_sha256,
        }
        object_format = _git_object_format(self.repository_root)
        run_entries: list[dict[str, Any]] = []
        deployment_evidence: dict[str, Any] | None = None
        primary_failure: BaseException | None = None
        try:
            exclusive_request, readback_request = _fresh_transport_requests()
            receipt = self.transport.prepare(
                prepared, deployment_root=layout.deployment_root,
                lock_path=layout.lock_path,
                request_id=exclusive_request[0],
                request_nonce=exclusive_request[1],
            )
            readback = self.transport.readback(
                prepared, deployment_root=layout.deployment_root,
                request_id=readback_request[0],
                request_nonce=readback_request[1],
            )
            deployment_evidence = validate_transport_evidence(
                receipt, readback, layout,
                tuple({
                    "role": artifact.role,
                    "target_path": artifact.target_path,
                    "sha256": artifact.sha256,
                    "size_bytes": len(artifact.payload),
                    "mode": artifact.mode,
                } for artifact in prepared),
                implementation_evidence=self.transport.implementation_evidence(),
                expected_pins=self.expected_transport_pins,
                repository_root=self.repository_root,
                exclusive_request_id=exclusive_request[0],
                exclusive_request_nonce=exclusive_request[1],
                readback_request_id=readback_request[0],
                readback_request_nonce=readback_request[1],
            )
            for run in plan:
                run_dir = phase_dir / "runs" / run.run_id
                role_reservations = {
                    "harness_stdout": by_path[run_dir / "harness.stdout.capture.json"],
                    "harness_stderr": by_path[run_dir / "harness.stderr.capture.json"],
                    "e049c_json": by_path[run_dir / "e049c.json"],
                    "e049c_stderr": by_path[run_dir / "e049c.stderr.capture.json"],
                    "runner_metadata": by_path[run_dir / "runner.json"],
                }
                cell = run.cell()
                harness_argv = canonical_harness_argv(
                    cell, layout.harness_path, run.iterations
                )
                e049c_remote = canonical_remote_output_path(layout, run.run_id)
                e049c_argv = canonical_e049c_launcher_argv(
                    cell, e049c_remote, harness_argv, layout.pmu_path
                )
                environment = safe_environment(run.build_name)
                try:
                    capture = self.transport.capture(
                        run=run, e049c_argv=e049c_argv, environment=environment
                    )
                except Exception as exc:
                    failure = _json_bytes({
                        "schema": "e055-runner-failure/v1",
                        "run_id": run.run_id,
                        "failure_kind": type(exc).__name__,
                        "target_workload_executed": True,
                    })
                    populate_reserved(role_reservations["runner_metadata"], failure)
                    raise TargetRunFailure(
                        f"target transport failed for {run.run_id}; partial phase preserved"
                    ) from exc
                if not isinstance(capture, TargetCapture):
                    failure_reason = "transport returned a noncanonical capture type"
                    capture = TargetCapture(
                        b"", b"", b"", b"", b"", 255, None, (),
                        run.cpu, run.cpu, 0,
                    )
                else:
                    failure_reason = _strict_capture(capture, run)
                    if failure_reason is not None:
                        capture = _sanitized_failure_capture(capture, run)
                populated: dict[str, bytes] = {
                    "harness_stdout": _stream_envelope(
                        run.run_id, "e055_harness", "stdout", capture.child_stdout
                    ),
                    "harness_stderr": _stream_envelope(
                        run.run_id, "e055_harness", "stderr", capture.child_stderr
                    ),
                    "e049c_stderr": _stream_envelope(
                        run.run_id, "e049c_launcher", "stderr", capture.wrapper_stderr
                    ),
                }
                for role, payload in populated.items():
                    populate_reserved(role_reservations[role], payload)
                if capture.e049c_json:
                    populate_reserved(role_reservations["e049c_json"], capture.e049c_json)
                if failure_reason is not None:
                    failure = _json_bytes({
                        "schema": "e055-runner-failure/v1",
                        "run_id": run.run_id,
                        "failure_kind": failure_reason,
                        "exit": {"code": capture.exit_code, "signal": capture.signal},
                        "wrapper_stdout_base64": base64.b64encode(
                            capture.wrapper_stdout
                        ).decode("ascii"),
                        "target_workload_executed": True,
                    })
                    populate_reserved(role_reservations["runner_metadata"], failure)
                    raise TargetRunFailure(
                        f"target run {run.run_id} failed; exact available bytes preserved"
                    )
                role_declarations = {
                    role: _artifact_declaration(
                        self.repository_root, role_reservations[role], role,
                        object_format,
                    )
                    for role in (
                        "harness_stdout", "harness_stderr", "e049c_json", "e049c_stderr"
                    )
                }
                runner = {
                    "schema": "e055-runner-capture/v2",
                    "run_id": run.run_id,
                    "pair_id": run.pair_id,
                    "pair_index": run.pair_index,
                    "pair_order": run.pair_order,
                    "order_index": run.order_index,
                    "build_name": run.build_name,
                    "argv": list(harness_argv),
                    "e049c_argv": list(e049c_argv),
                    "environment": environment,
                    "affinity": {
                        "requested_cpus": [run.cpu],
                        "effective_cpus": list(capture.effective_cpus),
                        "cpu_start": capture.cpu_start,
                        "cpu_end": capture.cpu_end,
                        "migration_count": capture.migration_count,
                    },
                    "exit": {"code": capture.exit_code, "signal": capture.signal},
                    "provenance": {
                        **qualification,
                        "binary_sha256": contract["allowed_builds"]["O3"],
                    },
                    "transport_evidence": deployment_evidence,
                    "artifact_sha256": {
                        role: declaration["sha256"]
                        for role, declaration in role_declarations.items()
                    },
                    "target_workload_executed": True,
                }
                populate_reserved(role_reservations["runner_metadata"], _json_bytes(runner))
                role_declarations["runner_metadata"] = _artifact_declaration(
                    self.repository_root, role_reservations["runner_metadata"],
                    "runner_metadata", object_format,
                )
                run_entries.append({
                    "run_id": run.run_id,
                    "pair_id": run.pair_id,
                    "pair_index": run.pair_index,
                    "pair_order": run.pair_order,
                    "order_index": run.order_index,
                    "build_name": run.build_name,
                    "cell": cell,
                    "artifacts": role_declarations,
                })
        except BaseException as exc:
            primary_failure = exc
        try:
            self.transport.restore()
        except BaseException as restore_failure:
            if primary_failure is not None:
                raise BaseExceptionGroup(
                    "E055 target execution and restoration both failed",
                    [primary_failure, restore_failure],
                )
            raise
        if primary_failure is not None:
            raise primary_failure.with_traceback(primary_failure.__traceback__)
        if deployment_evidence is None:
            raise AssertionError("validated transport evidence was not retained")
        build_declaration = _artifact_declaration(
            self.repository_root, harness_reservation, "harness_executable",
            object_format,
        )
        bundle = {
            "schema": "e055-raw-bundle/v2",
            "experiment": "E055-Q1-HOT-COLD",
            "phase_id": phase_id,
            "qualification": qualification,
            "transport_evidence": deployment_evidence,
            "build_artifacts": [{
                "build_name": "O3", "artifact": build_declaration,
            }],
            "runs": run_entries,
            "target_workload_executed": True,
        }
        populate_reserved(bundle_reservation, _json_bytes(bundle))
        return PhaseExecutionResult(phase_dir, bundle_reservation.path, len(run_entries))
