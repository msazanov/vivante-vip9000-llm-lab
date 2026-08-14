"""Disabled-by-default OpenSSH transport for the bounded E055 microgate.

Construction is local and side-effect free.  Network activity begins only when
``prepare`` is explicitly called.  No request value becomes remote command
text: a fixed content-addressed helper receives canonical JSON on stdin.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from tooling.e055_target_executor import (
    LimitedRun,
    TargetArtifact,
    TargetCapture,
)
from tooling.e055_transport_evidence import (
    BOARD_IDENTITY_RE,
    DEPLOYMENT_BASE,
    HOST_KEY_RE,
    REQUEST_TOKEN_RE,
    SHA256_RE,
    EndpointIdentity,
    ExclusiveDeploymentReceipt,
    FreshReadbackProof,
    RemoteRuntimeObservation,
    TargetArtifactObservation,
    TransportImplementationEvidence,
)


HOST_RE = re.compile(r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?)$")
USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_LOCAL_RE = re.compile(r"^/[A-Za-z0-9._/-]{1,4095}$")
HELPER_REMOTE_RE = re.compile(
    rf"^{re.escape(DEPLOYMENT_BASE)}/helpers/[0-9a-f]{{64}}/helper\.py$"
)
MAX_PROTOCOL_BYTES = 192 * 1024 * 1024
MAX_CLIENT_DIAGNOSTIC_BYTES = 1024 * 1024


class OpenSSHTransportError(RuntimeError):
    """A sanitized transport category with no remote diagnostics or secrets."""


def _sha256_file(path: Path, maximum: int) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or \
                before.st_size < 1 or before.st_size > maximum:
            raise ValueError("local dependency is not one bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("local dependency was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("local dependency grew while hashing")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mode
        ):
            raise ValueError("local dependency identity changed while hashing")
        return digest.hexdigest(), before
    finally:
        os.close(descriptor)


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def _strict_request_token(value: Any) -> str:
    if not isinstance(value, str) or REQUEST_TOKEN_RE.fullmatch(value) is None:
        raise ValueError("request token is not canonical")
    return value


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise OpenSSHTransportError(f"invalid_{label}_schema")
    return value


@dataclass(frozen=True)
class E055OpenSSHConfig:
    host: str
    port: int
    username: str = field(repr=False)
    endpoint_label: str = "orangepi-zero-3w-a733"
    known_hosts_file: Path = Path("/nonexistent")
    expected_host_key_fingerprint: str = ""
    expected_board_identity: str = ""
    ssh_path: Path = Path("/usr/bin/ssh")
    sftp_path: Path = Path("/usr/bin/sftp")
    helper_source_path: Path = Path(__file__).resolve().with_name("e055_remote_helper.py")
    identity_file: Path | None = field(default=None, repr=False)
    agent_socket: Path | None = field(default=None, repr=False)
    connect_timeout_seconds: int = 10
    _test_only_client_paths: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def for_test(cls, **values: Any) -> "E055OpenSSHConfig":
        """Construct an explicit non-operational fake-client test fixture."""

        return cls(_test_only_client_paths=True, **values)

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or HOST_RE.fullmatch(self.host) is None or \
                not isinstance(self.username, str) or USER_RE.fullmatch(self.username) is None or \
                not isinstance(self.endpoint_label, str) or LABEL_RE.fullmatch(self.endpoint_label) is None:
            raise ValueError("endpoint identity fields are not canonical")
        if type(self.port) is not int or not 1 <= self.port <= 65535 or \
                type(self.connect_timeout_seconds) is not int or \
                not 1 <= self.connect_timeout_seconds <= 60:
            raise ValueError("endpoint numeric bounds are invalid")
        if type(self._test_only_client_paths) is not bool:
            raise ValueError("test-only client mode must be one strict boolean")
        if not isinstance(self.expected_host_key_fingerprint, str) or \
                HOST_KEY_RE.fullmatch(self.expected_host_key_fingerprint) is None or \
                not isinstance(self.expected_board_identity, str) or \
                BOARD_IDENTITY_RE.fullmatch(self.expected_board_identity) is None:
            raise ValueError("pinned endpoint digests are not canonical")
        path_values = (
            self.known_hosts_file, self.ssh_path, self.sftp_path,
            self.helper_source_path,
        )
        if self.identity_file is not None:
            path_values += (self.identity_file,)
        if self.agent_socket is not None:
            path_values += (self.agent_socket,)
        for value in path_values:
            if not isinstance(value, Path) or not value.is_absolute() or \
                    ".." in value.parts or SAFE_LOCAL_RE.fullmatch(value.as_posix()) is None:
                raise ValueError("local transport path is not canonical")
        if self.helper_source_path.name != "e055_remote_helper.py" or \
                self.helper_source_path.parent.name != "tooling":
            raise ValueError("helper source must be the reviewed repository file")
        for binary, name in ((self.ssh_path, "ssh"), (self.sftp_path, "sftp")):
            if binary.name != name:
                raise ValueError("OpenSSH executable name is invalid")
            _, status = _sha256_file(binary, 32 * 1024 * 1024)
            if stat.S_IMODE(status.st_mode) != 0o755:
                raise ValueError("OpenSSH executable mode is invalid")
        if not self._test_only_client_paths and (
            self.ssh_path != Path("/usr/bin/ssh")
            or self.sftp_path != Path("/usr/bin/sftp")
        ):
            raise ValueError("operational mode requires exact system OpenSSH clients")
        _sha256_file(self.known_hosts_file, 1024 * 1024)
        _sha256_file(self.helper_source_path, 1024 * 1024)
        if (self.identity_file is None) == (self.agent_socket is None):
            raise ValueError("exactly one external agent or identity credential is required")
        if self.identity_file is not None:
            _, identity_status = _sha256_file(self.identity_file, 1024 * 1024)
            if stat.S_IMODE(identity_status.st_mode) & 0o077:
                raise ValueError("external identity permissions are too broad")
        if self.agent_socket is not None:
            socket_status = self.agent_socket.lstat()
            if not stat.S_ISSOCK(socket_status.st_mode):
                raise ValueError("external agent path is not a socket")


class E055OpenSSHTransport:
    """One explicit, stateful transport instance for one E055 target phase."""

    def __init__(self, config: E055OpenSSHConfig) -> None:
        if type(config) is not E055OpenSSHConfig:
            raise TypeError("OpenSSH transport requires exact configuration")
        self._config = config
        self._helper_payload = config.helper_source_path.read_bytes()
        self._helper_sha256 = hashlib.sha256(self._helper_payload).hexdigest()
        self._helper_remote = (
            f"{DEPLOYMENT_BASE}/helpers/{self._helper_sha256}/helper.py"
        )
        self._owner_id: str | None = None
        self._deployment_root: str | None = None
        self._lock_path: str | None = None
        self._helper_bootstrapped = False
        self._exclusive_attempted = False
        self._bootstrap_cleanup_attempted = False
        self._helper_remote_identity: tuple[int, int] | None = None
        self._capture_sequence = 2
        self.retained_restore_evidence: dict[str, Any] | None = None
        self.retained_finalization_evidence: dict[str, Any] | None = None

    @property
    def bootstrap_cleanup_attempted(self) -> bool:
        return self._bootstrap_cleanup_attempted

    @property
    def _actual_options(self) -> tuple[str, ...]:
        config = self._config
        return (
            "BatchMode=yes", "StrictHostKeyChecking=yes",
            f"UserKnownHostsFile={config.known_hosts_file.as_posix()}",
            "GlobalKnownHostsFile=/dev/null", "CheckHostIP=no",
            "PasswordAuthentication=no", "KbdInteractiveAuthentication=no",
            "NumberOfPasswordPrompts=0", "ForwardAgent=no",
            "ClearAllForwardings=yes", "PermitLocalCommand=no", "RequestTTY=no",
            f"ConnectTimeout={config.connect_timeout_seconds}",
            "ConnectionAttempts=1", "ServerAliveInterval=5",
            "ServerAliveCountMax=2", "LogLevel=ERROR", "IdentitiesOnly=yes",
            "ProxyCommand=none", "ProxyJump=none", "CanonicalizeHostname=no",
        )

    @property
    def _evidence_options(self) -> tuple[str, ...]:
        return tuple(
            "UserKnownHostsFile=external-pinned-file"
            if item.startswith("UserKnownHostsFile=") else item
            for item in self._actual_options
        )

    def _client_environment(self) -> dict[str, str]:
        environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
        if self._config.agent_socket is not None:
            environment["SSH_AUTH_SOCK"] = self._config.agent_socket.as_posix()
        return environment

    def _credential_argv(self) -> tuple[str, ...]:
        if self._config.identity_file is None:
            return ()
        return ("-i", self._config.identity_file.as_posix())

    def _ssh_common(self) -> tuple[str, ...]:
        values: list[str] = ["-F", "/dev/null", "-p", str(self._config.port)]
        for option in self._actual_options:
            values.extend(("-o", option))
        values.extend(self._credential_argv())
        return tuple(values)

    def _sftp_common(self) -> tuple[str, ...]:
        values: list[str] = ["-F", "/dev/null", "-P", str(self._config.port), "-b", "-"]
        for option in self._actual_options:
            values.extend(("-o", option))
        values.extend(self._credential_argv())
        return tuple(values)

    def _destination(self) -> str:
        return f"{self._config.username}@{self._config.host}"

    def _run_client(
        self, argv: Sequence[str], payload: bytes, *, timeout: float,
        maximum_stdout: int, maximum_stderr: int, failure_category: str,
    ) -> tuple[bytes, bytes]:
        if not isinstance(payload, bytes) or len(payload) > MAX_PROTOCOL_BYTES:
            raise OpenSSHTransportError("local_protocol_payload_invalid")
        with tempfile.TemporaryFile() as stdin_file, tempfile.TemporaryFile() as stdout_file, \
                tempfile.TemporaryFile() as stderr_file:
            stdin_file.write(payload)
            stdin_file.seek(0)
            try:
                process = subprocess.Popen(
                    tuple(argv), stdin=stdin_file, stdout=stdout_file, stderr=stderr_file,
                    env=self._client_environment(), close_fds=True,
                    start_new_session=True, shell=False,
                )
            except OSError:
                raise OpenSSHTransportError(failure_category) from None
            deadline = time.monotonic() + timeout
            oversized = False
            while process.poll() is None:
                if os.fstat(stdout_file.fileno()).st_size > maximum_stdout or \
                        os.fstat(stderr_file.fileno()).st_size > maximum_stderr:
                    oversized = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                if time.monotonic() >= deadline:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(0.01)
            try:
                status = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1.0)
                raise OpenSSHTransportError(failure_category) from None
            stdout_size = os.fstat(stdout_file.fileno()).st_size
            stderr_size = os.fstat(stderr_file.fileno()).st_size
            if oversized or status != 0 or stdout_size > maximum_stdout or \
                    stderr_size > maximum_stderr:
                raise OpenSSHTransportError(failure_category) from None
            stdout_file.seek(0)
            stderr_file.seek(0)
            return stdout_file.read(), stderr_file.read()

    def _verify_host_key(self) -> None:
        stdout, stderr = self._run_client(
            (
                "/usr/bin/ssh-keygen", "-lf",
                self._config.known_hosts_file.as_posix(), "-E", "sha256",
            ), b"", timeout=5.0, maximum_stdout=64 * 1024,
            maximum_stderr=64 * 1024, failure_category="host_key_material_invalid",
        )
        if stderr:
            raise OpenSSHTransportError("host_key_material_invalid")
        fingerprints = []
        try:
            lines = stdout.decode("ascii", "strict").splitlines()
        except UnicodeDecodeError:
            raise OpenSSHTransportError("host_key_material_invalid") from None
        for line in lines:
            fields = line.split()
            if len(fields) < 2 or not fields[0].isdigit() or \
                    HOST_KEY_RE.fullmatch(fields[1]) is None:
                raise OpenSSHTransportError("host_key_material_invalid")
            fingerprints.append(fields[1])
        if fingerprints != [self._config.expected_host_key_fingerprint]:
            raise OpenSSHTransportError("host_key_fingerprint_mismatch")

    def _sftp(self, batch: bytes, category: str) -> None:
        argv = (
            self._config.sftp_path.as_posix(), *self._sftp_common(), self._destination(),
        )
        stdout, stderr = self._run_client(
            argv, batch, timeout=float(self._config.connect_timeout_seconds + 10),
            maximum_stdout=MAX_CLIENT_DIAGNOSTIC_BYTES,
            maximum_stderr=MAX_CLIENT_DIAGNOSTIC_BYTES,
            failure_category=category,
        )
        if stdout or stderr:
            raise OpenSSHTransportError(category)

    def _bootstrap_helper(self, request_id: str) -> None:
        if SAFE_LOCAL_RE.fullmatch(self._config.helper_source_path.as_posix()) is None or \
                HELPER_REMOTE_RE.fullmatch(self._helper_remote) is None:
            raise OpenSSHTransportError("helper_path_invalid")
        stage_hash = hashlib.sha256(
            ("e055-helper-stage/v1\0" + self._helper_sha256 + "\0" + request_id).encode("ascii")
        ).hexdigest()
        stage_dir = f"{DEPLOYMENT_BASE}/helpers/staging/{stage_hash}"
        stage_helper = f"{stage_dir}/helper.py"
        target_dir = str(PurePosixPath(self._helper_remote).parent)
        safe_paths = (
            DEPLOYMENT_BASE, f"{DEPLOYMENT_BASE}/helpers",
            f"{DEPLOYMENT_BASE}/helpers/staging", stage_dir, stage_helper,
            target_dir, self._helper_remote,
        )
        if any(
            not isinstance(path, str) or " " in path or "\\" in path
            or not path.startswith(DEPLOYMENT_BASE) for path in safe_paths
        ):
            raise OpenSSHTransportError("helper_path_invalid")
        create_parents = (
            f"-mkdir {DEPLOYMENT_BASE}\n"
            f"-mkdir {DEPLOYMENT_BASE}/helpers\n"
            f"-mkdir {DEPLOYMENT_BASE}/helpers/staging\n"
        ).encode("ascii")
        create_stage = f"mkdir {stage_dir}\n".encode("ascii")
        upload = (
            f"put {self._config.helper_source_path.as_posix()} {stage_helper}\n"
            f"chmod 700 {stage_helper}\n"
        ).encode("ascii")
        promote = f"rename {stage_dir} {target_dir}\n".encode("ascii")
        cleanup = f"-rm {stage_helper}\n-rmdir {stage_dir}\n".encode("ascii")
        stage_owned = False
        try:
            self._sftp(create_parents, "sftp_bootstrap_failed")
            self._sftp(create_stage, "sftp_bootstrap_failed")
            stage_owned = True
            self._sftp(upload, "sftp_bootstrap_failed")
            self._sftp(promote, "sftp_bootstrap_failed")
            stage_owned = False
        except OpenSSHTransportError:
            self._bootstrap_cleanup_attempted = True
            if stage_owned:
                try:
                    self._sftp(cleanup, "sftp_cleanup_failed")
                except OpenSSHTransportError:
                    pass
            raise
        self._helper_bootstrapped = True

    def _invoke(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = _canonical_json(request)
        remote_command = f"exec /usr/bin/python3 -- {self._helper_remote}"
        argv = (
            self._config.ssh_path.as_posix(), *self._ssh_common(), "-T",
            self._destination(), remote_command,
        )
        stdout, stderr = self._run_client(
            argv, payload, timeout=130.0, maximum_stdout=MAX_PROTOCOL_BYTES,
            maximum_stderr=MAX_CLIENT_DIAGNOSTIC_BYTES,
            failure_category="ssh_protocol_failed",
        )
        if stderr or not stdout or len(stdout) > MAX_PROTOCOL_BYTES:
            raise OpenSSHTransportError("ssh_protocol_failed")
        try:
            document = json.loads(stdout.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raise OpenSSHTransportError("ssh_response_invalid") from None
        if not isinstance(document, dict) or _canonical_json(document) != stdout:
            raise OpenSSHTransportError("ssh_response_invalid")
        return document

    def _request_base(self, operation: str, sequence: int, request_id: str, nonce: str) -> dict[str, Any]:
        if self._owner_id is None or self._deployment_root is None:
            raise OpenSSHTransportError("transport_lifecycle_inactive")
        return {
            "schema": "e055-remote-request/v1", "operation": operation,
            "operation_sequence": sequence, "request_id": _strict_request_token(request_id),
            "request_nonce": _strict_request_token(nonce),
            "expected_board_identity": self._config.expected_board_identity,
            "owner_id": self._owner_id, "deployment_root": self._deployment_root,
        }

    def _identity(self, value: Any) -> EndpointIdentity:
        exact = _exact(
            value, {"trust_mode", "endpoint_label", "board_identity", "host_key_fingerprint"},
            "endpoint_identity",
        )
        if exact["endpoint_label"] != self._config.endpoint_label or \
                exact["board_identity"] != self._config.expected_board_identity or \
                exact["host_key_fingerprint"] is not None:
            raise OpenSSHTransportError("endpoint_identity_mismatch")
        return EndpointIdentity(
            "pinned_host_key", self._config.endpoint_label,
            self._config.expected_board_identity,
            self._config.expected_host_key_fingerprint,
        )

    def _runtime(self, value: Any, request: Mapping[str, Any]) -> RemoteRuntimeObservation:
        keys = {
            "schema", "operation_sequence", "request_id", "request_nonce",
            "helper_path", "helper_sha256", "helper_size_bytes", "helper_mode",
            "helper_device", "helper_inode", "helper_protocol",
            "python_requested_path", "python_realpath", "python_version",
            "python_sha256", "python_size_bytes", "python_mode", "python_device",
            "python_inode",
        }
        exact = _exact(value, keys, "runtime")
        runtime = RemoteRuntimeObservation(**exact)
        if runtime.operation_sequence != request["operation_sequence"] or \
                runtime.request_id != request["request_id"] or \
                runtime.request_nonce != request["request_nonce"] or \
                runtime.helper_path != self._helper_remote or \
                runtime.helper_sha256 != self._helper_sha256 or \
                runtime.helper_size_bytes != len(self._helper_payload) or \
                type(runtime.helper_device) is not int or runtime.helper_device < 0 or \
                type(runtime.helper_inode) is not int or runtime.helper_inode <= 0:
            raise OpenSSHTransportError("runtime_identity_mismatch")
        return runtime

    def _observations(
        self, value: Any, request: Mapping[str, Any],
        artifacts: tuple[TargetArtifact, ...],
    ) -> tuple[TargetArtifactObservation, ...]:
        if not isinstance(value, list) or len(value) != len(artifacts):
            raise OpenSSHTransportError("artifact_observation_invalid")
        result: list[TargetArtifactObservation] = []
        keys = {
            "role", "target_path", "sha256", "size_bytes", "mode", "device",
            "inode", "operation_sequence", "request_id", "request_nonce",
        }
        for document, artifact in zip(value, artifacts, strict=True):
            exact = _exact(document, keys, "artifact_observation")
            observation = TargetArtifactObservation(**exact)
            if (observation.role, observation.target_path, observation.sha256,
                    observation.size_bytes, observation.mode) != (
                artifact.role, artifact.target_path, artifact.sha256,
                len(artifact.payload), artifact.mode,
            ) or observation.operation_sequence != request["operation_sequence"] or \
                    observation.request_id != request["request_id"] or \
                    observation.request_nonce != request["request_nonce"]:
                raise OpenSSHTransportError("artifact_observation_mismatch")
            result.append(observation)
        return tuple(result)

    def _artifact_request(self, artifact: TargetArtifact, payload: bool) -> dict[str, Any]:
        if type(artifact) is not TargetArtifact or artifact.role not in (
            "harness_executable", "pmu_executable",
        ) or not isinstance(artifact.payload, bytes) or not artifact.payload or \
                hashlib.sha256(artifact.payload).hexdigest() != artifact.sha256 or \
                SHA256_RE.fullmatch(artifact.sha256) is None or \
                type(artifact.mode) is not int or artifact.mode != 0o755:
            raise ValueError("target artifact is not canonical")
        document: dict[str, Any] = {
            "role": artifact.role, "target_path": artifact.target_path,
            "sha256": artifact.sha256, "size_bytes": len(artifact.payload),
            "mode": artifact.mode,
        }
        if payload:
            document["payload_base64"] = base64.b64encode(artifact.payload).decode("ascii")
        return document

    def implementation_evidence(self) -> TransportImplementationEvidence:
        ssh_hash, ssh_status = _sha256_file(self._config.ssh_path, 32 * 1024 * 1024)
        sftp_hash, sftp_status = _sha256_file(
            self._config.sftp_path, 32 * 1024 * 1024,
        )
        known_hosts_hash, _ = _sha256_file(
            self._config.known_hosts_file, 1024 * 1024,
        )
        repository_root = self._config.helper_source_path.parents[1]
        helper_relative = self._config.helper_source_path.relative_to(
            repository_root
        ).as_posix()
        helper_blob_result = subprocess.run(
            ("git", "rev-parse", f"HEAD:{helper_relative}"), cwd=repository_root,
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        helper_blob = helper_blob_result.stdout.strip()
        if helper_blob_result.returncode != 0 or re.fullmatch(
            r"[0-9a-f]{40}(?:[0-9a-f]{24})?", helper_blob
        ) is None:
            raise OpenSSHTransportError("helper_git_binding_failed")
        target_endpoint_sha256 = hashlib.sha256(
            (
                "e055-target-endpoint/v1\0" + self._config.host + "\0"
                + str(self._config.port)
            ).encode("ascii")
        ).hexdigest()
        stdout, stderr = self._run_client(
            (self._config.ssh_path.as_posix(), "-V"), b"", timeout=5.0,
            maximum_stdout=4096, maximum_stderr=4096,
            failure_category="openssh_version_failed",
        )
        version_bytes = stderr if stderr else stdout
        try:
            version = version_bytes.decode("ascii").split(",", 1)[0].strip()
        except UnicodeDecodeError:
            raise OpenSSHTransportError("openssh_version_invalid") from None
        options = self._evidence_options
        digest = hashlib.sha256(("\n".join(options) + "\n").encode("ascii")).hexdigest()
        return TransportImplementationEvidence(
            "e055-openssh-implementation/v1", "openssh_fixed_helper",
            "test_fixture_clients" if self._config._test_only_client_paths
            else "operational_system_clients",
            "tooling/e055_remote_helper.py", self._helper_sha256,
            len(self._helper_payload), helper_blob,
            self._config.ssh_path.as_posix(), ssh_hash,
            ssh_status.st_size, stat.S_IMODE(ssh_status.st_mode), version,
            self._config.sftp_path.as_posix(), sftp_hash,
            sftp_status.st_size, stat.S_IMODE(sftp_status.st_mode),
            "/dev/null", options, digest, known_hosts_hash,
            target_endpoint_sha256, "external_agent_or_identity",
        )

    def prepare(
        self, artifacts: tuple[TargetArtifact, ...], *, deployment_root: str,
        lock_path: str, request_id: str, request_nonce: str,
    ) -> ExclusiveDeploymentReceipt:
        if self._owner_id is not None:
            raise OpenSSHTransportError("transport_already_prepared")
        if not isinstance(artifacts, tuple) or len(artifacts) != 2 or \
                lock_path != f"{DEPLOYMENT_BASE}/locks/a733-target.lock" or \
                not isinstance(deployment_root, str) or not deployment_root.startswith(
                    f"{DEPLOYMENT_BASE}/deployments/"
                ):
            raise ValueError("deployment request is not canonical")
        request_id = _strict_request_token(request_id)
        request_nonce = _strict_request_token(request_nonce)
        declarations = [self._artifact_request(artifact, True) for artifact in artifacts]
        self._verify_host_key()
        self._bootstrap_helper(request_id)
        self._owner_id = secrets.token_hex(32)
        self._deployment_root = deployment_root
        self._lock_path = lock_path
        request = self._request_base("exclusive_deploy", 1, request_id, request_nonce)
        request.update({"lock_path": lock_path, "artifacts": declarations})
        self._exclusive_attempted = True
        response = self._invoke(request)
        exact = _exact(response, {
            "schema", "operation_sequence", "request_id", "request_nonce",
            "deployment_root", "lock_path", "lock_acquired_exclusively",
            "deployment_created_exclusively", "endpoint_identity", "artifacts",
            "runtime",
        }, "exclusive_receipt")
        if exact["schema"] != "e055-exclusive-deployment-receipt/v1" or \
                exact["operation_sequence"] != 1 or exact["request_id"] != request_id or \
                exact["request_nonce"] != request_nonce or \
                exact["deployment_root"] != deployment_root or exact["lock_path"] != lock_path or \
                exact["lock_acquired_exclusively"] is not True or \
                exact["deployment_created_exclusively"] is not True:
            raise OpenSSHTransportError("exclusive_receipt_mismatch")
        runtime = self._runtime(exact["runtime"], request)
        self._helper_remote_identity = (
            runtime.helper_device, runtime.helper_inode,
        )
        return ExclusiveDeploymentReceipt(
            exact["schema"], 1, request_id, request_nonce, deployment_root, lock_path,
            True, True, self._identity(exact["endpoint_identity"]),
            self._observations(exact["artifacts"], request, artifacts),
            runtime,
        )

    def readback(
        self, artifacts: tuple[TargetArtifact, ...], *, deployment_root: str,
        request_id: str, request_nonce: str,
    ) -> FreshReadbackProof:
        if deployment_root != self._deployment_root or self._owner_id is None:
            raise OpenSSHTransportError("transport_lifecycle_mismatch")
        declarations = [self._artifact_request(artifact, False) for artifact in artifacts]
        request = self._request_base("fresh_readback", 2, request_id, request_nonce)
        request["artifacts"] = declarations
        response = self._invoke(request)
        exact = _exact(response, {
            "schema", "operation_sequence", "request_id", "request_nonce",
            "deployment_root", "endpoint_identity", "artifacts", "runtime",
        }, "readback")
        if exact["schema"] != "e055-fresh-readback-proof/v1" or \
                exact["operation_sequence"] != 2 or exact["request_id"] != request_id or \
                exact["request_nonce"] != request_nonce or \
                exact["deployment_root"] != deployment_root:
            raise OpenSSHTransportError("readback_mismatch")
        runtime = self._runtime(exact["runtime"], request)
        identity = (runtime.helper_device, runtime.helper_inode)
        if self._helper_remote_identity is None or \
                identity != self._helper_remote_identity:
            raise OpenSSHTransportError("runtime_identity_mismatch")
        return FreshReadbackProof(
            exact["schema"], 2, request_id, request_nonce, deployment_root,
            self._identity(exact["endpoint_identity"]),
            self._observations(exact["artifacts"], request, artifacts),
            runtime,
        )

    def capture(
        self, *, run: LimitedRun, e049c_argv: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> TargetCapture:
        if type(run) is not LimitedRun or not isinstance(e049c_argv, tuple) or \
                not isinstance(environment, Mapping):
            raise ValueError("capture inputs are not canonical types")
        self._capture_sequence += 1
        request_id, nonce = secrets.token_hex(32), secrets.token_hex(32)
        request = self._request_base(
            "capture", self._capture_sequence, request_id, nonce,
        )
        request.update({
            "run_id": run.run_id, "cpu": run.cpu, "argv": list(e049c_argv),
            "environment": dict(environment), "timeout_ms": 120000,
        })
        response = self._invoke(request)
        exact = _exact(response, {
            "schema", "operation_sequence", "request_id", "request_nonce", "run_id",
            "exit", "streams", "affinity", "runtime",
        }, "capture")
        if exact["schema"] != "e055-capture-result/v1" or \
                exact["operation_sequence"] != self._capture_sequence or \
                exact["request_id"] != request_id or exact["request_nonce"] != nonce or \
                exact["run_id"] != run.run_id:
            raise OpenSSHTransportError("capture_response_mismatch")
        self._runtime(exact["runtime"], request)
        exit_document = _exact(exact["exit"], {"code", "signal"}, "capture_exit")
        affinity = _exact(
            exact["affinity"], {"effective_cpus", "cpu_start", "cpu_end", "migration_count"},
            "capture_affinity",
        )
        streams = _exact(exact["streams"], {
            "child_stdout_base64", "child_stderr_base64", "e049c_json_base64",
            "wrapper_stdout_base64", "wrapper_stderr_base64",
        }, "capture_streams")
        try:
            decoded = {
                key: base64.b64decode(value, validate=True) for key, value in streams.items()
                if isinstance(value, str)
            }
        except (ValueError, base64.binascii.Error):
            raise OpenSSHTransportError("capture_stream_invalid") from None
        if len(decoded) != 5 or any(len(value) > 16 * 1024 * 1024 for value in decoded.values()):
            raise OpenSSHTransportError("capture_stream_invalid")
        try:
            effective = tuple(affinity["effective_cpus"])
            return TargetCapture(
                decoded["child_stdout_base64"], decoded["child_stderr_base64"],
                decoded["e049c_json_base64"], decoded["wrapper_stdout_base64"],
                decoded["wrapper_stderr_base64"], exit_document["code"],
                exit_document["signal"], effective, affinity["cpu_start"],
                affinity["cpu_end"], affinity["migration_count"],
            )
        except (KeyError, TypeError):
            raise OpenSSHTransportError("capture_response_invalid") from None

    def restore(self) -> None:
        if not self._helper_bootstrapped or self._owner_id is None or \
                self._deployment_root is None or self._lock_path is None:
            return
        if self._exclusive_attempted:
            request_id, nonce = secrets.token_hex(32), secrets.token_hex(32)
            request = self._request_base("restore", 1000, request_id, nonce)
            request["lock_path"] = self._lock_path
            response = self._invoke(request)
            exact = _exact(response, {
                "schema", "operation_sequence", "request_id", "request_nonce",
                "restored", "deployment_removed", "lock_released", "runtime",
            }, "restore")
            if exact["schema"] != "e055-restore-result/v1" or \
                    exact["operation_sequence"] != 1000 or \
                    exact["request_id"] != request_id or exact["request_nonce"] != nonce or \
                    exact["restored"] is not True or exact["deployment_removed"] is not True or \
                    exact["lock_released"] is not True:
                raise OpenSSHTransportError("restore_response_mismatch")
            self._runtime(exact["runtime"], request)
            self.retained_restore_evidence = dict(exact)
        final_id, final_nonce = secrets.token_hex(32), secrets.token_hex(32)
        finalize = self._request_base("finalize_helper", 1001, final_id, final_nonce)
        finalize["lock_path"] = self._lock_path
        response = self._invoke(finalize)
        exact_final = _exact(response, {
            "schema", "operation_sequence", "request_id", "request_nonce",
            "helper_removed", "helper_sha256", "helper_device", "helper_inode",
        }, "finalization")
        if exact_final["schema"] != "e055-helper-finalization/v1" or \
                exact_final["operation_sequence"] != 1001 or \
                exact_final["request_id"] != final_id or \
                exact_final["request_nonce"] != final_nonce or \
                exact_final["helper_removed"] is not True or \
                exact_final["helper_sha256"] != self._helper_sha256 or \
                type(exact_final["helper_device"]) is not int or \
                type(exact_final["helper_inode"]) is not int or \
                self._helper_remote_identity is None or \
                (exact_final["helper_device"], exact_final["helper_inode"]) != \
                self._helper_remote_identity:
            raise OpenSSHTransportError("finalization_response_mismatch")
        self.retained_finalization_evidence = dict(exact_final)
        self._helper_bootstrapped = False
