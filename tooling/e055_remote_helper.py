#!/usr/bin/env python3
"""Fixed, fail-closed remote protocol helper for the bounded E055 microgate.

The deployed copy is intentionally self-contained and uses only the Python
standard library.  It never evaluates a command string: requests are one
bounded canonical JSON document and process execution uses argv arrays.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping


PROTOCOL = "e055-remote-helper/v1"
REQUEST_SCHEMA = "e055-remote-request/v1"
BASE = "/tmp/e055-q1-hot-cold"
MAX_REQUEST_BYTES = 192 * 1024 * 1024
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BOARD_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DEPLOYMENT_RE = re.compile(rf"^{re.escape(BASE)}/deployments/[0-9a-f]{{64}}$")
HELPER_RE = re.compile(rf"^{re.escape(BASE)}/helpers/[0-9a-f]{{64}}/helper\.py$")
LOCK_PATH = f"{BASE}/locks/a733-target.lock"
ROLE_NAMES = {
    "harness_executable": "e055-O3-aarch64",
    "pmu_executable": "a733-pmu-exec-aarch64",
}
RUN_ID_RE = re.compile(r"^cpu([06])-pair([1-5])-(hot|cold)$")
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
OWNED_PROCESS_NAME = "owned-process.json"
OWNED_PROCESS_SCHEMA = "e055-owned-process/v1"


_CAPTURE_GATE = r"""
import base64, json, os, sys
path = sys.argv[1]
document = json.loads(base64.b64decode(sys.argv[2], validate=True).decode("ascii"))
payload = open("/proc/self/stat", "r", encoding="ascii").read()
boundary = payload.rfind(") ")
fields = payload[boundary + 2:].split()
pid = os.getpid()
pgid = int(fields[2])
session = int(fields[3])
start_time = int(fields[19])
if boundary < 0 or pid <= 1 or pgid != pid or session != pid or start_time <= 0:
    raise SystemExit(125)
document.update({"pid": pid, "pgid": pgid, "session": session,
                 "start_time": start_time})
encoded = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
descriptor = os.open(path, flags, 0o600)
try:
    offset = 0
    while offset < len(encoded):
        written = os.write(descriptor, encoded[offset:])
        if written <= 0:
            raise OSError("short owned-process write")
        offset += written
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.execve(sys.argv[3], sys.argv[3:], dict(os.environ))
""".strip()


class ProtocolError(ValueError):
    """The request or observed remote state is not canonical."""


def _strict_int(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _exact_mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProtocolError(f"{label} keys are not canonical")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def _sha256_file(path: Path, maximum: int) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or \
                before.st_size < 1 or before.st_size > maximum:
            raise ProtocolError("observed file is not one bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ProtocolError("observed file was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProtocolError("observed file grew while hashing")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mode
        ):
            raise ProtocolError("observed file identity changed while hashing")
        return digest.hexdigest(), before
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class HelperContext:
    """Trusted process-local paths; request values never alter this mapping."""

    filesystem_root: Path
    helper_source: Path
    helper_remote_path: str
    python_requested_path: str
    board_identity_payloads: tuple[bytes, ...]
    endpoint_label: str

    @classmethod
    def production(cls) -> "HelperContext":
        identity_payloads = []
        for path in (Path("/etc/machine-id"), Path("/proc/device-tree/model")):
            try:
                identity_payloads.append(path.read_bytes())
            except OSError as exc:
                raise ProtocolError("fixed board identity input is unavailable") from exc
        source = Path(__file__).resolve()
        source_hash, _ = _sha256_file(source, 1024 * 1024)
        remote = f"{BASE}/helpers/{source_hash}/helper.py"
        if source.as_posix() != remote:
            raise ProtocolError("helper was not executed from its content-addressed path")
        return cls(
            Path("/"), source, remote, "/usr/bin/python3",
            tuple(identity_payloads), "orangepi-zero-3w-a733",
        )

    @classmethod
    def for_test(
        cls, *, filesystem_root: Path, helper_source: Path,
        helper_remote_path: str, board_identity_payloads: tuple[bytes, ...],
    ) -> "HelperContext":
        return cls(
            Path(filesystem_root).resolve(), Path(helper_source).resolve(),
            helper_remote_path, "/usr/bin/python3", board_identity_payloads,
            "test-a733-endpoint",
        )

    @property
    def board_identity_digest(self) -> str:
        digest = hashlib.sha256(b"e055-board-identity/v1\0")
        for payload in self.board_identity_payloads:
            if not isinstance(payload, bytes) or not payload:
                raise ProtocolError("board identity payload is invalid")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return "sha256:" + digest.hexdigest()

    def local_path(self, remote_path: str) -> Path:
        if not isinstance(remote_path, str) or len(remote_path) > 4096 or \
                "\\" in remote_path or any(ord(character) < 32 for character in remote_path):
            raise ProtocolError("remote path contains invalid text")
        parsed = PurePosixPath(remote_path)
        if not parsed.is_absolute() or parsed.as_posix() != remote_path or \
                any(part in ("", ".", "..") for part in parsed.parts) or \
                not remote_path.startswith(BASE + "/"):
            raise ProtocolError("remote path is not canonical under the E055 root")
        if self.filesystem_root == Path("/"):
            return Path(remote_path)
        return self.filesystem_root.joinpath(*parsed.parts[1:])


def _validate_common(request: Any, operation_keys: set[str]) -> Mapping[str, Any]:
    common = {
        "schema", "operation", "operation_sequence", "request_id",
        "request_nonce", "expected_board_identity", "owner_id",
        "deployment_root",
    }
    exact = _exact_mapping(request, common | operation_keys, "request")
    if exact["schema"] != REQUEST_SCHEMA or not isinstance(exact["operation"], str):
        raise ProtocolError("request schema or operation is invalid")
    if not _strict_int(exact["operation_sequence"], 1):
        raise ProtocolError("operation sequence must be a positive strict integer")
    for key in ("request_id", "request_nonce", "owner_id"):
        if not isinstance(exact[key], str) or SHA256_RE.fullmatch(exact[key]) is None:
            raise ProtocolError(f"{key} is not canonical")
    if len({exact["request_id"], exact["request_nonce"]}) != 2:
        raise ProtocolError("request ID and nonce must be distinct")
    if not isinstance(exact["expected_board_identity"], str) or \
            BOARD_RE.fullmatch(exact["expected_board_identity"]) is None:
        raise ProtocolError("expected board identity is not canonical")
    if not isinstance(exact["deployment_root"], str) or \
            DEPLOYMENT_RE.fullmatch(exact["deployment_root"]) is None:
        raise ProtocolError("deployment root is not canonical")
    return exact


def _ensure_safe_directory(path: Path) -> None:
    """Create missing parents while rejecting symlink/non-directory components."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    status = cursor.lstat()
    if not stat.S_ISDIR(status.st_mode):
        raise ProtocolError("directory ancestry contains a non-directory")
    for component in reversed(missing):
        os.mkdir(component, 0o700)
    cursor = path
    while True:
        status = cursor.lstat()
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise ProtocolError("directory ancestry is not safe")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def _artifact_declarations(value: Any, deployment_root: str, *, payloads: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError("exactly two target artifacts are required")
    required = {"role", "target_path", "sha256", "size_bytes", "mode"}
    if payloads:
        required.add("payload_base64")
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        exact = dict(_exact_mapping(item, required, "artifact"))
        role = exact["role"]
        if role not in ROLE_NAMES or role in seen:
            raise ProtocolError("artifact role is unknown or duplicated")
        seen.add(role)
        digest = exact["sha256"]
        expected_path = f"{deployment_root}/artifacts/{digest}/{ROLE_NAMES[role]}"
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or \
                exact["target_path"] != expected_path or \
                not _strict_int(exact["size_bytes"], 1) or \
                exact["size_bytes"] > MAX_ARTIFACT_BYTES or \
                type(exact["mode"]) is not int or exact["mode"] != 0o755:
            raise ProtocolError("artifact declaration is not canonical")
        if payloads:
            encoded = exact["payload_base64"]
            if not isinstance(encoded, str) or len(encoded) > (MAX_ARTIFACT_BYTES * 4 // 3 + 8):
                raise ProtocolError("artifact payload is not bounded base64")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ProtocolError("artifact payload is invalid base64") from exc
            if len(payload) != exact["size_bytes"] or hashlib.sha256(payload).hexdigest() != digest:
                raise ProtocolError("artifact payload does not match its declaration")
            exact["decoded_payload"] = payload
        artifacts.append(exact)
    if seen != set(ROLE_NAMES):
        raise ProtocolError("target artifact roles are incomplete")
    return artifacts


def _write_exclusive(path: Path, payload: bytes, mode: int) -> os.stat_result:
    _ensure_safe_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | \
        getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short exclusive write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
            raise ProtocolError("exclusive output is not one regular file")
        return result
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _observe_artifact(
    artifact: Mapping[str, Any], context: HelperContext, *,
    sequence: int, request_id: str, request_nonce: str,
) -> dict[str, Any]:
    path = context.local_path(artifact["target_path"])
    digest, status = _sha256_file(path, MAX_ARTIFACT_BYTES)
    if digest != artifact["sha256"] or status.st_size != artifact["size_bytes"] or \
            stat.S_IMODE(status.st_mode) != artifact["mode"]:
        raise ProtocolError("fresh artifact observation does not match declaration")
    return {
        "role": artifact["role"], "target_path": artifact["target_path"],
        "sha256": digest, "size_bytes": status.st_size,
        "mode": stat.S_IMODE(status.st_mode), "device": status.st_dev,
        "inode": status.st_ino, "operation_sequence": sequence,
        "request_id": request_id, "request_nonce": request_nonce,
    }


def _runtime(context: HelperContext, request: Mapping[str, Any]) -> dict[str, Any]:
    helper_path = context.local_path(context.helper_remote_path)
    helper_hash, helper_status = _sha256_file(helper_path, 1024 * 1024)
    path_hash = PurePosixPath(context.helper_remote_path).parent.name
    if HELPER_RE.fullmatch(context.helper_remote_path) is None or helper_hash != path_hash or \
            stat.S_IMODE(helper_status.st_mode) != 0o700:
        raise ProtocolError("running helper identity is not content-addressed")
    requested = Path(context.python_requested_path)
    real = requested.resolve(strict=True)
    python_hash, python_status = _sha256_file(real, 128 * 1024 * 1024)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return {
        "schema": "e055-remote-runtime-observation/v1",
        "operation_sequence": request["operation_sequence"],
        "request_id": request["request_id"], "request_nonce": request["request_nonce"],
        "helper_path": context.helper_remote_path, "helper_sha256": helper_hash,
        "helper_size_bytes": helper_status.st_size,
        "helper_mode": stat.S_IMODE(helper_status.st_mode),
        "helper_device": helper_status.st_dev, "helper_inode": helper_status.st_ino,
        "helper_protocol": PROTOCOL,
        "python_requested_path": context.python_requested_path,
        "python_realpath": real.as_posix(), "python_version": version,
        "python_sha256": python_hash, "python_size_bytes": python_status.st_size,
        "python_mode": stat.S_IMODE(python_status.st_mode),
        "python_device": python_status.st_dev, "python_inode": python_status.st_ino,
    }


def _endpoint(context: HelperContext) -> dict[str, Any]:
    return {
        "trust_mode": "pinned_host_key", "endpoint_label": context.endpoint_label,
        "board_identity": context.board_identity_digest,
        "host_key_fingerprint": None,
    }


def _remove_created_deployment(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != identity or \
            not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise ProtocolError("created deployment identity was replaced")
    shutil.rmtree(path)


def _lock_document(owner_id: str, deployment_root: str, identity: tuple[int, int]) -> bytes:
    return _canonical_json({
        "schema": "e055-remote-lock/v1", "owner_id": owner_id,
        "deployment_root": deployment_root,
        "deployment_device": identity[0], "deployment_inode": identity[1],
    })


def _read_lock(path: Path, request: Mapping[str, Any]) -> tuple[dict[str, Any], os.stat_result]:
    payload_hash, status = _sha256_file(path, 4096)
    del payload_hash
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("lock owner record is malformed") from exc
    _exact_mapping(document, {
        "schema", "owner_id", "deployment_root", "deployment_device",
        "deployment_inode",
    }, "lock")
    if document["schema"] != "e055-remote-lock/v1" or \
            document["owner_id"] != request["owner_id"] or \
            document["deployment_root"] != request["deployment_root"] or \
            not _strict_int(document["deployment_device"]) or \
            not _strict_int(document["deployment_inode"], 1):
        raise ProtocolError("lock ownership does not match this request")
    return document, status


def _exclusive_deploy(request: Any, context: HelperContext) -> dict[str, Any]:
    exact = _validate_common(request, {"lock_path", "artifacts"})
    if exact["operation"] != "exclusive_deploy" or exact["lock_path"] != LOCK_PATH:
        raise ProtocolError("exclusive deployment path or operation is invalid")
    if exact["expected_board_identity"] != context.board_identity_digest:
        raise ProtocolError("board identity digest mismatch")
    artifacts = _artifact_declarations(exact["artifacts"], exact["deployment_root"], payloads=True)
    lock_path = context.local_path(exact["lock_path"])
    deployment_path = context.local_path(exact["deployment_root"])
    _ensure_safe_directory(lock_path.parent)
    _ensure_safe_directory(deployment_path.parent)
    lock_fd = os.open(
        lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0), 0o600,
    )
    lock_identity = os.fstat(lock_fd)
    deployment_identity: tuple[int, int] | None = None
    try:
        os.mkdir(deployment_path, 0o700)
        status = deployment_path.lstat()
        deployment_identity = (status.st_dev, status.st_ino)
        record = _lock_document(exact["owner_id"], exact["deployment_root"], deployment_identity)
        offset = 0
        while offset < len(record):
            offset += os.write(lock_fd, record[offset:])
        os.fsync(lock_fd)
        for artifact in artifacts:
            _write_exclusive(
                context.local_path(artifact["target_path"]),
                artifact["decoded_payload"], artifact["mode"],
            )
        observations = [
            _observe_artifact(
                artifact, context, sequence=exact["operation_sequence"],
                request_id=exact["request_id"], request_nonce=exact["request_nonce"],
            ) for artifact in artifacts
        ]
    except BaseException:
        if deployment_identity is not None:
            _remove_created_deployment(deployment_path, deployment_identity)
        try:
            current = lock_path.lstat()
            if (current.st_dev, current.st_ino) == (lock_identity.st_dev, lock_identity.st_ino):
                os.unlink(lock_path)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(lock_fd)
    return {
        "schema": "e055-exclusive-deployment-receipt/v1",
        "operation_sequence": exact["operation_sequence"],
        "request_id": exact["request_id"], "request_nonce": exact["request_nonce"],
        "deployment_root": exact["deployment_root"], "lock_path": exact["lock_path"],
        "lock_acquired_exclusively": True, "deployment_created_exclusively": True,
        "endpoint_identity": _endpoint(context), "artifacts": observations,
        "runtime": _runtime(context, exact),
    }


def _fresh_readback(request: Any, context: HelperContext) -> dict[str, Any]:
    exact = _validate_common(request, {"artifacts"})
    if exact["operation"] != "fresh_readback" or \
            exact["expected_board_identity"] != context.board_identity_digest:
        raise ProtocolError("fresh readback operation or board identity is invalid")
    artifacts = _artifact_declarations(exact["artifacts"], exact["deployment_root"], payloads=False)
    _read_lock(context.local_path(LOCK_PATH), exact)
    observations = [
        _observe_artifact(
            artifact, context, sequence=exact["operation_sequence"],
            request_id=exact["request_id"], request_nonce=exact["request_nonce"],
        ) for artifact in artifacts
    ]
    return {
        "schema": "e055-fresh-readback-proof/v1",
        "operation_sequence": exact["operation_sequence"],
        "request_id": exact["request_id"], "request_nonce": exact["request_nonce"],
        "deployment_root": exact["deployment_root"],
        "endpoint_identity": _endpoint(context), "artifacts": observations,
        "runtime": _runtime(context, exact),
    }


def _capture_argv(request: Mapping[str, Any], context: HelperContext) -> tuple[str, ...]:
    argv = request["argv"]
    if not isinstance(argv, list) or not argv or len(argv) > 64 or any(
        not isinstance(value, str) or not value or len(value) > 4096
        or "\\" in value or any(ord(character) < 32 for character in value)
        for value in argv
    ):
        raise ProtocolError("capture argv is not bounded canonical text")
    match = RUN_ID_RE.fullmatch(request["run_id"])
    if match is None or int(match.group(1)) != request["cpu"]:
        raise ProtocolError("capture run ID does not match its CPU")
    cache_state = "hot_repeat" if match.group(3) == "hot" else "cold_conditioned"
    iterations = "250" if cache_state == "hot_repeat" else "1"
    output = f"{request['deployment_root']}/captures/{request['run_id']}/e049c.json"
    child_stdout = output.replace("e049c.json", "target-harness.stdout.raw")
    child_stderr = output.replace("e049c.json", "target-harness.stderr.raw")
    pmu = argv[0]
    if re.fullmatch(
        rf"{re.escape(request['deployment_root'])}/artifacts/[0-9a-f]{{64}}/a733-pmu-exec-aarch64",
        pmu,
    ) is None:
        raise ProtocolError("capture PMU executable path is invalid")
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise ProtocolError("capture argv has no child separator") from exc
    if argv.count("--") != 1 or separator + 4 >= len(argv):
        raise ProtocolError("capture argv child separator is ambiguous")
    harness_path = argv[separator + 4]
    if re.fullmatch(
        rf"{re.escape(request['deployment_root'])}/artifacts/[0-9a-f]{{64}}/e055-O3-aarch64",
        harness_path,
    ) is None:
        raise ProtocolError("capture harness executable path is invalid")
    child = [
        "taskset", "-c", str(request["cpu"]), harness_path,
        "--mode", "full_dotprod", "--cache-state", cache_state,
        "--working-set-bytes", "65536", "--cpu", str(request["cpu"]),
        "--iterations", iterations,
    ]
    if cache_state == "cold_conditioned":
        child.extend(("--budget-ms", "0", "--warmup", "0"))
    else:
        child.extend(("--warmup", "16"))
    child.extend(("--thrash-bytes", "67108864", "--sync"))
    expected = [
        pmu, "-o", output, "--child-stdout", child_stdout,
        "--child-stderr", child_stderr, "--event-group", "core",
        "--min-running-ratio", "0.95", "--start-on-ready",
        "--sync-timeout-ms", "5000", "--max-temp-c", "85", "--", *child,
    ]
    if argv != expected:
        raise ProtocolError("capture argv is not the exact bounded E055 command")
    local = list(argv)
    for index in (0, 2, 4, 6, separator + 4):
        local[index] = context.local_path(argv[index]).as_posix()
    return tuple(local)


def _descendants(leader_pid: int) -> set[int]:
    """Return descendants still rooted below the live wrapper process."""

    discovered: set[int] = set()
    pending = [leader_pid]
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            fields = children_path.read_text(encoding="ascii").split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            raise ProtocolError("cannot inspect the target process tree") from exc
        for field in fields:
            if not field.isascii() or not field.isdigit():
                raise ProtocolError("process tree contains a malformed PID")
            child = int(field)
            if child <= 1 or child in discovered or child == os.getpid():
                raise ProtocolError("process tree identity is invalid")
            discovered.add(child)
            pending.append(child)
    return discovered


def _processor(pid: int) -> int | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    boundary = payload.rfind(") ")
    if boundary < 0:
        raise ProtocolError("process stat record is malformed")
    fields = payload[boundary + 2:].split()
    if len(fields) <= 36 or not fields[36].lstrip("-").isdigit():
        raise ProtocolError("process CPU record is malformed")
    return int(fields[36])


def _process_identity(pid: int) -> dict[str, int] | None:
    if type(pid) is not int or pid <= 1:
        raise ProtocolError("process PID is invalid")
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProtocolError("cannot inspect owned process identity") from exc
    boundary = payload.rfind(") ")
    if boundary < 0:
        raise ProtocolError("owned process stat record is malformed")
    fields = payload[boundary + 2:].split()
    indexes = (1, 2, 3, 19)
    if len(fields) <= max(indexes) or any(
        not fields[index].isascii() or not fields[index].lstrip("-").isdigit()
        for index in indexes
    ):
        raise ProtocolError("owned process stat fields are malformed")
    identity = {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "session": int(fields[3]),
        "start_time": int(fields[19]),
    }
    if identity["pgid"] < 0 or identity["session"] < 0 or \
            identity["start_time"] <= 0:
        raise ProtocolError("owned process identity is outside strict bounds")
    return identity


def _owned_process_document(
    path: Path, request: Mapping[str, Any], deployment_identity: tuple[int, int],
) -> tuple[dict[str, Any], os.stat_result]:
    payload, status = _read_bounded_with_status(path, 4096)
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("owned process record is malformed") from exc
    exact = _exact_mapping(document, {
        "schema", "owner_id", "deployment_root", "deployment_device",
        "deployment_inode", "run_id", "pid", "pgid", "session",
        "start_time",
    }, "owned process record")
    if exact["schema"] != OWNED_PROCESS_SCHEMA or \
            exact["owner_id"] != request["owner_id"] or \
            exact["deployment_root"] != request["deployment_root"] or \
            (exact["deployment_device"], exact["deployment_inode"]) != deployment_identity or \
            not isinstance(exact["run_id"], str) or \
            RUN_ID_RE.fullmatch(exact["run_id"]) is None or \
            exact["run_id"] != path.parent.name or \
            any(not _strict_int(exact[key], 1) for key in (
                "pid", "pgid", "session", "start_time",
            )) or exact["pid"] != exact["pgid"] or \
            exact["pid"] != exact["session"]:
        raise ProtocolError("owned process record does not match deployment ownership")
    return dict(exact), status


def _session_members(session_id: int) -> list[dict[str, int]]:
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise ProtocolError("cannot enumerate owned process session") from exc
    members: list[dict[str, int]] = []
    for entry in entries:
        if not entry.name.isascii() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid <= 1:
            continue
        identity = _process_identity(pid)
        if identity is not None and identity["session"] == session_id:
            members.append(identity)
    return members


def _signal_verified_process(identity: Mapping[str, int], selected_signal: int) -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ProtocolError("pidfd signaling is required for owned process cleanup")
    try:
        descriptor = os.pidfd_open(identity["pid"], 0)
    except ProcessLookupError:
        return
    try:
        current = _process_identity(identity["pid"])
        if current is None:
            return
        if any(current[key] != identity[key] for key in (
            "pid", "pgid", "session", "start_time",
        )):
            raise ProtocolError("owned process identity changed before signaling")
        try:
            signal.pidfd_send_signal(descriptor, selected_signal)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


def _quiesce_owned_process(
    path: Path, request: Mapping[str, Any], deployment_identity: tuple[int, int],
) -> None:
    record, record_status = _owned_process_document(
        path, request, deployment_identity,
    )
    leader = _process_identity(record["pid"])
    if leader is not None and any(
        leader[key] != record[key] for key in (
            "pid", "pgid", "session", "start_time",
        )
    ):
        raise ProtocolError("owned process leader identity changed")
    for selected_signal, timeout in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 1.0)):
        deadline = time.monotonic() + timeout
        while True:
            members = _session_members(record["session"])
            if not members:
                current = path.lstat()
                if (current.st_dev, current.st_ino) != (
                    record_status.st_dev, record_status.st_ino,
                ):
                    raise ProtocolError("owned process record identity changed")
                os.unlink(path)
                return
            reused_leader = next(
                (item for item in members if item["pid"] == record["pid"]), None
            )
            if reused_leader is not None and reused_leader["start_time"] != record["start_time"]:
                raise ProtocolError("owned process leader PID was reused")
            for member in members:
                _signal_verified_process(member, selected_signal)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
    raise ProtocolError("owned process tree did not become quiescent")


def _restore_owned_capture_trees(
    deployment: Path, request: Mapping[str, Any],
    deployment_identity: tuple[int, int],
) -> None:
    captures = deployment / "captures"
    try:
        capture_status = captures.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(capture_status.st_mode) or stat.S_ISLNK(capture_status.st_mode):
        raise ProtocolError("capture root identity is invalid during restore")
    for run_path in captures.iterdir():
        status = run_path.lstat()
        if RUN_ID_RE.fullmatch(run_path.name) is None or \
                not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise ProtocolError("capture run identity is invalid during restore")
        owner = run_path / OWNED_PROCESS_NAME
        try:
            owner.lstat()
        except FileNotFoundError:
            continue
        _quiesce_owned_process(owner, request, deployment_identity)


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1.0)


def _read_bounded_with_status(
    path: Path, maximum: int = MAX_CAPTURE_BYTES,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or \
                before.st_size < 0 or before.st_size > maximum:
            raise ProtocolError("capture is not one bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ProtocolError("capture was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProtocolError("capture grew during read")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size
        ):
            raise ProtocolError("capture identity changed during read")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _read_bounded(path: Path, maximum: int = MAX_CAPTURE_BYTES) -> bytes:
    return _read_bounded_with_status(path, maximum)[0]


def _capture(request: Any, context: HelperContext) -> dict[str, Any]:
    exact = _validate_common(
        request, {"run_id", "cpu", "argv", "environment", "timeout_ms"},
    )
    if exact["operation"] != "capture" or \
            exact["expected_board_identity"] != context.board_identity_digest or \
            not isinstance(exact["run_id"], str) or RUN_ID_RE.fullmatch(exact["run_id"]) is None or \
            type(exact["cpu"]) is not int or exact["cpu"] not in (0, 6) or \
            type(exact["timeout_ms"]) is not int or not 1000 <= exact["timeout_ms"] <= 120000:
        raise ProtocolError("capture request identity or bounds are invalid")
    environment = _exact_mapping(
        exact["environment"], {"E055_BUILD_NAME", "LANG", "LC_ALL"},
        "capture environment",
    )
    if dict(environment) != {"E055_BUILD_NAME": "O3", "LANG": "C", "LC_ALL": "C"}:
        raise ProtocolError("capture environment is not the non-secret allowlist")
    local_argv = _capture_argv(exact, context)
    lock, _ = _read_lock(context.local_path(LOCK_PATH), exact)
    deployment_identity = (
        lock["deployment_device"], lock["deployment_inode"],
    )
    run_remote = f"{exact['deployment_root']}/captures/{exact['run_id']}"
    run_path = context.local_path(run_remote)
    _ensure_safe_directory(run_path.parent)
    os.mkdir(run_path, 0o700)
    wrapper_stdout = run_path / "wrapper.stdout.raw"
    wrapper_stderr = run_path / "wrapper.stderr.raw"
    stdout_fd = os.open(
        wrapper_stdout, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600,
    )
    try:
        stderr_fd = os.open(
            wrapper_stderr, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600,
        )
    except BaseException:
        os.close(stdout_fd)
        raise
    process: subprocess.Popen[Any] | None = None
    owner_record = run_path / OWNED_PROCESS_NAME
    affinity_seen: set[int] = set()
    processor_samples: list[int] = []
    timed_out = False
    try:
        owner_template = _canonical_json({
            "schema": OWNED_PROCESS_SCHEMA,
            "owner_id": exact["owner_id"],
            "deployment_root": exact["deployment_root"],
            "deployment_device": deployment_identity[0],
            "deployment_inode": deployment_identity[1],
            "run_id": exact["run_id"],
        })
        gate_argv = (
            context.python_requested_path, "-c", _CAPTURE_GATE,
            owner_record.as_posix(), base64.b64encode(owner_template).decode("ascii"),
            *local_argv,
        )
        process = subprocess.Popen(
            gate_argv, stdin=subprocess.DEVNULL, stdout=stdout_fd, stderr=stderr_fd,
            env=dict(environment), close_fds=True, start_new_session=True, shell=False,
        )
        owner_deadline = time.monotonic() + 1.0
        while not owner_record.exists():
            if process.poll() is not None or time.monotonic() >= owner_deadline:
                raise ProtocolError("capture ownership was not persisted before execution")
            time.sleep(0.005)
        owned, _ = _owned_process_document(
            owner_record, exact, deployment_identity,
        )
        if owned["pid"] != process.pid:
            raise ProtocolError("capture ownership PID does not match child identity")
        deadline = time.monotonic() + exact["timeout_ms"] / 1000.0
        while process.poll() is None:
            for pid in _descendants(process.pid):
                try:
                    affinity = os.sched_getaffinity(pid)
                except ProcessLookupError:
                    continue
                if affinity == {exact["cpu"]}:
                    affinity_seen.update(affinity)
                    current = _processor(pid)
                    if current is not None:
                        processor_samples.append(current)
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_group(process)
                break
            time.sleep(0.005)
        if process.poll() is None:
            _terminate_group(process)
        status = process.wait(timeout=1.0)
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
        if process is not None and process.poll() is None:
            _terminate_group(process)
    if owner_record.exists():
        _quiesce_owned_process(owner_record, exact, deployment_identity)
    if timed_out:
        raise TimeoutError("bounded target capture timed out")
    output = run_path / "e049c.json"
    child_stdout = run_path / "target-harness.stdout.raw"
    child_stderr = run_path / "target-harness.stderr.raw"
    streams = {
        "child_stdout_base64": base64.b64encode(_read_bounded(child_stdout)).decode("ascii"),
        "child_stderr_base64": base64.b64encode(_read_bounded(child_stderr)).decode("ascii"),
        "e049c_json_base64": base64.b64encode(_read_bounded(output)).decode("ascii"),
        "wrapper_stdout_base64": base64.b64encode(_read_bounded(wrapper_stdout)).decode("ascii"),
        "wrapper_stderr_base64": base64.b64encode(_read_bounded(wrapper_stderr)).decode("ascii"),
    }
    exit_code = status if status >= 0 else 0
    exit_signal = -status if status < 0 else None
    migrations = sum(
        first != second for first, second in zip(processor_samples, processor_samples[1:])
    )
    return {
        "schema": "e055-capture-result/v1",
        "operation_sequence": exact["operation_sequence"],
        "request_id": exact["request_id"], "request_nonce": exact["request_nonce"],
        "run_id": exact["run_id"],
        "exit": {"code": exit_code, "signal": exit_signal}, "streams": streams,
        "affinity": {
            "effective_cpus": sorted(affinity_seen),
            "cpu_start": processor_samples[0] if processor_samples else exact["cpu"],
            "cpu_end": processor_samples[-1] if processor_samples else exact["cpu"],
            "migration_count": migrations,
        },
        "runtime": _runtime(context, exact),
    }


def _restore(request: Any, context: HelperContext) -> dict[str, Any]:
    exact = _validate_common(request, {"lock_path"})
    if exact["operation"] != "restore" or exact["lock_path"] != LOCK_PATH or \
            exact["expected_board_identity"] != context.board_identity_digest:
        raise ProtocolError("restore operation, path, or board identity is invalid")
    lock_path = context.local_path(exact["lock_path"])
    lock, lock_status = _read_lock(lock_path, exact)
    deployment = context.local_path(exact["deployment_root"])
    status = deployment.lstat()
    identity = (lock["deployment_device"], lock["deployment_inode"])
    if (status.st_dev, status.st_ino) != identity:
        raise ProtocolError("deployment identity changed before restore")
    _restore_owned_capture_trees(deployment, exact, identity)
    _remove_created_deployment(deployment, identity)
    current = lock_path.lstat()
    if (current.st_dev, current.st_ino) != (lock_status.st_dev, lock_status.st_ino):
        raise ProtocolError("lock identity changed before restore")
    os.unlink(lock_path)
    return {
        "schema": "e055-restore-result/v1",
        "operation_sequence": exact["operation_sequence"],
        "request_id": exact["request_id"], "request_nonce": exact["request_nonce"],
        "restored": True, "deployment_removed": True, "lock_released": True,
        "runtime": _runtime(context, exact),
    }


def _finalize_helper(request: Any, context: HelperContext) -> dict[str, Any]:
    exact = _validate_common(request, {"lock_path"})
    if exact["operation"] != "finalize_helper" or exact["lock_path"] != LOCK_PATH or \
            exact["expected_board_identity"] != context.board_identity_digest:
        raise ProtocolError("helper finalization request is invalid")
    if context.local_path(LOCK_PATH).exists() or context.local_path(exact["deployment_root"]).exists():
        raise ProtocolError("helper cannot be finalized before restoration")
    helper = context.local_path(context.helper_remote_path)
    digest, before = _sha256_file(helper, 1024 * 1024)
    if digest != PurePosixPath(context.helper_remote_path).parent.name:
        raise ProtocolError("helper bytes changed before finalization")
    current = helper.lstat()
    if (current.st_dev, current.st_ino, current.st_size, current.st_mode) != (
        before.st_dev, before.st_ino, before.st_size, before.st_mode,
    ) or not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode) or \
            current.st_nlink != 1:
        raise ProtocolError("helper identity changed before finalization unlink")
    os.unlink(helper)
    try:
        helper.parent.rmdir()
    except OSError:
        pass
    return {
        "schema": "e055-helper-finalization/v1",
        "operation_sequence": exact["operation_sequence"],
        "request_id": exact["request_id"], "request_nonce": exact["request_nonce"],
        "helper_removed": True, "helper_sha256": digest,
        "helper_device": before.st_dev, "helper_inode": before.st_ino,
    }


def handle_request(request: Any, context: HelperContext) -> dict[str, Any]:
    """Validate and execute one protocol operation without dynamic dispatch."""

    if not isinstance(request, dict):
        raise ProtocolError("request must be one JSON object")
    operation = request.get("operation")
    if operation == "exclusive_deploy":
        return _exclusive_deploy(request, context)
    if operation == "fresh_readback":
        return _fresh_readback(request, context)
    if operation == "capture":
        return _capture(request, context)
    if operation == "restore":
        return _restore(request, context)
    if operation == "finalize_helper":
        return _finalize_helper(request, context)
    raise ProtocolError("operation is not supported")


def main() -> int:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        return 64
    try:
        request = json.loads(payload.decode("ascii"))
        if _canonical_json(request) != payload:
            raise ProtocolError("request is not canonical JSON")
        response = handle_request(request, HelperContext.production())
        encoded = _canonical_json(response)
    except (OSError, ProtocolError, UnicodeDecodeError, ValueError, TypeError):
        return 65
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
