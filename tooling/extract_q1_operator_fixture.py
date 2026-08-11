#!/usr/bin/env python3
"""Extract one pinned Bonsai Q1 tensor into a fixture outside the repository.

Containment checks detect and reject a destination rename observed at
publication boundaries. A continuous same-UID namespace adversary requires a
private mount or isolated namespace, which is outside this helper's threat
model.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import stat
import sys
from typing import Any


SCHEMA_VERSION = "q1-cpu-operator-fixture/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WEIGHTS_NAME = "weights.q1_0.bin"
_ACTIVATION_NAME = "activation.f32.bin"
_MANIFEST_NAME = "fixture.json"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUPPORTED_DIRFD_NAMES = frozenset(
    getattr(function, "__name__", "") for function in getattr(os, "supports_dir_fd", ())
)


class FixtureError(ValueError):
    """The source model, tensor metadata, or destination violates the contract."""


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        requirement = "positive integer" if positive else "non-negative integer"
        raise FixtureError(f"{context}: expected {requirement}")
    return value


def _sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FixtureError(f"{context}: expected lowercase 64-character SHA-256")
    return value


def _validate_tensor(tensor: Mapping[str, object]) -> tuple[str, int, tuple[int, int], int, int]:
    required = {
        "name", "index", "ggml_type", "shape", "source_offset_bytes", "size_bytes"
    }
    missing = sorted(required - set(tensor))
    if missing:
        raise FixtureError(f"tensor: missing fields {', '.join(missing)}")
    name = tensor["name"]
    if type(name) is not str or not name:
        raise FixtureError("tensor.name: expected non-empty string")
    index = _integer(tensor["index"], "tensor.index")
    if tensor["ggml_type"] != "Q1_0":
        raise FixtureError("tensor.ggml_type: expected Q1_0")
    shape_value = tensor["shape"]
    if isinstance(shape_value, (str, bytes)) or not isinstance(shape_value, Sequence):
        raise FixtureError("tensor.shape: expected a two-dimensional sequence")
    if len(shape_value) != 2:
        raise FixtureError("tensor.shape: expected exactly two dimensions")
    ne0 = _integer(shape_value[0], "tensor.shape[0]", positive=True)
    ne1 = _integer(shape_value[1], "tensor.shape[1]", positive=True)
    if ne0 % 128 != 0:
        raise FixtureError("tensor.shape[0]: Q1_0 dimension must be divisible by 128")
    if ne1 % 16 != 0:
        raise FixtureError("tensor.shape[1]: Q1_0 dimension must be divisible by 16")
    source_offset = _integer(tensor["source_offset_bytes"], "tensor.source_offset_bytes")
    size_bytes = _integer(tensor["size_bytes"], "tensor.size_bytes", positive=True)
    expected_size = ne0 * ne1 // 128 * 18
    if size_bytes != expected_size:
        raise FixtureError(
            f"tensor.size_bytes={size_bytes} does not match Q1_0 shape size {expected_size}"
        )
    if "payload_sha256" not in tensor:
        raise FixtureError("tensor: missing field payload_sha256")
    _sha256(tensor["payload_sha256"], "tensor.payload_sha256")
    return name, index, (ne0, ne1), source_offset, size_bytes


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _stream_hash_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _model_and_payload(model: Path, offset: int, size: int, expected_sha256: str) -> tuple[int, str, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(model, flags)
    except OSError as exc:
        raise FixtureError(f"cannot open model: {model}") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise FixtureError("model must be a regular file")
        initial_signature = _stat_signature(initial)
        first_sha256 = _stream_hash_fd(descriptor)
        between_signature = _stat_signature(os.fstat(descriptor))
        if between_signature != initial_signature:
            raise FixtureError("model metadata changed during initial hash")
        payload = os.pread(descriptor, size, offset)
        if len(payload) != size:
            raise FixtureError(
                f"tensor payload truncated: pread returned {len(payload)} bytes, expected {size}"
            )
        after_pread_signature = _stat_signature(os.fstat(descriptor))
        if after_pread_signature != initial_signature:
            raise FixtureError("model metadata changed around payload read")
        second_sha256 = _stream_hash_fd(descriptor)
        final_signature = _stat_signature(os.fstat(descriptor))
        if final_signature != initial_signature:
            raise FixtureError("model metadata changed during verification hash")
        if first_sha256 != expected_sha256 or second_sha256 != expected_sha256 or first_sha256 != second_sha256:
            raise FixtureError("model SHA-256 changed during extraction")
        return initial.st_size, second_sha256, payload
    except OSError as exc:
        raise FixtureError("model read failed") from exc
    finally:
        os.close(descriptor)


def _xorshift32(state: int) -> int:
    state &= 0xFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _activation(ne0: int, seed: int) -> bytes:
    seed = _integer(seed, "seed")
    state = seed & 0xFFFFFFFF
    output = bytearray()
    for _ in range(ne0):
        state = _xorshift32(state)
        # The integer numerator and denominator are code-owned; Python's random
        # module and its implementation-specific stream are deliberately absent.
        value = (2.0 * state / 0xFFFFFFFF) - 1.0
        output.extend(struct.pack("<f", value))
    return bytes(output)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _exclusive_write_at(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
    except OSError as exc:
        raise FixtureError(f"refusing to overwrite existing output: {name}") from exc
    complete = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FixtureError(f"short write: {name}")
            view = view[written:]
        os.fsync(descriptor)
        complete = True
    except OSError as exc:
        raise FixtureError(f"cannot write fixture file: {name}") from exc
    except FixtureError:
        raise
    finally:
        os.close(descriptor)
        if not complete:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _require_linux_dirfd() -> None:
    if os.name != "posix" or not {"open", "mkdir", "unlink", "rmdir"}.issubset(_SUPPORTED_DIRFD_NAMES):
        raise FixtureError("race-safe fixture output requires Linux dir_fd primitives")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise FixtureError("race-safe fixture output requires O_DIRECTORY and O_NOFOLLOW")
    if not os.path.isdir("/proc/self/fd"):
        raise FixtureError("race-safe fixture output requires /proc/self/fd")


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise FixtureError(f"cannot open output path component: {name}") from exc
    return descriptor


@dataclass
class _Destination:
    parent_fd: int
    output_fd: int
    final_name: str
    root_fd: int
    opened: list[tuple[int, str, int, bool]]
    created_files: list[str]

    def validate_external(self) -> None:
        """Re-check the stable fd target immediately around publication."""

        stable_target = os.readlink(f"/proc/self/fd/{self.output_fd}")
        if stable_target.endswith(" (deleted)"):
            raise FixtureError("output directory was deleted during publication")
        stable_path = Path(stable_target).resolve()
        if _under(stable_path, _REPO_ROOT.resolve()):
            raise FixtureError("output directory moved inside the repository")

    def final_entry_matches_fd(self) -> bool:
        try:
            expected = os.fstat(self.output_fd)
            actual = os.stat(self.final_name, dir_fd=self.parent_fd, follow_symlinks=False)
        except OSError:
            return False
        return stat.S_ISDIR(actual.st_mode) and _stat_signature(expected)[:2] == _stat_signature(actual)[:2]

    def cleanup(self) -> None:
        for name in reversed(self.created_files):
            try:
                os.unlink(name, dir_fd=self.output_fd)
            except FileNotFoundError:
                pass
        if self.final_entry_matches_fd():
            try:
                os.rmdir(self.final_name, dir_fd=self.parent_fd)
            except OSError:
                pass
        for parent_fd, name, descriptor, created in reversed(self.opened):
            if not created:
                continue
            try:
                expected = os.fstat(descriptor)
                actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISDIR(actual.st_mode) and _stat_signature(expected)[:2] == _stat_signature(actual)[:2]:
                    os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass

    def close(self) -> None:
        os.close(self.output_fd)
        for _, _, descriptor, _ in reversed(self.opened):
            os.close(descriptor)
        os.close(self.root_fd)


def _stable_destination(output_dir: Path) -> _Destination:
    _require_linux_dirfd()
    absolute = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    components = Path(absolute).parts
    if len(components) < 2 or components[0] != "/" or any(part in ("", ".", "..") for part in components[1:]):
        raise FixtureError("output directory must be an absolute normal path")
    root_fd = os.open("/", _directory_flags())
    opened: list[tuple[int, str, int, bool]] = []
    output_fd: int | None = None
    parent_fd = root_fd
    try:
        for name in components[1:-1]:
            created = False
            try:
                descriptor = _open_directory_at(parent_fd, name)
            except FixtureError as exc:
                cause = exc.__cause__
                if not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
                    raise
                try:
                    os.mkdir(name, 0o755, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    pass
                descriptor = _open_directory_at(parent_fd, name)
            opened.append((parent_fd, name, descriptor, created))
            parent_fd = descriptor
        final_name = components[-1]
        try:
            os.mkdir(final_name, 0o755, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise FixtureError("output directory already exists") from exc
        output_fd = _open_directory_at(parent_fd, final_name)
        stable_target = os.readlink(f"/proc/self/fd/{output_fd}")
        if stable_target.endswith(" (deleted)"):
            raise FixtureError("output directory was deleted during creation")
        stable_path = Path(stable_target).resolve()
        if _under(stable_path, _REPO_ROOT.resolve()):
            raise FixtureError("output directory must be outside the repository")
        return _Destination(parent_fd, output_fd, final_name, root_fd, opened, [])
    except (FixtureError, OSError):
        if output_fd is not None:
            try:
                output_stat = os.fstat(output_fd)
                final_stat = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISDIR(final_stat.st_mode) and _stat_signature(output_stat)[:2] == _stat_signature(final_stat)[:2]:
                    os.rmdir(components[-1], dir_fd=parent_fd)
            except OSError:
                pass
            os.close(output_fd)
        # If the final fd was never opened, leave the empty directory in place:
        # there is no identity proof that its name still refers to our inode.
        for parent_fd_item, name, descriptor, created in reversed(opened):
            if created:
                try:
                    os.rmdir(name, dir_fd=parent_fd_item)
                except OSError:
                    pass
        for _, _, descriptor, _ in reversed(opened):
            os.close(descriptor)
        os.close(root_fd)
        raise


def extract_tensor(
    model: Path,
    tensor: Mapping[str, object],
    output_dir: Path,
    *,
    expected_model_sha256: str,
    seed: int = 0x733,
) -> dict[str, object]:
    """Extract one Q1_0 payload and its deterministic F32 activation.

    The source slice is obtained with one exact ``pread`` call. All destination
    files use exclusive creation, and a resolved destination under this repo is
    rejected so model-derived bytes cannot accidentally enter Git.
    """

    if not isinstance(model, Path):
        model = Path(model)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    expected_model_sha256 = _sha256(expected_model_sha256, "expected_model_sha256")
    if not isinstance(tensor, Mapping):
        raise FixtureError("tensor: expected a mapping")
    name, index, (ne0, ne1), source_offset, size_bytes = _validate_tensor(tensor)
    model_size, model_sha256, weights = _model_and_payload(
        model, source_offset, size_bytes, expected_model_sha256
    )
    weights_sha256 = hashlib.sha256(weights).hexdigest()
    if tensor["payload_sha256"] != weights_sha256:
        raise FixtureError("tensor.payload_sha256 does not match exact pread payload")
    activation = _activation(ne0, seed)
    activation_sha256 = hashlib.sha256(activation).hexdigest()

    destination = _stable_destination(output_dir)
    try:
        def publish(name: str, payload: bytes) -> None:
            destination.validate_external()
            try:
                _exclusive_write_at(destination.output_fd, name, payload)
                destination.created_files.append(name)
            finally:
                destination.validate_external()

        publish(_WEIGHTS_NAME, weights)
        publish(_ACTIVATION_NAME, activation)
        manifest: dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "model": {
                "filename": model.name,
                "sha256": model_sha256,
                "size_bytes": model_size,
            },
            "tensor": {
                "file": _WEIGHTS_NAME,
                "name": name,
                "index": index,
                "ggml_type": "Q1_0",
                "shape": [ne0, ne1],
                "source_offset_bytes": source_offset,
                "size_bytes": size_bytes,
                "sha256": weights_sha256,
            },
            "activation": {
                "file": _ACTIVATION_NAME,
                "dtype": "F32",
                "shape": [ne0],
                "seed": seed,
                "prng": "xorshift32",
                "transform": "2.0 * state / 0xffffffff - 1.0",
                "size_bytes": len(activation),
                "sha256": activation_sha256,
            },
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        publish(_MANIFEST_NAME, manifest_bytes)
        destination.validate_external()
        return manifest
    except (FixtureError, OSError):
        destination.cleanup()
        raise
    finally:
        destination.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0x733)
    args = parser.parse_args(argv)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tooling.q1_memory_accounting import Q1MemoryAccountingError, validate_bonsai_manifest

    try:
        workload = json.loads(args.workload.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise FixtureError("workload must be valid UTF-8") from exc
    try:
        validate_bonsai_manifest(workload)
    except Q1MemoryAccountingError as exc:
        raise FixtureError(f"workload validation failed: {exc}") from exc
    tensors = [tensor for tensor in workload["tensors"] if tensor.get("name") == args.tensor]
    if len(tensors) != 1:
        raise FixtureError(f"workload must contain exactly one tensor named {args.tensor}")
    manifest = extract_tensor(
        args.model,
        tensors[0],
        args.output_dir,
        expected_model_sha256=workload["model"]["sha256"],
        seed=args.seed,
    )
    print(f"fixture={args.output_dir}")
    print(f"tensor={manifest['tensor']['name']}")
    print(f"weights_bytes={manifest['tensor']['size_bytes']}")
    print(f"activation_bytes={manifest['activation']['size_bytes']}")
    print(f"model_sha256={manifest['model']['sha256']}")
    print(f"weights_sha256={manifest['tensor']['sha256']}")
    print(f"activation_sha256={manifest['activation']['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (FixtureError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
