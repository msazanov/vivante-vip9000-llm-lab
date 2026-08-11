#!/usr/bin/env python3
"""Generate the deterministic public Q1 VIP C0 fixture bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

# Keep the real CLI usable when invoked by absolute path from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tooling.q1_vip_golden import dot_packed_tile
from tooling.q1_vip_layout import pack_tensor


OUTPUT_FILES = (
    "q1_canonical.bin",
    "q1_vip.bin",
    "q8.bin",
    "expected_f32.bin",
    "manifest.json",
)
Q1_PATTERNS = (
    b"\xff" * 16,
    b"\x00" * 16,
    b"\x55" * 16,
    b"\xaa" * 16,
    bytes.fromhex("0102040810204080") * 2,
)
Q1_SCALE_VALUES = (1.0, 2.0, 0.5, -1.0)
Q8_VALUE_PATTERNS = (
    (-128,) * 32,
    (127,) * 32,
    (0,) * 32,
    (127, -128) * 16,
)


def _q1_canonical() -> bytes:
    rows = []
    for row in range(16):
        scale = struct.pack("<e", Q1_SCALE_VALUES[row % len(Q1_SCALE_VALUES)])
        rows.append(scale + Q1_PATTERNS[row % len(Q1_PATTERNS)])
    return b"".join(rows)


def _q8_vector() -> bytes:
    blocks = []
    for subblock, values in enumerate(Q8_VALUE_PATTERNS):
        scale = struct.pack("<e", Q1_SCALE_VALUES[subblock])
        blocks.append(scale + bytes(value & 0xFF for value in values))
    return b"".join(blocks)


def _repository_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("git HEAD имеет неожиданный формат")
    return commit


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(payloads: dict[str, bytes], repository_commit: str) -> bytes:
    files = {
        name: {"sha256": _sha256(payloads[name]), "size_bytes": len(payloads[name])}
        for name in payloads
    }
    manifest = {
        "expected_output_count": 16,
        "files": files,
        "generator_repository_commit": repository_commit,
        "layout": "Q1_VIP_16x128/v1",
        "schema": "q1-vip-c0-fixture/v1",
        "shape": {"K": 128, "M": 16},
        "types": {"expected": "FP32", "q1": "Q1_0", "q8": "Q8_0"},
    }
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def generate(output_dir: Path, repo_root: Path) -> None:
    """Create one new fixture directory and never overwrite an existing path."""
    canonical = _q1_canonical()
    q8 = _q8_vector()
    vip = pack_tensor(canonical, 128, 16)
    expected = struct.pack("<16f", *dot_packed_tile(vip, q8))
    payloads = {
        "q1_canonical.bin": canonical,
        "q1_vip.bin": vip,
        "q8.bin": q8,
        "expected_f32.bin": expected,
    }
    payloads["manifest.json"] = _manifest(payloads, _repository_commit(repo_root))

    output_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
    for name in OUTPUT_FILES:
        _exclusive_write(output_dir / name, payloads[name])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Создать детерминированный Q1 VIP C0 fixture")
    parser.add_argument("--output-dir", type=Path, required=True, help="новый каталог fixture")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        generate(args.output_dir, repo_root)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"ошибка генерации fixture: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
