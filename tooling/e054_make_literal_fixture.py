#!/usr/bin/env python3
"""Создать маленький hand-derived Q1 fixture для host golden E054."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


EXPECTED = [
    -56.0, -48.0, -40.0, -32.0, -24.0, -16.0, -8.0, 0.0,
    8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0,
]


def write_fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    q1 = bytearray()
    for row in range(16):
        q1 += struct.pack("<H", 0x3800)  # FP16 0.5
        q1 += bytes([0xFF] * (row + 1))
        q1 += bytes([0x00] * (16 - row - 1))
    activation = struct.pack("<128f", *([1.0] * 128))
    manifest: dict[str, object] = {
        "schema": "q1-cpu-operator-fixture/v1",
        "model": {
            "filename": "Bonsai-27B-Q1_0.gguf",
            "sha256": "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
            "size_bytes": 3803452480,
        },
        "tensor": {
            "file": "weights.q1_0.bin",
            "ggml_type": "Q1_0",
            "name": "literal.q1",
            "sha256": hashlib.sha256(q1).hexdigest(),
            "shape": [128, 16],
            "size_bytes": len(q1),
        },
        "activation": {
            "dtype": "F32",
            "file": "activation.f32.bin",
            "sha256": hashlib.sha256(activation).hexdigest(),
            "shape": [128],
            "size_bytes": len(activation),
        },
    }
    (root / "weights.q1_0.bin").write_bytes(q1)
    (root / "activation.f32.bin").write_bytes(activation)
    (root / "fixture.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "expected_output_f32": EXPECTED}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_fixture(args.output_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
