#!/usr/bin/env python3
"""Generate the E022 packed carriers for a long K=128*n projection."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tooling.generate_e022_fused_fixture import build_fixture
except ModuleNotFoundError:  # direct execution as tooling/<script>.py
    from generate_e022_fused_fixture import build_fixture


def build_long_carrier_fixture(m: int = 4, k: int = 5120,
                               pattern: str = "mixed") -> dict:
    if k < 128 or k % 128:
        raise ValueError("long carrier K must be >=128 and divisible by 128")
    logical = build_fixture(m, k, pattern)
    q1 = bytearray()
    q1_sign_bytes = k // 8
    for row in range(m):
        row_start = row * q1_sign_bytes
        for block in range(k // 128):
            sign_start = row_start + block * 16
            q1.extend(logical["packed_q1"][sign_start:sign_start + 16])
            q1.extend(b"\0\0")

    q8 = bytearray()
    for block in range(k // 32):
        q8.extend(b"\0\0")
        start = block * 32
        q8.extend(logical["activation_i8"][start:start + 32])

    metadata = {
        **logical["metadata"],
        "schema": "vip9000-e022-fused-q1-long-carrier/v1",
        "carrier": {
            "q1_superblock_values": 128,
            "q1_superblock_bytes": 18,
            "q1_sign_offset": 0,
            "q1_scale_offset": 16,
            "q1_superblocks_per_row": k // 128,
            "q8_block_values": 32,
            "q8_block_bytes": 34,
            "q8_value_offset": 2,
            "q8_blocks": k // 32,
            "q1_carrier_sha256": hashlib.sha256(q1).hexdigest(),
            "q8_carrier_sha256": hashlib.sha256(q8).hexdigest(),
        },
    }
    return {
        "weights_q1_carrier": bytes(q1),
        "activation_q8_carrier": bytes(q8),
        "golden_i32": logical["golden_i32"],
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--pattern", choices=("zeros", "ones", "alternating", "mixed", "random"),
                        default="mixed")
    args = parser.parse_args()
    fixture = build_long_carrier_fixture(args.m, args.k, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weights_q1.bin").write_bytes(fixture["weights_q1_carrier"])
    (args.output_dir / "activation_q8.bin").write_bytes(fixture["activation_q8_carrier"])
    (args.output_dir / "golden_i32.json").write_text(
        json.dumps(fixture["golden_i32"], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fixture.json").write_text(
        json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
