#!/usr/bin/env python3
"""Wrap the compact E022 Q1/Q8 logical fixture in byte carriers for VIPLite."""

import argparse
import json
from pathlib import Path

try:
    from tooling.generate_e022_fused_fixture import build_fixture
except ModuleNotFoundError:  # direct execution as tooling/<script>.py
    from generate_e022_fused_fixture import build_fixture


def build_carrier_fixture(m: int = 4, k: int = 128, pattern: str = "mixed") -> dict:
    if k != 128:
        raise ValueError("the first target carrier gate is fixed at K=128")
    logical = build_fixture(m, k, pattern)
    q1 = bytearray()
    for row in range(m):
        start = row * (k // 8)
        q1.extend(logical["packed_q1"][start:start + 16])
        q1.extend(b"\0\0")

    q8 = bytearray()
    for block in range(4):
        q8.extend(b"\0\0")
        start = block * 32
        q8.extend(logical["activation_i8"][start:start + 32])

    return {
        "weights_q1_carrier": bytes(q1),
        "activation_q8_carrier": bytes(q8),
        "golden_i32": logical["golden_i32"],
        "metadata": {
            **logical["metadata"],
            "schema": "vip9000-e022-fused-q1-carrier/v1",
            "carrier": {
                "q1_row_bytes": 18,
                "q1_sign_offset": 0,
                "q1_scale_offset": 16,
                "q8_block_bytes": 34,
                "q8_value_offset": 2,
                "q8_blocks": 4,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--pattern", choices=("zeros", "ones", "alternating", "mixed", "random"), default="mixed")
    args = parser.parse_args()
    fixture = build_carrier_fixture(args.m, args.k, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weights_q1.bin").write_bytes(fixture["weights_q1_carrier"])
    (args.output_dir / "activation_q8.bin").write_bytes(fixture["activation_q8_carrier"])
    (args.output_dir / "golden_i32.json").write_text(
        json.dumps(fixture["golden_i32"], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fixture.json").write_text(
        json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
