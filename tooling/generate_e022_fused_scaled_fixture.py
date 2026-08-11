#!/usr/bin/env python3
"""Generate canonical E022 carriers with interleaved FP16 scales."""

import argparse
import json
import struct
from pathlib import Path

try:
    from tooling.generate_e022_fused_fixture import build_fixture
except ModuleNotFoundError:  # direct execution as tooling/<script>.py
    from generate_e022_fused_fixture import build_fixture


Q1_SCALES = (1.0, 0.5, 1.25, 0.75)
Q8_SCALES = (1.0, 0.5, 0.25, 1.5)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def build_scaled_fixture(m: int = 4, k: int = 5120,
                         pattern: str = "mixed") -> dict:
    if k < 128 or k % 128:
        raise ValueError("scaled carrier K must be >=128 and divisible by 128")
    logical = build_fixture(m, k, pattern)
    q1_blocks = k // 128
    q8_blocks = k // 32
    q1 = bytearray()
    for row in range(m):
        row_start = row * (k // 8)
        for block in range(q1_blocks):
            sign_start = row_start + block * 16
            q1.extend(logical["packed_q1"][sign_start:sign_start + 16])
            q1.extend(struct.pack("<e", Q1_SCALES[(row + block) % len(Q1_SCALES)]))

    q8 = bytearray()
    for block in range(q8_blocks):
        q8.extend(struct.pack("<e", Q8_SCALES[block % len(Q8_SCALES)]))
        start = block * 32
        q8.extend(logical["activation_i8"][start:start + 32])

    activation = [x if x < 128 else x - 256 for x in logical["activation_i8"]]
    golden = []
    for row in range(m):
        result = 0.0
        for block128 in range(q1_blocks):
            partial = 0.0
            q1_scale = Q1_SCALES[(row + block128) % len(Q1_SCALES)]
            for block32_in_128 in range(4):
                block32 = block128 * 4 + block32_in_128
                q8_scale = Q8_SCALES[block32 % len(Q8_SCALES)]
                selected = 0
                total = 0
                for i in range(32):
                    col = block32 * 32 + i
                    value = activation[col]
                    total += value
                    bit = (logical["packed_q1"][row * (k // 8) + (col >> 3)] >> (col & 7)) & 1
                    if bit:
                        selected += value
                signed_dot = 2 * selected - total
                partial = _f32(partial + _f32(float(signed_dot) * q8_scale))
            result = _f32(result + _f32(partial * q1_scale))
        golden.append(result)

    metadata = {
        **logical["metadata"],
        "schema": "vip9000-e022-fused-q1-scaled-carrier/v1",
        "scale_semantics": "fp16_le_q1_per_128_and_q8_per_32",
        "q1_scales": list(Q1_SCALES),
        "q8_scales": list(Q8_SCALES),
        "carrier": {
            "q1_superblock_values": 128,
            "q1_superblock_bytes": 18,
            "q1_sign_offset": 0,
            "q1_scale_offset": 16,
            "q1_superblocks_per_row": q1_blocks,
            "q8_block_values": 32,
            "q8_block_bytes": 34,
            "q8_value_offset": 2,
            "q8_blocks": q8_blocks,
        },
        "golden_f32": golden,
    }
    return {
        "weights_q1_carrier": bytes(q1),
        "activation_q8_carrier": bytes(q8),
        "golden_f32": golden,
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
    fixture = build_scaled_fixture(args.m, args.k, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weights_q1.bin").write_bytes(fixture["weights_q1_carrier"])
    (args.output_dir / "activation_q8.bin").write_bytes(fixture["activation_q8_carrier"])
    (args.output_dir / "golden_f32.json").write_text(
        json.dumps(fixture["golden_f32"], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fixture.json").write_text(
        json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
