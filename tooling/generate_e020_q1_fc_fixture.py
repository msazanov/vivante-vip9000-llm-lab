#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


M = 16
K = 32


def signed_byte(value: int) -> int:
    return value & 0xFF


def build_fixture(m: int = M, k: int = K,
                  pattern: str = "mixed") -> tuple[bytes, bytes, bytes, dict]:
    if m <= 0 or k < 32 or k % 32:
        raise ValueError("M must be positive and K must be divisible by 32")
    stride = max(1, k // 64)
    activation_values = [
        ((index * 5 + 3) % 3) - 1 if index % stride == 0 else 0
        for index in range(k)
    ]
    packed = bytearray()
    golden = []
    row_descriptions = []
    for row in range(m):
        signs = []
        row_bytes = bytearray(k // 8)
        for column in range(k):
            if pattern == "zeros":
                bit = False
            elif pattern == "ones":
                bit = True
            elif pattern == "alternating":
                bit = column % 2 == row % 2
            elif pattern == "mixed":
                bit = ((row * 11 + column * 7 + column // 3) % 13) < 6
            else:
                raise ValueError(f"unknown pattern: {pattern}")
            signs.append(1 if bit else -1)
            if bit:
                row_bytes[column // 8] |= 1 << (column % 8)
        dot = sum(sign * value for sign, value in zip(signs, activation_values))
        if not -128 <= dot <= 127:
            raise ValueError(f"golden output overflows INT8: row={row}, dot={dot}")
        packed.extend(row_bytes)
        golden.append(dot)
        row_descriptions.append({"row": row, "packed_hex": row_bytes.hex(), "dot": dot})
    activation = bytes(signed_byte(value) for value in activation_values)
    output = bytes(signed_byte(value) for value in golden)
    metadata = {
        "schema": "vip9000-e020-q1-unpack-native-fc-fixture/v1",
        "shape": {"m": m, "k": k, "n": 1},
        "bit_order": "lsb_first",
        "weight_mapping": "bit_0=-1,bit_1=+1",
        "pattern": pattern,
        "activation_i8": activation_values,
        "golden_i8": golden,
        "rows": row_descriptions,
    }
    return bytes(packed), activation, output, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--pattern", choices=("zeros", "ones", "alternating", "mixed"),
                        default="mixed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packed, activation, golden, metadata = build_fixture(
        args.m, args.k, args.pattern)
    (args.output_dir / "packed_q1.bin").write_bytes(packed)
    (args.output_dir / "activation_i8.bin").write_bytes(activation)
    (args.output_dir / "golden_output_i8.bin").write_bytes(golden)
    (args.output_dir / "fixture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
