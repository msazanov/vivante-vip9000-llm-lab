#!/usr/bin/env python3
"""Generate exact packed-Q1/Q8 fixtures for E022."""

import argparse
import hashlib
import json
from pathlib import Path


PATTERNS = ("zeros", "ones", "alternating", "mixed", "random")


def _bit(row: int, col: int, pattern: str) -> int:
    if pattern == "zeros":
        return 0
    if pattern == "ones":
        return 1
    if pattern == "alternating":
        return (row + col) & 1
    if pattern == "mixed":
        return 1 if ((row * 11 + col * 7 + col // 3) % 13) < 6 else 0
    if pattern == "random":
        # Deterministic integer hash; no host RNG state is involved.
        x = (row + 1) * 0x9E3779B1 ^ (col + 7) * 0x85EBCA77
        x ^= x >> 16
        return (x >> 3) & 1
    raise ValueError(f"unknown pattern: {pattern}")


def build_fixture(m: int = 16, k: int = 128,
                  pattern: str = "mixed") -> dict:
    if m <= 0 or k <= 0 or k % 8:
        raise ValueError("M must be positive and K must be divisible by 8")
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")

    packed = bytearray(m * (k // 8))
    activation = bytearray(k)
    for col in range(k):
        # Includes zero, negative and positive values without risking INT8 dot overflow.
        activation[col] = (((col * 5 + 3) % 7) - 3) & 0xFF

    golden = []
    for row in range(m):
        row_start = row * (k // 8)
        total = 0
        selected = 0
        for col in range(k):
            bit = _bit(row, col, pattern)
            if bit:
                packed[row_start + (col >> 3)] |= 1 << (col & 7)
                selected += int(activation[col]) - (256 if activation[col] >= 128 else 0)
            total += int(activation[col]) - (256 if activation[col] >= 128 else 0)
        golden.append(2 * selected - total)

    metadata = {
        "schema": "vip9000-e022-fused-q1-fixture/v1",
        "shape": {"m": m, "k": k, "n": 1},
        "pattern": pattern,
        "bit_order": "lsb_first",
        "weight_mapping": "bit_0=-1,bit_1=+1",
        "activation_i8": [x if x < 128 else x - 256 for x in activation],
        "golden_i32": golden,
        "packed_sha256": hashlib.sha256(packed).hexdigest(),
    }
    return {
        "packed_q1": bytes(packed),
        "activation_i8": bytes(activation),
        "golden_i32": golden,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--pattern", choices=PATTERNS, default="mixed")
    args = parser.parse_args()

    fixture = build_fixture(args.m, args.k, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "packed_q1.bin").write_bytes(fixture["packed_q1"])
    (args.output_dir / "activation_i8.bin").write_bytes(fixture["activation_i8"])
    (args.output_dir / "golden_i32.json").write_text(
        json.dumps(fixture["golden_i32"], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fixture.json").write_text(
        json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
