#!/usr/bin/env python3
"""Создать независимые adversarial fixtures для E019 INT16 DFP8 GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


SCHEMA = "vip9000-q1-i16-evis-gemm-fixture/v1"


class FixtureError(ValueError):
    pass


def pack_i16(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_cases(k: int) -> list[tuple[str, list[int]]]:
    return [
        ("all-negative", [-256] * k),
        ("all-positive", [256] * k),
        ("alternating-plus", [256 if index % 2 == 0 else -256 for index in range(k)]),
        ("alternating-minus", [-256 if index % 2 == 0 else 256 for index in range(k)]),
    ]


def generate(output_dir: Path, m: int, k: int, n: int) -> dict[str, object]:
    if (m, k, n) != (1, 32, 1):
        raise FixtureError("golden gate currently requires M=1, K=32, N=1")
    if output_dir.exists():
        raise FixtureError(f"output directory already exists: {output_dir}")

    activation = [-128, -127, -1, 0, 1, 126, 127, 64] * 4
    output_dir.mkdir(parents=True)
    manifest_cases: list[dict[str, object]] = []
    input_b = pack_i16(activation)

    for name, weights in build_cases(k):
        expected = sum((weight // 256) * value
                       for weight, value in zip(weights, activation, strict=True))
        if not -32768 <= expected <= 32767:
            raise FixtureError(f"expected dot overflows INT16 for {name}: {expected}")
        input_a = pack_i16(weights)
        expected_c = pack_i16([expected])
        case_dir = output_dir / name
        case_dir.mkdir()
        (case_dir / "input_a.i16.bin").write_bytes(input_a)
        (case_dir / "input_b.i16.bin").write_bytes(input_b)
        (case_dir / "expected_c.i16.bin").write_bytes(expected_c)
        manifest_cases.append(
            {
                "name": name,
                "expected_physical_i16": [expected],
                "files": {
                    "input_a": f"{name}/input_a.i16.bin",
                    "input_b": f"{name}/input_b.i16.bin",
                    "expected_c": f"{name}/expected_c.i16.bin",
                },
                "sha256": {
                    "input_a": sha256(input_a),
                    "input_b": sha256(input_b),
                    "expected_c": sha256(expected_c),
                },
            }
        )

    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "shape": {"m": m, "k": k, "n": n},
        "dfp_fixed_point_pos": 8,
        "semantics": {
            "input_a_physical": "-256/+256 represent exact -1/+1",
            "input_b_physical": "signed Q8 integer q represents q/256",
            "output_c_physical": "exact integer sum(sign*q)",
        },
        "cases": manifest_cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Создать E019 INT16 DFP8 golden fixtures")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        manifest = generate(args.output_dir, args.m, args.k, args.n)
    except (FixtureError, OSError, struct.error) as error:
        print(f"generate_q1_i16_evis_gemm_fixture: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir),
                      "cases": len(manifest["cases"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
