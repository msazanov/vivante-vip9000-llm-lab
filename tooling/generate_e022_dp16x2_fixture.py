#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_vectors(case: str):
    if case == "base":
        a_hi = [-128, -127, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        a_lo = [12, -11, 10, -9, 8, -7, 6, -5, 4, -3, 2, -1, 0, 1, -2, 3]
        b = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8]
    elif case == "selector_all_zero":
        a_hi = [1] * 16
        a_lo = [-1] * 16
        b = list(range(-8, 8))
    elif case == "selector_all_five":
        a_hi = list(range(-8, 8))
        a_lo = list(range(8, -8, -1))
        b = [127, -128, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
    elif case == "reverse_split":
        a_hi = [((i * 13 + 5) % 31) - 15 for i in range(16)]
        a_lo = list(reversed(a_hi))
        b = [((i * 17 + 3) % 29) - 14 for i in range(16)]
    else:
        raise ValueError(case)
    return a_hi, a_lo, b


def build_fixture(case: str) -> dict:
    a_hi, a_lo, b = build_vectors(case)
    golden = [sum(x * y for x, y in zip(a_hi, b)),
              sum(x * y for x, y in zip(a_lo, b))]
    return {
        "a_hi": bytes(x & 0xFF for x in a_hi),
        "a_lo": bytes(x & 0xFF for x in a_lo),
        "b": bytes(x & 0xFF for x in b),
        "golden": golden,
        "metadata": {
            "schema": "vip9000-e022-dp16x2-fixture/v1",
            "case": case,
            "a_hi": a_hi,
            "a_lo": a_lo,
            "b": b,
            "golden": golden,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", required=True,
                        choices=("base", "selector_all_zero", "selector_all_five", "reverse_split"))
    args = parser.parse_args()
    fixture = build_fixture(args.case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "a_hi.bin").write_bytes(fixture["a_hi"])
    (args.output_dir / "a_lo.bin").write_bytes(fixture["a_lo"])
    (args.output_dir / "b.bin").write_bytes(fixture["b"])
    (args.output_dir / "golden.json").write_text(
        json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
