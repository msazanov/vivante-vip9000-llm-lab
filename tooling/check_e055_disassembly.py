#!/usr/bin/env python3
"""Fail closed if optimized E055 controls regain DOTPROD or inner calls."""

from __future__ import annotations

import argparse
import json
import re
import subprocess


SYMBOLS = ("native_simd_group", "packed_stream_only", "unpack_scale_only")


def sections(disassembly: str) -> dict[str, str]:
    # Demangled C++ template names contain embedded '>' characters, so the
    # symbol capture must be greedy up to the final ``>:\n`` delimiter.
    headers = list(re.finditer(r"(?m)^[0-9a-f]+ <(.+)>:\s*$", disassembly))
    result: dict[str, str] = {}
    for index, header in enumerate(headers):
        name = header.group(1)
        end = headers[index + 1].start() if index + 1 < len(headers) else len(disassembly)
        for wanted in SYMBOLS:
            if wanted in name and wanted not in result:
                result[wanted] = disassembly[header.start():end]
    return result


def inspect(binary: str, objdump: str) -> dict:
    text = subprocess.run([objdump, "-d", "-C", binary], check=True,
                          text=True, stdout=subprocess.PIPE).stdout
    found = sections(text)
    failures: list[str] = []
    for name in SYMBOLS:
        if name not in found:
            failures.append(f"missing symbol: {name}")
    for control in ("packed_stream_only", "unpack_scale_only"):
        body = found.get(control, "")
        if re.search(r"\bsdot\b", body):
            failures.append(f"{control} unexpectedly contains SDOT")
        if re.search(r"\bbl\s", body):
            failures.append(f"{control} unexpectedly contains a function call")
        if not re.search(r"\b(?:ldr|ldp|ld1)\b", body):
            failures.append(f"{control} has no retained load instruction")
        if not re.search(r"\b(?:eor|addv)\b", body):
            failures.append(f"{control} has no retained vector consume instruction")
    if not re.search(r"\bsdot\b", found.get("native_simd_group", "")):
        failures.append("native_simd_group lost stock SDOT")
    return {
        "schema": "e055-disassembly-review/v1",
        "binary": binary,
        "objdump": objdump,
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "controls_retain_loads": not any("no retained load" in f for f in failures),
            "controls_have_bounded_vector_consume": not any("consume" in f for f in failures),
            "controls_have_no_sdot": not any("unexpectedly contains SDOT" in f for f in failures),
            "controls_have_no_calls": not any("function call" in f for f in failures),
            "stock_full_kernel_has_sdot": "native_simd_group lost stock SDOT" not in failures,
        },
        "failures": failures,
        "limitation": "Static inspection proves instruction shape, not target timing or cache behavior.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--objdump", default="aarch64-linux-gnu-objdump")
    args = parser.parse_args()
    report = inspect(args.binary, args.objdump)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
