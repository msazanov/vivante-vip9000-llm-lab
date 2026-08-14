#!/usr/bin/env python3
"""Вынести бинарные capture E054 из публикуемого raw-manifest.

Файлы остаются локально и идентифицируются публичными SHA-256 и размером.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "e054-external-sensitive-artifacts/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def externalize(experiment: Path) -> dict:
    raw = experiment / "raw"
    raw_manifest = experiment / "data" / "raw-manifest.sha256"
    output = experiment / "data" / "external-sensitive-artifacts.json"
    captures = sorted(raw.rglob("*.bin"))
    if len(captures) != 46:
        raise ValueError(f"ожидалось 46 binary capture, найдено {len(captures)}")

    entries = []
    for capture in captures:
        relative = capture.relative_to(experiment).as_posix()
        synthetic = "/literal-" in f"/{relative}"
        entries.append(
            {
                "path": relative,
                "sha256": sha256(capture),
                "size_bytes": capture.stat().st_size,
                "origin": "synthetic_fixture" if synthetic else "real_model_derived",
                "payload_kind": (
                    "q8_activation" if capture.name == "activation.q8_0.bin"
                    else "f32_output"
                ),
                "published": False,
                "retention": "local_only",
            }
        )

    if sum(item["origin"] == "real_model_derived" for item in entries) != 40:
        raise ValueError("неожиданное число real-model-derived capture")
    if sum(item["origin"] == "synthetic_fixture" for item in entries) != 6:
        raise ValueError("неожиданное число synthetic capture")

    document = {
        "schema_version": SCHEMA,
        "publication_policy": (
            "Бинарные входы и выходы не публикуются; локальные payload "
            "сверяются по SHA-256 и размеру."
        ),
        "external_sensitive_artifacts": entries,
    }
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    retained = [
        line for line in raw_manifest.read_text(encoding="utf-8").splitlines()
        if not line.rstrip().endswith(".bin")
    ]
    raw_manifest.write_text("\n".join(retained) + "\n", encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    document = externalize(args.experiment.resolve())
    print(json.dumps({
        "externalized": len(document["external_sensitive_artifacts"]),
        "manifest": str(
            args.experiment / "data" / "external-sensitive-artifacts.json"
        ),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
