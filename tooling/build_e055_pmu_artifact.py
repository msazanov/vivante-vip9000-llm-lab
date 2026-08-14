#!/usr/bin/env python3
"""Build the publication-bound E049c AArch64 executable reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


COMPILER_NAME = "aarch64-linux-gnu-gcc"
SOURCE_RELATIVE = Path("tooling/a733_pmu_exec.c")
ARTIFACT_NAME = "a733-pmu-exec-aarch64"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pmu_artifact(
    root: Path, build_dir: Path, output_path: Path | None = None,
) -> dict[str, Any]:
    """Build E049c from one fixed object and return its exact provenance.

    The compiler never receives the destination publication path. Two builds in
    different clean directories therefore exercise the same compile/link graph.
    """

    root = root.resolve()
    build_dir = build_dir.resolve()
    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError("PMU build directory must be empty")
    build_dir.mkdir(parents=True, exist_ok=True)
    source = root / SOURCE_RELATIVE
    if not source.is_file():
        raise FileNotFoundError(source)
    compiler_value = shutil.which(COMPILER_NAME)
    if compiler_value is None:
        raise RuntimeError(f"required compiler is unavailable: {COMPILER_NAME}")
    compiler = Path(compiler_value).resolve()
    compiler_id_output = subprocess.run(
        [str(compiler), "--version"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.splitlines()
    if not compiler_id_output:
        raise RuntimeError("cross compiler returned an empty identity")
    compiler_id = compiler_id_output[0]
    object_path = build_dir / "a733_pmu_exec.o"
    binary_path = build_dir / ARTIFACT_NAME
    prefix_map = f"-ffile-prefix-map={root}=."
    debug_prefix_map = f"-fdebug-prefix-map={root}=."
    compile_argv = (
        str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-record-gcc-switches", prefix_map, debug_prefix_map,
        "-c", SOURCE_RELATIVE.as_posix(), "-o", str(object_path),
    )
    link_argv = (
        str(compiler), "-Wl,--build-id=none", str(object_path),
        "-o", str(binary_path),
    )
    subprocess.run(compile_argv, cwd=root, check=True)
    subprocess.run(link_argv, cwd=root, check=True)
    if not binary_path.is_file():
        raise RuntimeError("cross linker did not produce the PMU executable")
    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binary_path, output_path)
        output_path.chmod(0o755)
    return {
        "source_path": SOURCE_RELATIVE.as_posix(),
        "source_sha256": _sha256(source),
        "compiler_path": str(compiler),
        "compiler_sha256": _sha256(compiler),
        "compiler_id": compiler_id,
        "binary_path": str(binary_path),
        "binary_sha256": _sha256(binary_path),
        "compile_argv": list(compile_argv),
        "link_argv": list(link_argv),
    }


def reproducibility_report(root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Build twice in unrelated clean directories and bind the published bytes."""

    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="e055-pmu-build-a-") as first_temp, \
            tempfile.TemporaryDirectory(prefix="e055-pmu-build-b-") as second_temp:
        first = build_pmu_artifact(root, Path(first_temp))
        second = build_pmu_artifact(root, Path(second_temp))
    stable_fields = (
        "source_path", "source_sha256", "compiler_sha256", "compiler_id",
        "binary_sha256",
    )
    stable = all(first[field] == second[field] for field in stable_fields)
    report: dict[str, Any] = {
        "schema": "e055-pmu-build-review/v1",
        "status": "PASS" if stable else "FAIL",
        "source_path": first["source_path"],
        "source_sha256": first["source_sha256"],
        "compiler_sha256": first["compiler_sha256"],
        "compiler_id": first["compiler_id"],
        "binary_sha256": first["binary_sha256"],
        "independent_build_sha256": [
            first["binary_sha256"], second["binary_sha256"],
        ],
        "compile_flags": [
            "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fno-record-gcc-switches", "-ffile-prefix-map=<root>=.",
            "-fdebug-prefix-map=<root>=.", "-c",
        ],
        "link_flags": ["-Wl,--build-id=none"],
        "fixed_object_name": "a733_pmu_exec.o",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-output", type=Path)
    args = parser.parse_args()
    result = build_pmu_artifact(args.root, args.build_dir, args.output)
    if args.review_output is not None:
        result["reproducibility"] = reproducibility_report(
            args.root, args.review_output
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
