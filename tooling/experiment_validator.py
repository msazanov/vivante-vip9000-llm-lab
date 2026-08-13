#!/usr/bin/env python3
"""Validate the evidence layout of one E053 experiment.

The validator is deliberately independent of a model runtime.  It checks the
paper trail around an experiment: Russian README, hypothesis preflight,
provenance, an explicit lifecycle status, and hashes for raw artifacts.  A
``planned`` experiment may exist before any model is run; every other status
must carry the unmodified stdout/stderr, telemetry and trace streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "e053-experiment-manifest/v1"
SUMMARY_SCHEMA = "e053-experiment-summary/v1"
VALID_STATUSES = {"planned", "running", "passed", "failed", "rejected", "blocked"}
EXECUTED_STATUSES = VALID_STATUSES - {"planned"}
REQUIRED_FILES = (
    "README.md",
    "hypothesis-preflight.md",
    "commands.txt",
    "environment.json",
    "device.json",
    "data/manifest.json",
    "results/summary.json",
)
REQUIRED_RAW_FILES = (
    "raw/stdout.log",
    "raw/stderr.log",
    "raw/telemetry.jsonl",
    "raw/trace.jsonl",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: не удалось прочитать JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: корневое значение должно быть JSON-объектом")
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        return None
    return path.as_posix()


def _validate_manifest(root: Path, manifest: dict[str, Any], status: str, errors: list[str]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"data/manifest.json: нужен schema_version {MANIFEST_SCHEMA}")
    if manifest.get("status") != status:
        errors.append("data/manifest.json: status не совпадает с results/summary.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        errors.append("data/manifest.json: files должен быть массивом")
        return

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"data/manifest.json: files[{index}] должен быть объектом")
            continue
        relative = _safe_relative_path(entry.get("path"))
        if relative is None:
            errors.append(f"data/manifest.json: files[{index}].path небезопасен")
            continue
        if relative in seen:
            errors.append(f"data/manifest.json: путь повторяется: {relative}")
        seen.add(relative)
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"data/manifest.json: путь выходит за пределы эксперимента: {relative}")
            continue
        if not path.is_file():
            errors.append(f"data/manifest.json: файл отсутствует: {relative}")
            continue
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            errors.append(f"data/manifest.json: некорректный sha256 для {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"data/manifest.json: sha256 не совпадает для {relative}")
        declared_size = entry.get("size_bytes")
        if declared_size is not None and declared_size != path.stat().st_size:
            errors.append(f"data/manifest.json: size_bytes не совпадает для {relative}")

    if status in EXECUTED_STATUSES:
        listed = seen
        for relative in REQUIRED_RAW_FILES:
            if relative not in listed:
                errors.append(f"data/manifest.json: raw-файл не включён в manifest: {relative}")
    for relative in REQUIRED_FILES:
        if relative != "data/manifest.json" and relative not in seen:
            errors.append(f"data/manifest.json: обязательный файл не включён в manifest: {relative}")


def _validate_capture_provenance(path: Path, value: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(value.get("captured"), bool):
        errors.append(f"{path.name}: поле captured должно быть boolean")
    if not isinstance(value.get("status"), str) or not value["status"].strip():
        errors.append(f"{path.name}: поле status должно быть непустой строкой")


def validate_experiment(experiment_root: Path | str) -> dict[str, Any]:
    """Return a machine-readable validation result for an experiment directory."""

    root = Path(experiment_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return {"valid": False, "status": None, "errors": [f"нет каталога эксперимента: {root}"], "warnings": []}

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"отсутствует обязательный файл: {relative}")

    readme = root / "README.md"
    if readme.is_file():
        try:
            readme_text = readme.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"README.md: не удалось прочитать: {exc}")
        else:
            if not CYRILLIC_RE.search(readme_text):
                errors.append("README.md: документация должна содержать русский текст")

    summary: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    if (root / "results/summary.json").is_file():
        summary = _read_json(root / "results/summary.json", errors)
    if (root / "data/manifest.json").is_file():
        manifest = _read_json(root / "data/manifest.json", errors)

    for relative in ("environment.json", "device.json"):
        path = root / relative
        if path.is_file():
            provenance = _read_json(path, errors)
            if provenance is not None:
                _validate_capture_provenance(path, provenance, errors)
    for relative in ("commands.txt", "hypothesis-preflight.md"):
        path = root / relative
        if path.is_file():
            try:
                if not path.read_text(encoding="utf-8").strip():
                    errors.append(f"{relative}: файл не должен быть пустым")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"{relative}: не удалось прочитать: {exc}")

    status = summary.get("status") if summary else None
    if status not in VALID_STATUSES:
        errors.append(
            "results/summary.json: status должен быть одним из "
            + ", ".join(sorted(VALID_STATUSES))
        )
        status = None
    if status is not None and manifest is not None:
        _validate_manifest(root, manifest, status, errors)

    if status in EXECUTED_STATUSES:
        for relative in REQUIRED_RAW_FILES:
            if not (root / relative).is_file():
                errors.append(f"для статуса {status} отсутствует обязательный raw-файл: {relative}")
    if status in {"failed", "rejected", "blocked"} and summary is not None:
        if not isinstance(summary.get("reason"), str) or not summary["reason"].strip():
            warnings.append(f"results/summary.json: для статуса {status} желательно указать reason")

    if manifest is not None and manifest.get("status") == "planned" and status == "planned":
        warnings.append("эксперимент только запланирован; сырые результаты ещё не требуются")

    return {
        "valid": not errors,
        "status": status,
        "experiment_root": str(root),
        "errors": errors,
        "warnings": warnings,
    }


def build_file_manifest(root: Path | str, paths: Iterable[str], *, status: str) -> dict[str, Any]:
    """Create the hash portion of an E053 manifest for existing relative files."""

    root = Path(root).resolve()
    entries = []
    for relative in sorted(set(paths)):
        safe = _safe_relative_path(relative)
        if safe is None:
            raise ValueError(f"небезопасный относительный путь: {relative!r}")
        path = root / safe
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"путь выходит за пределы эксперимента: {relative!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": safe, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": status,
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="вывести только JSON")
    args = parser.parse_args()
    result = validate_experiment(args.experiment)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
