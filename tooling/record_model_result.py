#!/usr/bin/env python3
"""Validate one model run, append JSONL, and render one model-card row."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


START_MARKER = "<!-- MODEL_RESULTS_START -->"
END_MARKER = "<!-- MODEL_RESULTS_END -->"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_STATUSES = {"qualified", "unqualified", "rejected", "failed"}
SOFTWARE_FIELDS = (
    "repository_commit",
    "runtime",
    "runtime_commit",
    "compiler",
    "sdk",
    "driver",
    "kernel",
    "command",
)
WORKLOAD_FIELDS = (
    "tokenizer",
    "prompt_suite",
    "prompt_tokens",
    "generated_tokens",
    "seed",
    "deterministic",
    "warmup_iterations",
)
STATISTIC_FIELDS = ("median", "p10", "p90", "min", "max", "cv", "samples")
STATISTIC_NAMES = ("prompt_tps", "decode_tps", "ttft_ms")
REFERENCE_WORKLOAD_FIELDS = (
    "tokenizer",
    "prompt_suite",
    "prompt_tokens",
    "generated_tokens",
    "seed",
    "deterministic",
)


class ValidationError(ValueError):
    pass


def mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def require_keys(value: dict[str, Any], path: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in value:
            dotted = f"{path}.{key}" if path else key
            raise ValidationError(f"missing required field: {dotted}")


def finite_number_or_null(value: Any, path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{path} must be a finite number or null")
    if value < 0:
        raise ValidationError(f"{path} must not be negative")


def nonnegative_integer(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{path} must be a non-negative integer")


def validate_statistics(performance: dict[str, Any], repetitions: int, qualified: bool) -> None:
    statistics = mapping(performance.get("statistics"), "performance.statistics")
    require_keys(statistics, "performance.statistics", STATISTIC_NAMES)
    for name in STATISTIC_NAMES:
        statistic = mapping(statistics[name], f"performance.statistics.{name}")
        require_keys(statistic, f"performance.statistics.{name}", STATISTIC_FIELDS)
        for field in STATISTIC_FIELDS[:-1]:
            finite_number_or_null(
                statistic[field], f"performance.statistics.{name}.{field}"
            )
        samples = statistic["samples"]
        if not isinstance(samples, list):
            raise ValidationError(
                f"performance.statistics.{name}.samples must be a list"
            )
        for index, sample in enumerate(samples):
            if (
                isinstance(sample, bool)
                or not isinstance(sample, (int, float))
                or not math.isfinite(sample)
                or sample < 0
            ):
                raise ValidationError(
                    f"performance.statistics.{name}.samples[{index}] must be a finite non-negative number"
                )
        if qualified:
            if len(samples) != repetitions:
                raise ValidationError(
                    f"qualified result requires performance.statistics.{name}.samples "
                    "count to equal performance.repetitions"
                )
            if any(statistic[field] is None for field in STATISTIC_FIELDS[:-1]):
                raise ValidationError(
                    f"qualified result requires non-null performance.statistics.{name} summaries"
                )
            ordered = tuple(statistic[field] for field in ("min", "p10", "median", "p90", "max"))
            if any(left > right for left, right in zip(ordered, ordered[1:])):
                raise ValidationError(
                    f"performance.statistics.{name} summaries must be ordered min<=p10<=median<=p90<=max"
                )


def validate_reference_workload_and_runtime(
    result: dict[str, Any], reference: dict[str, Any]
) -> bool:
    return (
        isinstance(reference.get("workload"), dict)
        and all(
            reference["workload"].get(field) == result["workload"][field]
            for field in REFERENCE_WORKLOAD_FIELDS
        )
        and isinstance(reference.get("software"), dict)
        and reference["software"].get("runtime") == result["software"]["runtime"]
        and reference["software"].get("runtime_commit")
        == result["software"]["runtime_commit"]
    )


def validate_raw_result_path(result: dict[str, Any]) -> tuple[str, ...]:
    raw_result = result["raw_result"]
    if not isinstance(raw_result, str) or not raw_result:
        raise ValidationError("raw_result must be a non-empty repository-relative path")
    raw_parts = tuple(raw_result.split("/"))
    if (
        len(raw_parts) < 4
        or raw_parts[:2] != ("benchmarks", "results")
        or raw_parts[2] != result["run_id"]
        or not raw_parts[-1].endswith(".json")
        or any(part in {".", ".."} for part in raw_parts)
        or any(PATH_COMPONENT_PATTERN.fullmatch(part) is None for part in raw_parts)
    ):
        raise ValidationError(
            "raw_result must use safe components under benchmarks/results/<run_id>/ and end in .json"
        )
    return raw_parts


def validate_raw_evidence(result: dict[str, Any], repo_root: Path | str) -> None:
    root = Path(repo_root).resolve()
    raw_parts = validate_raw_result_path(result)
    raw_path = (root / Path(*raw_parts)).resolve(strict=False)
    try:
        raw_path.relative_to(root)
    except ValueError as error:
        raise ValidationError("qualified result raw evidence must remain beneath repo root") from error
    if not raw_path.is_file():
        raise ValidationError(f"qualified result raw evidence missing: {result['raw_result']}")

    run_dir = (root / "benchmarks" / "results" / result["run_id"]).resolve(strict=False)
    try:
        run_dir.relative_to(root)
    except ValueError as error:
        raise ValidationError("qualified result raw evidence run directory must remain beneath repo root") from error
    for filename in ("metadata.json", "stdout.log", "stderr.log", "telemetry.jsonl"):
        if not (run_dir / filename).is_file():
            raise ValidationError(f"qualified result raw evidence missing: {filename}")

    metadata_path = run_dir / "metadata.json"
    try:
        metadata = mapping(json.loads(metadata_path.read_text()), "metadata")
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"qualified result raw metadata is not valid JSON: {metadata_path}") from error
    require_keys(
        metadata,
        "metadata",
        ("run_id", "exit_code", "launch_error", "profiler_error", "files"),
    )
    if metadata["run_id"] != result["run_id"]:
        raise ValidationError("metadata.run_id must equal result.run_id")
    if (
        not isinstance(metadata["exit_code"], int)
        or isinstance(metadata["exit_code"], bool)
        or metadata["exit_code"] != 0
    ):
        raise ValidationError("qualified result raw metadata requires exit_code=0")
    if metadata["launch_error"] is not None or metadata["profiler_error"] is not None:
        raise ValidationError("qualified result raw metadata requires null launch_error/profiler_error")

    files = mapping(metadata["files"], "metadata.files")
    require_keys(files, "metadata.files", ("phases",))
    phase_value = files["phases"]
    if not isinstance(phase_value, str) or not phase_value:
        raise ValidationError("metadata.files.phases must be a safe relative path")
    phase_parts = phase_value.split("/")
    if (
        phase_value.startswith("/")
        or "\\" in phase_value
        or any(part in {"", ".", ".."} for part in phase_parts)
        or any(PATH_COMPONENT_PATTERN.fullmatch(part) is None for part in phase_parts)
    ):
        raise ValidationError("metadata.files.phases must be a safe relative path")
    phase_path = (run_dir / Path(*phase_parts)).resolve(strict=False)
    try:
        phase_path.relative_to(run_dir)
    except ValueError as error:
        raise ValidationError("metadata.files.phases must remain within the run directory") from error
    if not phase_path.is_file():
        raise ValidationError("metadata.files.phases must point to an existing file")


def validate(
    result: Any,
    ledger_records: list[dict[str, Any]] | None = None,
    repo_root: Path | str = Path("."),
) -> dict[str, Any]:
    root = mapping(result, "result")
    require_keys(
        root,
        "",
        (
            "schema_version",
            "run_id",
            "date",
            "experiment",
            "model",
            "configuration",
            "software",
            "workload",
            "performance",
            "quality",
            "status",
            "raw_result",
            "notes",
        ),
    )
    if root["schema_version"] != 1:
        raise ValidationError("schema_version must equal 1")
    if not isinstance(root["run_id"], str) or not RUN_ID_PATTERN.fullmatch(root["run_id"]):
        raise ValidationError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if not isinstance(root["date"], str):
        raise ValidationError("date must be an ISO date string")
    try:
        date.fromisoformat(root["date"])
    except ValueError as error:
        raise ValidationError("date must be an ISO date string") from error
    for field in ("experiment", "notes"):
        if not isinstance(root[field], str):
            raise ValidationError(f"{field} must be a string")

    software = mapping(root["software"], "software")
    require_keys(software, "software", SOFTWARE_FIELDS)
    for field in SOFTWARE_FIELDS:
        if not isinstance(software[field], str):
            raise ValidationError(f"software.{field} must be a string")

    workload = mapping(root["workload"], "workload")
    require_keys(workload, "workload", WORKLOAD_FIELDS)
    for field in ("tokenizer", "prompt_suite"):
        if not isinstance(workload[field], str):
            raise ValidationError(f"workload.{field} must be a string")
    for field in ("prompt_tokens", "generated_tokens", "warmup_iterations"):
        nonnegative_integer(workload[field], f"workload.{field}")
    if not isinstance(workload["seed"], int) or isinstance(workload["seed"], bool):
        raise ValidationError("workload.seed must be an integer")
    if not isinstance(workload["deterministic"], bool):
        raise ValidationError("workload.deterministic must be a boolean")

    model = mapping(root["model"], "model")
    require_keys(model, "model", ("id", "sha256", "quantization", "size_bytes"))
    for field in ("id", "quantization"):
        if not isinstance(model[field], str) or not model[field]:
            raise ValidationError(f"model.{field} must be a non-empty string")
    if not isinstance(model["sha256"], str) or not HEX64_PATTERN.fullmatch(model["sha256"]):
        raise ValidationError("model.sha256 must be exactly 64 hexadecimal characters")
    if not isinstance(model["size_bytes"], int) or isinstance(model["size_bytes"], bool) or model["size_bytes"] <= 0:
        raise ValidationError("model.size_bytes must be a positive integer")

    configuration = mapping(root["configuration"], "configuration")
    require_keys(
        configuration,
        "configuration",
        ("backend", "partition", "context", "batch", "ubatch", "threads"),
    )
    for field in ("backend", "partition"):
        if not isinstance(configuration[field], str) or not configuration[field]:
            raise ValidationError(f"configuration.{field} must be a non-empty string")
    for field in ("context", "batch", "ubatch", "threads"):
        value = configuration[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValidationError(f"configuration.{field} must be a positive integer")

    performance = mapping(root["performance"], "performance")
    performance_fields = (
        "prompt_tps",
        "decode_tps",
        "ttft_ms",
        "peak_rss_mib",
        "repetitions",
    )
    require_keys(performance, "performance", performance_fields)
    for field in performance_fields[:-1]:
        finite_number_or_null(performance[field], f"performance.{field}")
    repetitions = performance["repetitions"]
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 0:
        raise ValidationError("performance.repetitions must be a non-negative integer")
    status = root["status"]
    if status not in ALLOWED_STATUSES:
        raise ValidationError(f"status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}")
    validate_statistics(performance, repetitions, status == "qualified")

    quality = mapping(root["quality"], "quality")
    require_keys(
        quality,
        "quality",
        ("method", "reference_run_id", "metric", "value", "threshold", "passed"),
    )
    for field in ("method", "reference_run_id", "metric"):
        if not isinstance(quality[field], str):
            raise ValidationError(f"quality.{field} must be a string")
    for field in ("value", "threshold"):
        finite_number_or_null(quality[field], f"quality.{field}")
    if quality["passed"] is not None and not isinstance(quality["passed"], bool):
        raise ValidationError("quality.passed must be true, false, or null")

    if status == "qualified":
        for field in SOFTWARE_FIELDS:
            value = software[field]
            if not value.strip() or value.casefold() in {"unknown", "unresolved"}:
                raise ValidationError(
                    f"qualified result requires resolved software.{field}"
                )
        if COMMIT_PATTERN.fullmatch(software["repository_commit"]) is None:
            raise ValidationError(
                "qualified result requires software.repository_commit to be 7-64 hexadecimal characters"
            )
        for field in ("tokenizer", "prompt_suite"):
            value = workload[field]
            if not value.strip() or value.casefold() in {"unknown", "unresolved"}:
                raise ValidationError(
                    f"qualified result requires resolved workload.{field}"
                )
        for field in ("prompt_tokens", "generated_tokens"):
            if workload[field] <= 0:
                raise ValidationError(
                    f"qualified result requires workload.{field} > 0"
                )
        if workload["deterministic"] is not True:
            raise ValidationError("qualified result requires workload.deterministic=true")
        if workload["warmup_iterations"] < 1:
            raise ValidationError(
                "qualified result requires workload.warmup_iterations >= 1"
            )
        for field in ("prompt_tps", "decode_tps", "ttft_ms", "peak_rss_mib"):
            if performance[field] is None:
                raise ValidationError(f"qualified result requires performance.{field}")
        for field in STATISTIC_NAMES:
            median = performance["statistics"][field]["median"]
            if not math.isclose(
                performance[field], median, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValidationError(
                    f"qualified result requires performance.{field} to equal "
                    f"performance.statistics.{field}.median"
                )
        if repetitions <= 0:
            raise ValidationError("qualified result requires performance.repetitions > 0")
        for field in ("method", "metric", "reference_run_id"):
            if not quality[field].strip():
                raise ValidationError(f"qualified result requires non-empty quality.{field}")
        if quality["passed"] is not True:
            raise ValidationError("qualified result requires quality.passed=true")
        if quality["reference_run_id"] == root["run_id"]:
            backend = configuration["backend"].casefold()
            partition = configuration["partition"].casefold()
            if backend != "cpu" or "cpu" not in partition:
                raise ValidationError(
                    "qualified self-reference requires cpu backend and cpu partition"
                )
        elif ledger_records is not None:
            validate_reference(root, ledger_records)
    if status == "rejected" and quality["passed"] is not False:
        raise ValidationError("rejected result requires quality.passed=false")

    validate_raw_result_path(root)
    if status == "qualified":
        validate_raw_evidence(root, repo_root)
    return root


def validate_reference(result: dict[str, Any], ledger_records: list[dict[str, Any]]) -> None:
    reference_run_id = result["quality"]["reference_run_id"]
    reference = next(
        (record for record in ledger_records if record.get("run_id") == reference_run_id),
        None,
    )
    if reference is None:
        raise ValidationError(f"qualified result reference_run_id not found: {reference_run_id}")
    reference_configuration = reference.get("configuration")
    reference_model = reference.get("model")
    if (
        reference.get("status") != "qualified"
        or not isinstance(reference_configuration, dict)
        or str(reference_configuration.get("backend", "")).casefold() != "cpu"
        or not isinstance(reference_model, dict)
        or reference_model.get("sha256") != result["model"]["sha256"]
        or not validate_reference_workload_and_runtime(result, reference)
        or any(
            reference_configuration.get(field) != result["configuration"][field]
            for field in ("context", "batch", "ubatch")
        )
    ):
        raise ValidationError(
            f"qualified result reference_run_id is not a matching qualified CPU reference: {reference_run_id}"
        )


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def compact_number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:g}"


def raw_link(raw_result: str) -> str:
    path = PurePosixPath(raw_result)
    if path.parts and path.parts[0] == "benchmarks":
        target = PurePosixPath("..", *path.parts[1:])
    else:
        target = path
    return f"[raw]({target.as_posix()})"


def render_row(result: dict[str, Any]) -> str:
    configuration = result["configuration"]
    performance = result["performance"]
    quality = result["quality"]
    quality_state = "pass" if quality["passed"] is True else "fail" if quality["passed"] is False else "not-run"
    quality_text = (
        f"{quality['metric']}={compact_number(quality['value'])}; "
        f"threshold={compact_number(quality['threshold'])}; {quality_state}"
    )
    values = (
        result["date"],
        result["run_id"],
        result["experiment"],
        f"{configuration['backend']} / {configuration['partition']}",
        result["model"]["quantization"],
        f"{configuration['context']} / {configuration['batch']} / {configuration['ubatch']} / {configuration['threads']}",
        number(performance["prompt_tps"]),
        number(performance["decode_tps"]),
        number(performance["ttft_ms"], 1),
        number(performance["peak_rss_mib"], 1),
        quality_text,
        result["status"],
        raw_link(result["raw_result"]),
    )
    return "| " + " | ".join(markdown_escape(value) for value in values) + " |"


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"invalid JSON in ledger line {line_number}: {error}") from error
        records.append(mapping(record, f"ledger line {line_number}"))
    return records


def update_contents(
    result: dict[str, Any],
    ledger: Path,
    card: Path,
    records: list[dict[str, Any]] | None = None,
    repo_root: Path | str = Path("."),
) -> tuple[str, str]:
    if records is None:
        records = load_ledger(ledger)
    validate(result, records, repo_root)
    if any(record.get("run_id") == result["run_id"] for record in records):
        raise ValidationError(f"duplicate run_id: {result['run_id']}")
    card_content = card.read_text()
    if card_content.count(START_MARKER) != 1 or card_content.count(END_MARKER) != 1:
        raise ValidationError("model card must contain exactly one start and one end marker")
    if card_content.index(START_MARKER) > card_content.index(END_MARKER):
        raise ValidationError("model card result markers are out of order")

    ledger_prefix = ledger.read_text() if ledger.exists() else ""
    if ledger_prefix and not ledger_prefix.endswith("\n"):
        ledger_prefix += "\n"
    ledger_content = ledger_prefix + json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n"

    insert_at = card_content.index(END_MARKER)
    prefix = card_content[:insert_at]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    card_content = prefix + render_row(result) + "\n" + card_content[insert_at:]
    return ledger_content, card_content


def fsync_parent(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_fsync(path: Path, content: str | bytes) -> None:
    data = content.encode("utf-8") if isinstance(content, str) else content
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_pair(ledger: Path, ledger_content: str, card: Path, card_content: str) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    card.parent.mkdir(parents=True, exist_ok=True)
    ledger_tmp = ledger.with_name(f".{ledger.name}.{os.getpid()}.tmp")
    card_tmp = card.with_name(f".{card.name}.{os.getpid()}.tmp")
    ledger_backup = ledger.with_name(f".{ledger.name}.{os.getpid()}.backup")
    original_ledger = ledger.read_bytes() if ledger.exists() else None
    ledger_replaced = False
    card_replaced = False
    try:
        write_fsync(ledger_tmp, ledger_content)
        write_fsync(card_tmp, card_content)
        os.replace(ledger_tmp, ledger)
        ledger_replaced = True
        fsync_parent(ledger.parent)
        os.replace(card_tmp, card)
        card_replaced = True
        fsync_parent(card.parent)
    except OSError:
        if ledger_replaced and not card_replaced:
            if original_ledger is None:
                try:
                    ledger.unlink()
                except FileNotFoundError:
                    pass
            else:
                write_fsync(ledger_backup, original_ledger)
                os.replace(ledger_backup, ledger)
            fsync_parent(ledger.parent)
        raise
    finally:
        for path in (ledger_tmp, card_tmp, ledger_backup):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def pair_lock(ledger: Path, card: Path):
    if ledger.resolve(strict=False) == card.resolve(strict=False):
        raise ValidationError("ledger and card must not resolve to the same file")
    lock_path = ledger.with_name(f".{ledger.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one canonical model benchmark result.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw_result = json.loads(args.input.read_text())
        with pair_lock(args.ledger, args.card):
            records = load_ledger(args.ledger)
            result = validate(raw_result, records, args.repo_root)
            ledger_content, card_content = update_contents(
                result, args.ledger, args.card, records, args.repo_root
            )
            write_pair(args.ledger, ledger_content, args.card, card_content)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(result["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
