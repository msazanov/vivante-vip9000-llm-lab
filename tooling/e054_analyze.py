#!/usr/bin/env python3
"""Строгий анализ пяти пар E054 без скрытого усреднения провалов."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e054-a76-microgate/v1"
PAIR_COUNT = 5
SAMPLES_PER_RUN = 50
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnalysisError(ValueError):
    """Вход не доказывает заявленный экспериментальный gate."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{field} must be a nonempty string")
    return value


def _sha(value: Any, field: str) -> str:
    text = _text(value, field)
    if not SHA256_RE.fullmatch(text):
        raise AnalysisError(f"{field} must be lowercase SHA-256")
    return text


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AnalysisError(f"{field} must be finite and positive")
    return result


def _run(value: Any, field: str) -> dict[str, Any]:
    run = _mapping(value, field)
    run_id = _text(run.get("run_id"), f"{field}.run_id")
    samples_raw = run.get("samples_us")
    if not isinstance(samples_raw, list) or len(samples_raw) != SAMPLES_PER_RUN:
        raise AnalysisError(f"{field}.samples_us must contain exactly 50 values")
    samples = [
        _number(sample, f"{field}.samples_us[{index}]")
        for index, sample in enumerate(samples_raw)
    ]
    return {
        "run_id": run_id,
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "output_f32_sha256": _sha(
            run.get("output_f32_sha256"), f"{field}.output_f32_sha256"
        ),
        "activation_q8_sha256": _sha(
            run.get("activation_q8_sha256"), f"{field}.activation_q8_sha256"
        ),
    }


def analyze_document(document: Any) -> dict[str, Any]:
    root = _mapping(document, "root")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise AnalysisError("schema_version mismatch")
    gate = root.get("gate")
    if gate not in {"a76_candidate", "null_dispatch"}:
        raise AnalysisError("gate must be a76_candidate or null_dispatch")
    pairs_raw = root.get("pairs")
    if not isinstance(pairs_raw, list) or len(pairs_raw) != PAIR_COUNT:
        raise AnalysisError("pairs must contain exactly five entries")

    if gate == "a76_candidate":
        threshold = _number(
            root.get("minimum_pair_gain_percent"), "minimum_pair_gain_percent"
        )
        if threshold != 12.0:
            raise AnalysisError("a76 candidate threshold must be exactly 12.0 percent")
        comparison_name = "candidate"
    else:
        threshold = _number(
            root.get("maximum_median_overhead_percent"),
            "maximum_median_overhead_percent",
        )
        if threshold != 1.0:
            raise AnalysisError("null dispatch overhead threshold must be exactly 1.0 percent")
        comparison_name = "null"

    seen_pairs: set[str] = set()
    seen_runs: set[str] = set()
    output_hashes: set[str] = set()
    q8_hashes: set[str] = set()
    pair_rows: list[dict[str, Any]] = []

    for index, value in enumerate(pairs_raw):
        pair = _mapping(value, f"pairs[{index}]")
        pair_id = _text(pair.get("pair_id"), f"pairs[{index}].pair_id")
        if pair_id in seen_pairs:
            raise AnalysisError(f"duplicate pair_id: {pair_id}")
        seen_pairs.add(pair_id)
        stock = _run(pair.get("stock"), f"pairs[{index}].stock")
        comparison = _run(pair.get(comparison_name), f"pairs[{index}].{comparison_name}")
        for run in (stock, comparison):
            if run["run_id"] in seen_runs:
                raise AnalysisError(f"duplicate run_id: {run['run_id']}")
            seen_runs.add(run["run_id"])
            output_hashes.add(run["output_f32_sha256"])
            q8_hashes.add(run["activation_q8_sha256"])
        if stock["output_f32_sha256"] != comparison["output_f32_sha256"]:
            raise AnalysisError(f"{pair_id}: F32 golden mismatch")
        if stock["activation_q8_sha256"] != comparison["activation_q8_sha256"]:
            raise AnalysisError(f"{pair_id}: Q8 golden mismatch")

        gain = 100.0 * (stock["median_us"] - comparison["median_us"]) / stock["median_us"]
        overhead = 100.0 * (comparison["median_us"] - stock["median_us"]) / stock["median_us"]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "stock_run_id": stock["run_id"],
                "comparison_run_id": comparison["run_id"],
                "stock_median_us": stock["median_us"],
                "comparison_median_us": comparison["median_us"],
                "gain_percent": gain,
                "overhead_percent": overhead,
            }
        )

    if len(output_hashes) != 1 or len(q8_hashes) != 1:
        raise AnalysisError("all ten runs must have identical F32 and Q8 hashes")

    median_gain = statistics.median(row["gain_percent"] for row in pair_rows)
    median_overhead = statistics.median(row["overhead_percent"] for row in pair_rows)
    if gate == "a76_candidate":
        decision = "PASS_FULL_GATE" if median_gain >= threshold else "REJECT_MICROGATE"
    else:
        decision = "PASS_NULL_OVERHEAD" if median_overhead <= threshold else "REJECT_NULL_OVERHEAD"

    return {
        "schema_version": "e054-a76-analysis/v1",
        "gate": gate,
        "decision": decision,
        "pair_count": PAIR_COUNT,
        "samples_per_run": SAMPLES_PER_RUN,
        "golden_exact": True,
        "output_f32_sha256": next(iter(output_hashes)),
        "activation_q8_sha256": next(iter(q8_hashes)),
        "median_pair_gain_percent": median_gain,
        "median_pair_overhead_percent": median_overhead,
        "pairs": pair_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        result = analyze_document(document)
    except (OSError, json.JSONDecodeError, AnalysisError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
