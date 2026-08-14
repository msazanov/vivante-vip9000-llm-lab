#!/usr/bin/env python3
"""Построить честный 5-pair график E054 из проверенного result JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


class PlotError(ValueError):
    pass


def validate_result(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise PlotError("result must be an object")
    if document.get("schema_version") != "e054-a76-analysis/v1":
        raise PlotError("result schema mismatch")
    if document.get("gate") != "a76_candidate":
        raise PlotError("plot accepts only the A76 candidate gate")
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 5:
        raise PlotError("exactly five completed pairs are required")
    identifiers: set[str] = set()
    for index, row in enumerate(pairs):
        if not isinstance(row, dict):
            raise PlotError(f"pairs[{index}] must be an object")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in identifiers:
            raise PlotError("pair IDs must be nonempty and unique")
        identifiers.add(pair_id)
        for key in ("stock_median_us", "comparison_median_us"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PlotError(f"{pair_id}.{key} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise PlotError(f"{pair_id}.{key} must be finite and positive")
        gain = row.get("gain_percent")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
            raise PlotError(f"{pair_id}.gain_percent must be finite")
        recomputed = 100.0 * (
            float(row["stock_median_us"]) - float(row["comparison_median_us"])
        ) / float(row["stock_median_us"])
        if not math.isclose(recomputed, float(gain), rel_tol=1e-12, abs_tol=1e-12):
            raise PlotError(f"{pair_id}.gain_percent is inconsistent")
    median = statistics.median(float(row["gain_percent"]) for row in pairs)
    reported = document.get("median_pair_gain_percent")
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        raise PlotError("median_pair_gain_percent must be numeric")
    if not math.isclose(median, float(reported), rel_tol=1e-12, abs_tol=1e-12):
        raise PlotError("reported median gain is inconsistent")
    return pairs


def render(document: dict[str, Any], output_prefix: Path) -> None:
    rows = validate_result(document)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "e054-a76-q1-multiversion"

    labels = [str(row["pair_id"]) for row in rows]
    stock_ms = [float(row["stock_median_us"]) / 1000.0 for row in rows]
    candidate_ms = [float(row["comparison_median_us"]) / 1000.0 for row in rows]
    gains = [float(row["gain_percent"]) for row in rows]
    x = list(range(5))
    width = 0.36

    fig, (timing, gain) = plt.subplots(2, 1, figsize=(10, 7.2), constrained_layout=True)
    timing.bar([value - width / 2 for value in x], stock_ms, width, label="stock 4×4", color="#64748b")
    timing.bar([value + width / 2 for value in x], candidate_ms, width, label="A76 -mtune", color="#0ea5e9")
    timing.set_ylabel("Медиана 50 GEMV, мс (меньше — лучше)")
    timing.set_xticks(x, labels)
    timing.legend(loc="upper right")
    timing.grid(axis="y", alpha=0.25)
    timing.set_title("E054: реальный Bonsai Q1 tensor, CPU6–7, пять paired-прогонов")

    colors = ["#16a34a" if value >= 12.0 else "#dc2626" for value in gains]
    gain.scatter(x, gains, s=70, c=colors, zorder=3)
    gain.plot(x, gains, color="#94a3b8", linewidth=1, zorder=2)
    gain.axhline(12.0, color="#7c3aed", linestyle="--", label="gate ≥12%")
    gain.axhline(0.0, color="#334155", linewidth=0.8)
    gain.set_xticks(x, labels)
    gain.set_ylabel("Выигрыш A76, % (больше — лучше)")
    gain.set_xlabel("Pair ID; порядок stock/candidate чередовался")
    gain.grid(axis="y", alpha=0.25)
    gain.legend(loc="upper right")
    median = float(document["median_pair_gain_percent"])
    fig.suptitle(
        f"Решение: REJECT — медиана {median:.3f}% < 12%; full n=32 не запускался",
        fontsize=12,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(output_prefix.with_suffix(".svg"), metadata={"Date": "2026-08-14"})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    render(document, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
