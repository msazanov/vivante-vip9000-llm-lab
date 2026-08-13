#!/usr/bin/env python3
"""Проверка данных и воспроизводимые графики E044.

Скрипт намеренно строит графики только из компактных CSV, которые вошли в
репозиторий. Большие telemetry/thermal trace остаются доказательствами на
машине эксперимента и не являются входом для публикации.

Примеры:
    python3 plot_e044.py --check
    MPLCONFIGDIR=/tmp/e044-mpl python3 plot_e044.py --output-dir generated
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SCREEN_CSV = DATA_DIR / "screen_results.csv"
FULL_CSV = DATA_DIR / "full_temperature_1s.csv"
MANIFEST_JSON = DATA_DIR / "manifest.json"

SCREEN_FIELDS = {
    "run_order",
    "variant",
    "mode",
    "avg_tok_s",
    "sample_tok_s",
    "elapsed_s",
    "exit_code",
    "thermal_abort",
    "telemetry_samples",
    "max_cpu_c",
    "max_ddr_c",
    "max_gpu_c",
    "max_npu_c",
    "source_result_dir",
}
FULL_FIELDS = {
    "run_id",
    "elapsed_s",
    "cpu_max_c",
    "ddr_c",
    "gpu_c",
    "npu_c",
}
FULL_RUNS = ("A-split", "B-big-PRFM-1", "B-big-PRFM-2")


def _read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual = set(reader.fieldnames or ())
        if actual != fields:
            raise ValueError(f"{path}: schema mismatch: {sorted(actual)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def _number(row: dict[str, str], key: str, *, nonnegative: bool = True) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {key!r}: {row!r}") from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"invalid value for {key!r}: {value!r}")
    return value


def validate() -> dict[str, Any]:
    """Validate schemas, cardinalities, safety facts, and manifest links."""

    if not MANIFEST_JSON.is_file():
        raise ValueError(f"missing {MANIFEST_JSON}")
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    if manifest.get("schema") != "e044-dataset/v1":
        raise ValueError("unexpected manifest schema")

    screen = _read_csv(SCREEN_CSV, SCREEN_FIELDS)
    full = _read_csv(FULL_CSV, FULL_FIELDS)

    if len(screen) != 4:
        raise ValueError(f"screen row count {len(screen)} != 4")
    orders = [int(row["run_order"]) for row in screen]
    if orders != [1, 2, 3, 4]:
        raise ValueError(f"screen order is not native/A/B/native: {orders}")
    if [row["variant"] for row in screen] != ["native", "a-split", "b-big-prfm", "native"]:
        raise ValueError("screen variants are not the recorded sequence")

    for row in screen:
        if row["mode"] != "screen" or row["exit_code"] != "0" or row["thermal_abort"] != "false":
            raise ValueError(f"screen safety/exit failure: {row}")
        speed = _number(row, "avg_tok_s")
        if speed <= 0:
            raise ValueError(f"non-positive screen throughput: {row}")
        for key in ("sample_tok_s", "elapsed_s", "max_cpu_c", "max_ddr_c", "max_gpu_c", "max_npu_c"):
            _number(row, key)
        if _number(row, "max_cpu_c") >= 85:
            raise ValueError("screen row claims a thermal limit breach")

    if len(full) != int(manifest["full_reset_timeline"]["rows"]):
        raise ValueError(f"full row count {len(full)} does not match manifest")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in full:
        if row["run_id"] not in FULL_RUNS:
            raise ValueError(f"unknown full run {row['run_id']}")
        grouped[row["run_id"]].append(row)
        for key in ("elapsed_s", "cpu_max_c", "ddr_c", "gpu_c", "npu_c"):
            _number(row, key)
        if _number(row, "cpu_max_c") >= 85:
            raise ValueError("full timeline claims a thermal limit breach")
    if tuple(grouped) != FULL_RUNS:
        raise ValueError(f"full run order is not stable: {tuple(grouped)}")
    if [len(grouped[name]) for name in FULL_RUNS] != [22, 37, 36]:
        raise ValueError(f"unexpected one-second bin counts: {[len(grouped[name]) for name in FULL_RUNS]}")
    for name, rows in grouped.items():
        elapsed = [_number(row, "elapsed_s") for row in rows]
        if elapsed != sorted(elapsed) or elapsed[0] != 0:
            raise ValueError(f"{name}: elapsed timeline is not monotonic from zero")

    expected_full_status = {run["throughput_status"] for run in manifest["full_reset_timeline"]["source_runs"]}
    if expected_full_status != {"unqualified; board reset before sample"}:
        raise ValueError("manifest does not preserve the full-run gate decision")

    return {
        "manifest": manifest,
        "screen": screen,
        "full": full,
        "grouped_full": grouped,
    }


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "svg.hashsalt": "e044-cluster-prfm",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Date=None prevents Matplotlib from embedding a wall-clock timestamp in
    # SVG metadata. This makes a second generation byte-for-byte comparable.
    metadata = {"Date": None, "Creator": "E044 plot_e044.py"}
    fig.savefig(output_dir / f"{stem}.png", metadata=metadata, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", metadata=metadata, bbox_inches="tight")


def plot_screen(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt = _matplotlib()
    labels = ["native 1", "A / split", "B / big-only PRFM", "native 2"]
    values = [_number(row, "avg_tok_s") for row in rows]
    colors = ["#7f8c8d", "#2f80ed", "#e67e22", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.7)
    ax.axhline(1.0, color="#263238", linestyle="--", linewidth=1.2, label="1 токен/с")
    ax.set_ylim(0.90, 1.07)
    ax.set_ylabel("Скорость декодирования, токенов/с")
    ax.set_title("E044: короткий screen вариантов cluster-aware PRFM")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.6f}", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.01,
        0.02,
        "n=8, r=1, stock DDR, thermal guard 85 °C; ось Y увеличена вокруг 1 ток/с.\n"
        "B = promising ranking only; full n=32/r=3 gate — UNQUALIFIED.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff8e1", "edgecolor": "#ffb300"},
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, output_dir, "e044_screen_throughput")
    plt.close(fig)


def plot_full(grouped: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    colors = {"cpu_max_c": "#d32f2f", "ddr_c": "#1976d2", "gpu_c": "#388e3c", "npu_c": "#7b1fa2"}
    labels = {"cpu_max_c": "CPU max", "ddr_c": "DDR", "gpu_c": "GPU", "npu_c": "NPU"}
    for ax, run_id in zip(axes, FULL_RUNS):
        rows = grouped[run_id]
        x = [_number(row, "elapsed_s") for row in rows]
        for key in colors:
            y = [_number(row, key) for row in rows]
            ax.plot(x, y, color=colors[key], linewidth=1.7 if key == "cpu_max_c" else 1.0, label=labels[key])
        ax.axhline(85, color="#b71c1c", linestyle="--", linewidth=1.3, label="thermal guard 85 °C" if run_id == FULL_RUNS[0] else None)
        ax.plot(x[-1], max(_number(rows[-1], key) for key in colors), marker="X", markersize=8, color="#212121")
        ax.annotate(
            "RESET\nнет tok/s / quality",
            xy=(x[-1], max(_number(row, "cpu_max_c") for row in rows)),
            xytext=(-65, 12),
            textcoords="offset points",
            fontsize=9,
            color="#212121",
            arrowprops={"arrowstyle": "-", "color": "#616161"},
        )
        ax.set_ylim(25, 90)
        ax.set_ylabel("°C")
        ax.set_title(f"{run_id}: sustained load, результат UNQUALIFIED", loc="left", fontsize=11)
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Время от старта guard, с (агрегация по 1 с)")
    axes[0].legend(loc="upper left", ncol=5, fontsize=9)
    fig.suptitle(
        "E044: температура во время трёх полных запусков и последующий reset платы\n"
        "Лимит 85 °C не достигнут; throughput/quality gate не получен",
        fontsize=14,
    )
    _save(fig, output_dir, "e044_full_reset_temperature")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="только проверить CSV/JSON схему и safety invariants")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()

    data = validate()
    if args.check:
        print("E044 data/schema checks: PASS (screen=4 rows, full=95 rows, 3 reset runs, CPU < 85 C)")
        return 0
    plot_screen(data["screen"], args.output_dir)
    plot_full(data["grouped_full"], args.output_dir)
    print(f"E044 plots written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
