#!/usr/bin/env python3
"""Проверка данных и воспроизводимые графики последовательности E045.

E045 содержит пять запусков на плате и одну ошибку wrapper до запуска:

* E045a — пропущен ``-c``; kernel OOM при контексте 262144;
* E045b — ``-c 512``; обе модели загрузились, но ``n=0`` привёл к assertion;
* E045c — ``-c 512 -n 1 -p x``; один BPE-токен оставил пустой prefill;
* E045d — wrapper fail, плата не запускалась;
* E045e — рабочий DSpark smoke: 0.263 ток/с, принято 0 из 8 draft-токенов;
* E045f — target-only control: 1.120 eval ток/с.

Графики показывают реально записанные RSS/MemAvailable, температуры и
результаты E045e/E045f. Сравнение скорости является только направляющим:
E045e и E045f используют different timer scopes. Golden-сравнение токенов ещё
не выполнено; следующий строгий A/B-тест обозначен как E045g.

Примеры:
    python3 plot_e045.py --check
    MPLCONFIGDIR=/tmp/e045-mpl python3 plot_e045.py --output-dir generated
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CSV_PATH = DATA_DIR / "memory_thermal_1s.csv"
CSV_B_PATH = DATA_DIR / "memory_thermal_b_1s.csv"
CSV_C_PATH = DATA_DIR / "memory_thermal_c_1s.csv"
CSV_E_PATH = DATA_DIR / "memory_thermal_e_1s.csv"
SMOKE_PATH = DATA_DIR / "speculation_smoke.csv"
METRICS_PATH = DATA_DIR / "metrics.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

FIELDS = {
    "elapsed_s",
    "rss_gib",
    "mem_available_gib",
    "cpu_max_c",
    "ddr_c",
    "gpu_c",
    "npu_c",
    "process_samples",
    "samples",
}
FIELDS_WITH_RUN = FIELDS | {"run"}

RUN_FILES = {
    "E045a": CSV_PATH,
    "E045b": CSV_B_PATH,
    "E045c": CSV_C_PATH,
    "E045e": CSV_E_PATH,
}
RUN_COLORS = {"E045a": "#c62828", "E045b": "#1565c0", "E045c": "#2e7d32", "E045e": "#6a1b9a", "E045f": "#455a64"}
RUN_LABELS = {
    "E045a": "E045a — OOM / invalid -c",
    "E045b": "E045b — load pass / n=0 assertion",
    "E045c": "E045c — load pass / empty prefill",
    "E045e": "E045e — functional smoke / 0.263 tok/s",
    "E045f": "E045f — target-only eval / 1.120 tok/s",
}
EXPECTED_CLASSIFICATIONS = {
    "E045a": "CONFIGURATION_FAIL",
    "E045b": "LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_N0",
    "E045c": "LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_EMPTY_PREFILL",
    "E045d": "WRAPPER_FAIL_NOT_RUN",
    "E045e": "FUNCTIONAL_SMOKE_PASS",
}
SMOKE_FIELDS = {
    "run",
    "prompt_tokens",
    "requested_tokens",
    "actual_tokens",
    "decoded_s",
    "tok_s",
    "drafted",
    "accepted",
    "acceptance_pct",
    "status",
    "comparison_scope",
}
COMPARISON_PATH = DATA_DIR / "speculation_comparison.csv"
COMPARISON_FIELDS = {
    "case",
    "engine",
    "timer_scope",
    "measured_tokens",
    "timer_s",
    "tok_s",
    "drafted",
    "accepted",
    "acceptance_pct",
    "comparison_scope",
}


def _read_rows(path: Path, *, expected_run: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual = set(reader.fieldnames or ())
        expected = FIELDS_WITH_RUN if expected_run else FIELDS
        if actual != expected:
            raise ValueError(f"{path}: schema mismatch: {sorted(actual)} != {sorted(expected)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    if expected_run and any(row.get("run") != expected_run[-1].lower() for row in rows):
        raise ValueError(f"{path}: run column does not identify {expected_run}")
    elapsed = [_number(row, "elapsed_s") for row in rows]
    if elapsed != list(range(len(rows))):
        raise ValueError(f"{path}: elapsed bins must be consecutive integers from zero")
    for row in rows:
        _number(row, "rss_gib", allow_blank=True)
        _number(row, "mem_available_gib")
        for key in ("cpu_max_c", "ddr_c", "gpu_c", "npu_c", "process_samples", "samples"):
            _number(row, key)
        process_samples = int(float(row["process_samples"]))
        samples = int(float(row["samples"]))
        if process_samples < 0 or samples <= 0 or process_samples > samples:
            raise ValueError(f"{path}: invalid sample counts: {row}")
    return rows


def _read_smoke() -> dict[str, str]:
    with SMOKE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != SMOKE_FIELDS:
            raise ValueError(f"{SMOKE_PATH}: schema mismatch")
        rows = list(reader)
    if len(rows) != 1 or rows[0].get("run") != "e":
        raise ValueError(f"{SMOKE_PATH}: expected exactly one E045e smoke row")
    row = rows[0]
    for key in (
        "prompt_tokens",
        "requested_tokens",
        "actual_tokens",
        "decoded_s",
        "tok_s",
        "drafted",
        "accepted",
        "acceptance_pct",
    ):
        _number(row, key)
    if row["status"] != "FUNCTIONAL_SMOKE_PASS":
        raise ValueError("E045e smoke status changed")
    if row["comparison_scope"] != "SMOKE_NOT_FULL_BENCHMARK_DIFFERENT_TIMER_FROM_E045F":
        raise ValueError("E045e must remain explicitly unpaired smoke data")
    if float(row["actual_tokens"]) != 2 or float(row["tok_s"]) != 0.263:
        raise ValueError("E045e smoke point changed")
    if float(row["drafted"]) != 8 or float(row["accepted"]) != 0 or float(row["acceptance_pct"]) != 0:
        raise ValueError("E045e acceptance evidence changed")
    return row


def _read_comparison() -> list[dict[str, str]]:
    with COMPARISON_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != COMPARISON_FIELDS:
            raise ValueError(f"{COMPARISON_PATH}: schema mismatch")
        rows = list(reader)
    if {row.get("case") for row in rows} != {"E045e", "E045f"}:
        raise ValueError("comparison must contain exactly E045e and E045f")
    for row in rows:
        _number(row, "measured_tokens")
        _number(row, "timer_s")
        _number(row, "tok_s")
        for key in ("drafted", "accepted", "acceptance_pct"):
            _number(row, key, allow_blank=True)
    e = next(row for row in rows if row["case"] == "E045e")
    f = next(row for row in rows if row["case"] == "E045f")
    if e["comparison_scope"] != "SMOKE_PHASE_ONLY" or f["comparison_scope"] != "TARGET_ONLY_PHASE_ONLY":
        raise ValueError("comparison scope must preserve phase-only semantics")
    if e["timer_scope"] == f["timer_scope"]:
        raise ValueError("E045e and E045f timer scopes must remain distinct")
    if float(e["tok_s"]) != 0.263 or float(f["tok_s"]) != 1.120:
        raise ValueError("unexpected directional comparison values")
    return rows


def _number(row: dict[str, str], key: str, *, allow_blank: bool = False) -> float | None:
    raw = row.get(key, "")
    if allow_blank and raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {key!r}: {row!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid value for {key!r}: {value!r}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _summary(rows: list[dict[str, str]]) -> dict[str, float]:
    rss = [_number(row, "rss_gib", allow_blank=True) for row in rows]
    rss_values = [value for value in rss if value is not None]
    return {
        "bins": float(len(rows)),
        "max_rss_gib": max(rss_values) if rss_values else 0.0,
        "min_mem_available_gib": min(float(row["mem_available_gib"]) for row in rows),
        "max_cpu_c": max(float(row["cpu_max_c"]) for row in rows),
        "max_ddr_c": max(float(row["ddr_c"]) for row in rows),
        "max_gpu_c": max(float(row["gpu_c"]) for row in rows),
        "max_npu_c": max(float(row["npu_c"]) for row in rows),
    }


def validate() -> dict[str, Any]:
    """Validate all committed data and the explicit no-throughput gate."""

    manifest = _load_json(MANIFEST_PATH)
    metrics = _load_json(METRICS_PATH)
    if manifest.get("schema") != "e045a-dataset/v1":
        raise ValueError("unexpected manifest schema")
    if metrics.get("schema") != "e045a-metrics/v1":
        raise ValueError("unexpected metrics schema")

    # Keep the original E045a invariants as regression checks.
    if manifest.get("status") != "REJECTED_INVALID_CONFIG/OOM":
        raise ValueError("E045a must remain rejected as invalid-config/OOM")
    if manifest.get("classification") != "CONFIGURATION_FAIL":
        raise ValueError("E045a classification changed")
    if metrics.get("throughput_status") != "unavailable":
        raise ValueError("E045a throughput must be unavailable")
    if metrics.get("acceptance_status") != "unavailable":
        raise ValueError("E045a acceptance must be unavailable")
    config = manifest.get("invocation", {})
    if config.get("context_argument") != "omitted":
        raise ValueError("dataset must preserve omitted -c configuration")
    if metrics.get("configuration_gate", {}).get("context_argument") != "omitted":
        raise ValueError("metrics must preserve omitted -c configuration")

    rows_a = _read_rows(CSV_PATH)
    rows = {"E045a": rows_a}
    rows["E045b"] = _read_rows(CSV_B_PATH, expected_run="E045b")
    rows["E045c"] = _read_rows(CSV_C_PATH, expected_run="E045c")
    rows["E045e"] = _read_rows(CSV_E_PATH, expected_run="E045e")
    rows["E045f"] = _read_rows(DATA_DIR / "memory_thermal_f_1s.csv", expected_run="E045f")
    smoke = _read_smoke()
    comparison = _read_comparison()

    run_a = metrics["run"]
    memory_a = metrics["memory"]
    thermal_a = metrics["thermal"]
    if run_a["n_predict"] != 0 or run_a["exit_code"] != 137 or run_a["signal"] != 9:
        raise ValueError("E045a is not the recorded n=0 SIGKILL result")
    if not memory_a["min_mem_available_gib"] < 0.1:
        raise ValueError("E045a no longer records the OOM pressure")
    if not thermal_a["max_cpu_c"] < thermal_a["limit_c"]:
        raise ValueError("E045a thermal evidence contradicts the no-abort result")
    marker = float(run_a["oom_chart_marker_s"])
    if not 197.6 < marker < 197.8:
        raise ValueError("unexpected E045a OOM marker")

    sequence = metrics.get("sequence", {})
    if sequence.get("pending") != "E045g":
        raise ValueError("E045g must remain the next pending step")
    metric_runs = {entry.get("id"): entry for entry in sequence.get("runs", [])}
    if set(metric_runs) != {"E045a", "E045b", "E045c", "E045d", "E045e", "E045f"}:
        raise ValueError("sequence must contain exactly E045a/E045b/E045c/E045d/E045e/E045f")
    for run_id in ("E045a", "E045b", "E045c", "E045d"):
        expected_classification = EXPECTED_CLASSIFICATIONS[run_id]
        entry = metric_runs[run_id]
        if entry.get("classification") != expected_classification:
            raise ValueError(f"{run_id}: classification changed")
        if entry.get("throughput_status") != "unavailable":
            raise ValueError(f"{run_id}: throughput must remain unavailable")
        if entry.get("acceptance_status") != "unavailable":
            raise ValueError(f"{run_id}: acceptance must remain unavailable")
        if entry.get("quality_status") != "not_run":
            raise ValueError(f"{run_id}: quality must remain not_run")
        if "tok_s" in entry or "acceptance_rate" in entry:
            raise ValueError(f"{run_id}: fake speed/acceptance value present")
        if run_id == "E045d":
            if "summary" in entry or "tok_s" in entry:
                raise ValueError("E045d wrapper failure must have no hardware summary")
            continue
        summary = _summary(rows[run_id])
        committed_summary = entry.get("summary", {})
        for key in ("bins", "max_rss_gib", "min_mem_available_gib", "max_cpu_c"):
            if not math.isclose(float(committed_summary[key]), summary[key], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{run_id}: summary {key} is not data-derived")

    entry_e = metric_runs["E045e"]
    if entry_e.get("classification") != EXPECTED_CLASSIFICATIONS["E045e"]:
        raise ValueError("E045e classification changed")
    if entry_e.get("status") != "FUNCTIONAL_SMOKE_PASS":
        raise ValueError("E045e functional smoke is not marked PASS")
    if entry_e.get("throughput_status") != "measured_smoke_only":
        raise ValueError("E045e throughput must be marked smoke-only")
    if entry_e.get("acceptance_status") != "measured_smoke_only":
        raise ValueError("E045e acceptance must be marked smoke-only")
    if entry_e.get("quality_status") != "not_compared":
        raise ValueError("E045e quality must remain unpaired")
    if entry_e.get("comparison_scope") != "SMOKE_NOT_FULL_BENCHMARK_DIFFERENT_TIMER_FROM_E045F":
        raise ValueError("E045e comparison scope must forbid baseline claims")
    if float(entry_e.get("tok_s")) != float(smoke["tok_s"]):
        raise ValueError("E045e tok/s is not data-derived")
    for key in ("actual_tokens", "drafted", "accepted", "acceptance_pct"):
        if float(entry_e.get(key)) != float(smoke[key]):
            raise ValueError(f"E045e {key} is not data-derived")
    summary_e = _summary(rows["E045e"])
    for key in ("bins", "max_rss_gib", "min_mem_available_gib", "max_cpu_c"):
        if not math.isclose(float(entry_e["summary"][key]), summary_e[key], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"E045e summary {key} is not data-derived")

    entry_f = metric_runs["E045f"]
    if entry_f.get("classification") != "TARGET_ONLY_BASELINE_PASS":
        raise ValueError("E045f baseline classification changed")
    if entry_f.get("throughput_status") != "measured_target_eval_only":
        raise ValueError("E045f must remain target eval-only")
    if entry_f.get("acceptance_status") != "not_applicable":
        raise ValueError("E045f acceptance must remain not applicable")
    if entry_f.get("quality_status") != "visible_prefix_only" or entry_f.get("golden_status") != "not_run":
        raise ValueError("E045f quality scope changed")
    if entry_f.get("token_ids_available") is not False:
        raise ValueError("E045f token-id limitation must remain explicit")
    summary_f = _summary(rows["E045f"])
    for key in ("bins", "max_rss_gib", "min_mem_available_gib", "max_cpu_c"):
        if not math.isclose(float(entry_f["summary"][key]), summary_f[key], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"E045f summary {key} is not data-derived")
    f_cmp = next(row for row in comparison if row["case"] == "E045f")
    if float(entry_f["tok_s"]) != float(f_cmp["tok_s"]):
        raise ValueError("E045f target eval speed is not data-derived")

    return {
        "rows": rows_a,
        "run_rows": rows,
        "metrics": metrics,
        "manifest": manifest,
        "oom_marker_s": marker,
        "sequence_runs": metric_runs,
        "smoke": smoke,
        "comparison": comparison,
    }


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "svg.hashsalt": "e045-dspark-sequence",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Creator": "E045 plot_e045.py"}
    fig.savefig(output_dir / f"{stem}.png", metadata=metadata, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", metadata=metadata, bbox_inches="tight")


def plot_memory(data: dict[str, Any], output_dir: Path) -> None:
    """Retain the focused E045a OOM plot for backwards-compatible links."""

    plt = _matplotlib()
    rows = data["rows"]
    marker = data["oom_marker_s"]
    x = [float(row["elapsed_s"]) for row in rows]
    rss = [_number(row, "rss_gib", allow_blank=True) for row in rows]
    available = [_number(row, "mem_available_gib") for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(x, rss, color="#c62828", linewidth=2.0, marker="o", markersize=2.5, label="RSS процесса, GiB")
    ax.plot(x, available, color="#1565c0", linewidth=2.0, label="MemAvailable, GiB")
    ax.axvline(marker, color="#212121", linestyle="--", linewidth=1.4, label="kernel OOM ≈ 197.7 с")
    ax.scatter([marker], [0], color="#212121", marker="X", s=80, zorder=5)
    ax.annotate(
        "OOM / SIGKILL\nнет throughput",
        xy=(marker, 0),
        xytext=(-85, 32),
        textcoords="offset points",
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#616161"},
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff8e1", "edgecolor": "#ffb300"},
    )
    ax.set_xlim(0, marker + 2.5)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Время от старта профайлера, с")
    ax.set_ylabel("Память, GiB")
    ax.set_title("E045a: загрузка target + DSpark — память до kernel OOM")
    ax.grid(alpha=0.24)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.02,
        "n=0: токены не генерировались; -c был опущен, поэтому это не capacity-gate.\n"
        "RSS после SIGKILL отсутствует; MemAvailable возвращается после убийства процесса.",
        transform=ax.transAxes,
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eceff1", "edgecolor": "#90a4ae"},
    )
    fig.tight_layout()
    _save(fig, output_dir, "e045_memory_oom")
    plt.close(fig)


def plot_temperature(data: dict[str, Any], output_dir: Path) -> None:
    """Retain the focused E045a temperature plot for backwards compatibility."""

    plt = _matplotlib()
    rows = data["rows"]
    marker = data["oom_marker_s"]
    x = [float(row["elapsed_s"]) for row in rows]
    series = {
        "cpu_max_c": ("CPU max", "#d32f2f", 2.0),
        "ddr_c": ("DDR", "#1565c0", 1.4),
        "gpu_c": ("GPU", "#2e7d32", 1.4),
        "npu_c": ("NPU", "#7b1fa2", 1.4),
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for key, (label, color, width) in series.items():
        ax.plot(x, [float(row[key]) for row in rows], color=color, linewidth=width, label=label)
    ax.axhline(85, color="#b71c1c", linestyle="--", linewidth=1.5, label="thermal guard 85 °C")
    ax.axvline(marker, color="#212121", linestyle=":", linewidth=1.2, label="OOM ≈ 197.7 с")
    ax.set_xlim(0, marker + 2.5)
    ax.set_ylim(25, 90)
    ax.set_xlabel("Время от старта профайлера, с")
    ax.set_ylabel("Температура, °C")
    ax.set_title("E045a: температуры платы — OOM произошёл далеко до thermal limit")
    ax.grid(alpha=0.24)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3)
    ax.text(
        0.01,
        0.02,
        "CPU max 52.018 °C; DDR 46.066 °C; GPU 48.794 °C; NPU 45.136 °C.\n"
        "Thermal guard не срабатывал; скорость/acceptance не измерялись.",
        transform=ax.transAxes,
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e8f5e9", "edgecolor": "#81c784"},
    )
    fig.tight_layout()
    _save(fig, output_dir, "e045_temperature")
    plt.close(fig)


def _plot_run(ax: Any, run_id: str, rows: list[dict[str, str]], field: str, **kwargs: Any) -> None:
    x = [float(row["elapsed_s"]) for row in rows]
    y = [_number(row, field, allow_blank=field == "rss_gib") for row in rows]
    ax.plot(x, y, color=RUN_COLORS[run_id], linewidth=2.0, label=RUN_LABELS[run_id], **kwargs)


def plot_memory_comparison(data: dict[str, Any], output_dir: Path) -> None:
    """Compare only measured memory; statuses are annotations, never speed values."""

    plt = _matplotlib()
    fig, (rss_ax, available_ax) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for run_id, rows in data["run_rows"].items():
        _plot_run(rss_ax, run_id, rows, "rss_gib")
        _plot_run(available_ax, run_id, rows, "mem_available_gib")

    rss_ax.axvline(data["oom_marker_s"], color="#212121", linestyle="--", linewidth=1.2, label="E045a kernel OOM")
    rss_ax.scatter([data["oom_marker_s"]], [0], color="#212121", marker="X", s=75, zorder=5)
    rss_ax.set_ylim(0, 12)
    rss_ax.set_ylabel("RSS процесса, GiB")
    rss_ax.set_title("E045: память при загрузке target + DSpark — a/b/c/e/f")
    rss_ax.grid(alpha=0.24)
    rss_ax.set_axisbelow(True)
    rss_ax.legend(loc="upper left", fontsize=9)
    available_ax.axhline(0.1, color="#b71c1c", linestyle=":", linewidth=1.0, label="OOM pressure < 0.1 GiB")
    available_ax.set_ylim(0, 12)
    available_ax.set_xlabel("Время от старта профайлера, с")
    available_ax.set_ylabel("MemAvailable, GiB")
    available_ax.grid(alpha=0.24)
    available_ax.set_axisbelow(True)
    available_ax.legend(loc="lower left", fontsize=9)
    available_ax.set_xlim(0, max(float(row["elapsed_s"]) for rows in data["run_rows"].values() for row in rows) + 3)

    status_lines = [
        "E045a: CONFIGURATION_FAIL — OOM при пропущенном -c",
        "E045b: LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_N0 — n=0 assertion",
        "E045c: LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_EMPTY_PREFILL — prompt x → 1 токен",
        "E045d: WRAPPER_FAIL_NOT_RUN — железо не запускалось",
        "E045e: FUNCTIONAL_SMOKE_PASS — 2 токена, 0.263 tok/s; drafted 8, accepted 0 (0%)",
        "E045f: TARGET_ONLY_BASELINE_PASS — один target eval шаг, 1.120 tok/s",
        "E045e — SMOKE; E045f — target-only eval; timer scopes различаются, это не общий benchmark",
    ]
    fig.text(
        0.01,
        0.005,
        "\n".join(status_lines),
        fontsize=9,
        color="#263238",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffde7", "edgecolor": "#f9a825"},
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _save(fig, output_dir, "e045_memory_comparison")
    plt.close(fig)


def plot_temperature_comparison(data: dict[str, Any], output_dir: Path) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    series = {
        "cpu_max_c": (axes[0, 0], "CPU max, °C"),
        "ddr_c": (axes[0, 1], "DDR, °C"),
        "gpu_c": (axes[1, 0], "GPU, °C"),
        "npu_c": (axes[1, 1], "NPU, °C"),
    }
    for field, (ax, ylabel) in series.items():
        for run_id, rows in data["run_rows"].items():
            _plot_run(ax, run_id, rows, field)
        ax.axhline(85, color="#b71c1c", linestyle="--", linewidth=1.2, label="thermal guard 85 °C")
        ax.set_xlabel("Время, с")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.24)
        ax.set_axisbelow(True)
    axes[0, 0].set_title("CPU")
    axes[0, 1].set_title("DDR")
    axes[1, 0].set_title("GPU")
    axes[1, 1].set_title("NPU")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("E045: температурный профиль preflight/smoke-запусков", fontsize=14)
    fig.text(
        0.01,
        0.005,
        "Во всех аппаратных запусках thermal guard 85 °C не сработал. E045d — wrapper-only; E045e — один smoke без сопоставимого контроля.",
        fontsize=9,
        color="#263238",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e8f5e9", "edgecolor": "#81c784"},
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, output_dir, "e045_temperature_comparison")
    plt.close(fig)


def plot_speculation_smoke(data: dict[str, Any], output_dir: Path) -> None:
    """Show what happened inside the one E045e run without fake A/B comparison."""

    plt = _matplotlib()
    smoke = data["smoke"]
    actual = float(smoke["actual_tokens"])
    decoded_s = float(smoke["decoded_s"])
    tok_s = float(smoke["tok_s"])
    drafted = float(smoke["drafted"])
    accepted = float(smoke["accepted"])
    fig, (draft_ax, metrics_ax) = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": (1.15, 1)})

    # Panel 1: the only actual speculative-decoding comparison in E045e.
    labels = ["draft\nчерновик", "accepted\nBonsai-27B"]
    bars = draft_ax.bar(
        labels,
        [drafted, accepted],
        color=("#6a1b9a", "#455a64"),
        width=0.55,
    )
    for bar, label, value in zip(bars, ("draft = 8", "accepted = 0"), (drafted, accepted)):
        draft_ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.25, 0.25),
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    draft_ax.set_ylim(0, max(drafted * 1.25, 10))
    draft_ax.set_ylabel("Количество токенов")
    draft_ax.set_title("1. Черновые токены и принятые токены")
    draft_ax.grid(axis="y", alpha=0.24)
    draft_ax.set_axisbelow(True)
    draft_ax.text(
        0.02,
        0.02,
        "draft — предложенные черновой моделью токены\n"
        "accepted — Принятые Bonsai-27B токены\n"
        "acceptance = 0 / 8 = 0%",
        transform=draft_ax.transAxes,
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f3e5f5", "edgecolor": "#8e24aa"},
    )

    # Panel 2: values with different units are shown as metric strips, not a
    # misleading shared-scale bar chart.
    metrics_ax.axis("off")
    metrics_ax.set_title("2. Что измерил timer E045e", pad=18)
    metric_rows = (
        ("decoded", f"decoded = {decoded_s:.3f} с", "Время DSpark decode-фазы"),
        ("tokens", f"actual_tokens = {actual:.0f}", "Фактически декодировано"),
        ("rate", f"throughput = {tok_s:.3f} токенов/с", "actual_tokens / decoded"),
    )
    y_positions = (0.78, 0.53, 0.28)
    colors = ("#1565c0", "#2e7d32", "#6a1b9a")
    for (key, value_label, explanation), y, color in zip(metric_rows, y_positions, colors):
        metrics_ax.text(
            0.06,
            y,
            value_label,
            transform=metrics_ax.transAxes,
            fontsize=13 if key == "rate" else 12,
            fontweight="bold",
            color=color,
            va="center",
        )
        metrics_ax.text(
            0.06,
            y - 0.08,
            explanation,
            transform=metrics_ax.transAxes,
            fontsize=10,
            color="#37474f",
            va="center",
        )
    metrics_ax.text(
        0.06,
        0.04,
        "Это разбор одного smoke-запуска, не сравнение\n"
        "с E045f и не итоговый benchmark качества.",
        transform=metrics_ax.transAxes,
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fffde7", "edgecolor": "#f9a825"},
    )
    fig.suptitle("E045e: что произошло за один DSpark smoke-запуск", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_dir, "e045_speculation_smoke")
    plt.close(fig)


def plot_speculation_comparison(data: dict[str, Any], output_dir: Path) -> None:
    """Show phase rates while explicitly exposing the timer mismatch."""

    plt = _matplotlib()
    rows = {row["case"]: row for row in data["comparison"]}
    labels = ["E045e\nDSpark smoke\n(decode phase)", "E045f\ntarget-only\n(eval phase)"]
    values = [float(rows["E045e"]["tok_s"]), float(rows["E045f"]["tok_s"])]
    colors = [RUN_COLORS["E045e"], "#455a64"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f} tok/s", ha="center", fontsize=11)
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("Измеренная скорость внутри указанного timer, токенов/с")
    ax.set_title("E045e vs E045f: направленное phase-only сравнение")
    ax.grid(axis="y", alpha=0.24)
    ax.set_axisbelow(True)
    ax.text(
        0.02,
        0.97,
        "Важно: timers не совпадают. E045e = decode после load+encode,\n"
        "включает draft/Markov/verify/accept; E045f = target eval,\n"
        "без load и prompt eval. Поэтому это не speedup benchmark.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffde7", "edgecolor": "#f9a825"},
    )
    ratio = values[1] / values[0]
    ax.text(
        0.02,
        0.02,
        f"Направленное отношение phase rates: {ratio:.2f}× в пользу target-only.\n"
        "Не является выводом о качестве, golden или итоговом ускорении модели.",
        transform=ax.transAxes,
        fontsize=9,
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#eceff1", "edgecolor": "#90a4ae"},
    )
    fig.tight_layout()
    _save(fig, output_dir, "e045_speculation_comparison")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="проверить CSV/JSON и sequence invariants")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    data = validate()
    if args.check:
        print("E045 sequence data/schema checks: PASS (198/31/32/43 bins; E045e smoke-only point, no fake baseline)")
        return 0
    plot_memory(data, args.output_dir)
    plot_temperature(data, args.output_dir)
    plot_memory_comparison(data, args.output_dir)
    plot_temperature_comparison(data, args.output_dir)
    plot_speculation_smoke(data, args.output_dir)
    plot_speculation_comparison(data, args.output_dir)
    print(f"E045 plots written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
