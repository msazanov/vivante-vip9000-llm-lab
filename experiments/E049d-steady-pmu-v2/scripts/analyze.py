#!/usr/bin/env python3
"""Validate E049d-v2 raw samples and render reproducible summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics


GROUPS = ("core", "cache", "memory")
REPEATS = range(1, 6)
MEASURED_TOKENS = 3
EXPECTED_GENERATED_IDS = [271, 248068, 198, 8160]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_generated_ids(data: bytes) -> list[int]:
    records = [json.loads(line) for line in data.splitlines()]
    return [record["token_id"] for record in records if record["accept_grammar"] == 1]


def validate_sample(run_dir: pathlib.Path, group: str, stock_tokens: bytes) -> dict:
    if (run_dir / "exit.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"nonzero launcher exit: {run_dir}")
    sample = json.loads((run_dir / "pmu.json").read_text(encoding="utf-8"))
    if sample["status"] != "ok" or sample["sample_valid"] is not True:
        raise ValueError(f"invalid sample: {run_dir}")
    if sample["event_group"] != group:
        raise ValueError(f"wrong event group: {run_dir}")
    if not all(sample["sync"].get(key) is True for key in ("started", "acknowledged", "ended")):
        raise ValueError(f"incomplete marker handshake: {run_dir}")
    thermal = sample["thermal"]
    if not thermal["guard_enabled"] or not thermal["readable"] or thermal["tripped"]:
        raise ValueError(f"invalid thermal gate: {run_dir}")
    for event in sample["events"]:
        numeric = (event["value"], event["time_enabled_ns"], event["time_running_ns"], event["running_ratio"])
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite event value: {run_dir}")
        if event["support"] != "supported" or event["sample_valid"] is not True:
            raise ValueError(f"unsupported/invalid event: {run_dir}")
        if event["running_ratio"] < sample["min_running_ratio"]:
            raise ValueError(f"multiplexed event below gate: {run_dir}")
    token_bytes = (run_dir / "token-ids.jsonl").read_bytes()
    if token_bytes != stock_tokens:
        raise ValueError(f"token capture differs from stock: {run_dir}")
    if load_generated_ids(token_bytes) != EXPECTED_GENERATED_IDS:
        raise ValueError(f"unexpected generated ids: {run_dir}")
    return sample


def build_summary(raw_root: pathlib.Path, stock_path: pathlib.Path) -> tuple[dict, list[dict]]:
    stock_tokens = stock_path.read_bytes()
    rows: list[dict] = []
    groups: dict[str, dict] = {}
    all_samples: list[dict] = []
    for group in GROUPS:
        samples = []
        for repeat in REPEATS:
            run_dir = raw_root / "series" / f"{group}-r{repeat}"
            sample = validate_sample(run_dir, group, stock_tokens)
            samples.append(sample)
            all_samples.append(sample)
            elapsed_s = sample["measured_elapsed_ns"] / 1e9
            for event in sample["events"]:
                rows.append(
                    {
                        "group": group,
                        "repeat": repeat,
                        "event": event["name"],
                        "raw_event_count": event["value"],
                        "event_count_per_token": event["value"] / MEASURED_TOKENS,
                        "elapsed_s_for_3_tokens": elapsed_s,
                        "seconds_per_token": elapsed_s / MEASURED_TOKENS,
                        "steady_tokens_per_s": MEASURED_TOKENS / elapsed_s,
                        "running_ratio": event["running_ratio"],
                        "max_temp_c": sample["thermal"]["max_observed_c"],
                    }
                )
        elapsed = [sample["measured_elapsed_ns"] for sample in samples]
        event_summaries = {}
        for index, event in enumerate(samples[0]["events"]):
            values = [sample["events"][index]["value"] for sample in samples]
            event_summaries[event["name"]] = {
                "raw_counts": values,
                "median_raw_count": statistics.median(values),
                "median_event_count_per_token": statistics.median(values) / MEASURED_TOKENS,
                "semantics": "event count; not bytes",
            }
        median_elapsed = statistics.median(elapsed)
        groups[group] = {
            "valid_repeats": len(samples),
            "elapsed_ns": elapsed,
            "median_elapsed_ns_for_3_tokens": median_elapsed,
            "median_seconds_per_token": median_elapsed / 1e9 / MEASURED_TOKENS,
            "median_steady_tokens_per_s": MEASURED_TOKENS / (median_elapsed / 1e9),
            "max_observed_temp_c": max(sample["thermal"]["max_observed_c"] for sample in samples),
            "events": event_summaries,
        }
    all_elapsed = [sample["measured_elapsed_ns"] for sample in all_samples]
    summary = {
        "schema_version": "e049d-steady-pmu/v2",
        "status": "accepted",
        "optimization_claim": False,
        "measured_tokens_per_sample": MEASURED_TOKENS,
        "sample_count": len(all_samples),
        "all_samples_valid": True,
        "stock_token_capture_sha256": sha256(stock_tokens),
        "generated_token_ids": EXPECTED_GENERATED_IDS,
        "overall": {
            "median_elapsed_ns_for_3_tokens": statistics.median(all_elapsed),
            "median_seconds_per_token": statistics.median(all_elapsed) / 1e9 / MEASURED_TOKENS,
            "median_steady_tokens_per_s": MEASURED_TOKENS / (statistics.median(all_elapsed) / 1e9),
            "min_steady_tokens_per_s": min(MEASURED_TOKENS / (value / 1e9) for value in all_elapsed),
            "max_steady_tokens_per_s": max(MEASURED_TOKENS / (value / 1e9) for value in all_elapsed),
            "max_observed_temp_c": max(sample["thermal"]["max_observed_c"] for sample in all_samples),
        },
        "groups": groups,
        "counter_semantics": "ARM PMU event counts only; never DDR bytes",
    }
    core = groups["core"]["events"]
    summary["derived_core_ratios"] = {
        "median_instructions_per_cycle": core["instructions"]["median_raw_count"] / core["cpu_cycles"]["median_raw_count"],
        "median_stall_backend_events_per_cycle": core["stall_backend"]["median_raw_count"] / core["cpu_cycles"]["median_raw_count"],
        "warning": "ratio of separately reduced event medians; not a DDR bandwidth measure",
    }
    return summary, rows


def write_outputs(summary: dict, rows: list[dict], output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_chart(summary: dict, rows: list[dict], destination: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"core": "#2563eb", "cache": "#f97316", "memory": "#16a34a"}
    fig, (speed_ax, event_ax) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group and row["event"] == next(iter(summary["groups"][group]["events"]))]
        x = [row["repeat"] for row in group_rows]
        y = [row["steady_tokens_per_s"] for row in group_rows]
        speed_ax.plot(x, y, marker="o", linewidth=2, color=colors[group], label=group)
        speed_ax.axhline(summary["groups"][group]["median_steady_tokens_per_s"], color=colors[group], alpha=0.3, linestyle="--")
    speed_ax.set_title("Bonsai-27B Q1_0: скорость ровно трёх steady decode-токенов")
    speed_ax.set_xlabel("Номер повтора")
    speed_ax.set_ylabel("Токенов/с в marker-окне")
    speed_ax.set_xticks(list(REPEATS))
    speed_ax.grid(alpha=0.25)
    speed_ax.legend(title="Группа PMU")

    event_names = []
    event_values = []
    event_colors = []
    for group in GROUPS:
        for name, event in summary["groups"][group]["events"].items():
            event_names.append(name.replace("_", "\n"))
            event_values.append(event["median_event_count_per_token"])
            event_colors.append(colors[group])
    event_ax.bar(event_names, event_values, color=event_colors)
    event_ax.set_yscale("log")
    event_ax.set_title("Медиана ARM PMU event counts на токен (логарифмическая шкала; не DDR bytes)")
    event_ax.set_ylabel("Событий на токен")
    event_ax.grid(axis="y", alpha=0.25)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def write_manifest(experiment_root: pathlib.Path, destination: pathlib.Path) -> None:
    entries = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        relative = path.relative_to(experiment_root).as_posix()
        if relative == "README.md":
            kind = "documentation"
        elif relative.endswith(".png"):
            kind = "chart"
        elif relative.startswith("scripts/"):
            kind = "reproducer"
        elif relative.startswith("data/"):
            kind = "derived_data"
        else:
            kind = "raw_evidence"
        payload = path.read_bytes()
        entries.append((relative, sha256(payload), len(payload), kind))
    destination.write_text(
        "path\tsha256\tbytes\tkind\n"
        + "".join(f"{path}\t{digest}\t{size}\t{kind}\n" for path, digest, size, kind in entries),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--stock-tokens", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    summary, rows = build_summary(args.raw_root, args.stock_tokens)
    write_outputs(summary, rows, args.output_dir)
    write_chart(summary, rows, args.output_dir / "e049d-v2-pmu.png")
    write_manifest(args.raw_root.parent, args.output_dir / "manifest.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
