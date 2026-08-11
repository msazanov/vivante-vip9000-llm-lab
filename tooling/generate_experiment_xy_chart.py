#!/usr/bin/env python3
"""Generate an honest XY overview without mixing tokens and NPU inferences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


class ChartError(ValueError):
    pass


@dataclass(frozen=True)
class ModelPoint:
    run_id: str
    label: str
    latency_ms: float
    decode_tps: float
    p10: float
    p90: float
    comparable: bool
    golden: bool | None


@dataclass(frozen=True)
class OperatorPoint:
    run_id: str
    label: str
    latency_ms: float
    inference_ps: float
    boundary: str
    golden: bool


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChartError(f"schema error at {context}: expected object")
    return value


def _positive(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError(f"schema error at {context}: expected number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ChartError(f"schema error at {context}: expected positive finite number")
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            return _object(json.load(stream), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChartError(f"malformed JSON: {path}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ChartError(f"malformed JSONL: {path}:{line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ChartError(f"malformed JSONL: {path}:{line_number}") from error
                rows.append(_object(value, f"{path}:{line_number}"))
    except (OSError, UnicodeError) as error:
        raise ChartError(f"malformed JSONL: {path}") from error
    if not rows:
        raise ChartError(f"malformed JSONL: empty ledger {path}")
    return rows


def _short_model_label(row: dict[str, Any]) -> str:
    configuration = _object(row.get("configuration"), "configuration")
    workload = _object(row.get("workload"), "workload")
    partition = configuration.get("partition")
    if not isinstance(partition, str) or not partition:
        raise ChartError("schema error at configuration.partition")
    generated = workload.get("generated_tokens")
    if isinstance(generated, bool) or not isinstance(generated, int) or generated <= 0:
        raise ChartError("schema error at workload.generated_tokens")
    if "A55" in partition:
        return f"A55 ×6 · {generated} токенов"
    if "A76" in partition:
        return f"A76 ×2 · {generated} токенов"
    return f"CPU · {generated} токенов"


def load_model_points(ledger: Path) -> tuple[list[ModelPoint], list[str]]:
    points: list[ModelPoint] = []
    missing: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(_jsonl(ledger), 1):
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise ChartError(f"schema error at ledger row {index}.run_id")
        seen.add(run_id)
        status = row.get("status")
        if status == "failed":
            missing.append(run_id)
            continue
        performance = _object(row.get("performance"), f"{run_id}.performance")
        statistics = _object(performance.get("statistics"), f"{run_id}.statistics")
        decode_stats = _object(statistics.get("decode_tps"), f"{run_id}.decode_tps")
        median_raw = decode_stats.get("median")
        if median_raw is None:
            missing.append(run_id)
            continue
        median = _positive(median_raw, f"{run_id}.decode_tps.median")
        p10 = _positive(decode_stats.get("p10"), f"{run_id}.decode_tps.p10")
        p90 = _positive(decode_stats.get("p90"), f"{run_id}.decode_tps.p90")
        if not p10 <= median <= p90:
            raise ChartError(f"schema error at {run_id}: p10 <= median <= p90 violated")
        workload = _object(row.get("workload"), f"{run_id}.workload")
        prompt = workload.get("prompt_tokens")
        generated = workload.get("generated_tokens")
        comparable = prompt == 512 and generated == 128
        quality = _object(row.get("quality"), f"{run_id}.quality")
        golden = quality.get("passed")
        if golden not in (True, False, None):
            raise ChartError(f"schema error at {run_id}.quality.passed")
        points.append(ModelPoint(
            run_id, _short_model_label(row), 1000.0 / median, median,
            p10, p90, comparable, golden,
        ))
    if not points:
        raise ChartError("model ledger has no measured decode points")
    return points, missing


def load_operator_points(results_dir: Path) -> tuple[list[OperatorPoint], list[str]]:
    points: list[OperatorPoint] = []
    missing: list[str] = []
    for path in sorted(results_dir.glob("*/summary.json")):
        row = _json(path)
        schema = row.get("schema_version")
        run_id = row.get("run_id")
        if not isinstance(run_id, str):
            continue
        if schema == "vip9000-capability-run/v1":
            status = row.get("status")
            if isinstance(status, str) and status.startswith("failed"):
                missing.append(run_id)
                continue
            median_value: Any = None
            label = "ShuffleNet · команда run"
            host = row.get("host_run_us")
            if isinstance(host, dict) and host.get("median") is not None:
                median_value = host.get("median")
                label = "ShuffleNet · steady run-only"
            network = row.get("network")
            if median_value is None and isinstance(network, dict):
                median_value = network.get("host_run_us")
                label = "ShuffleNet · один холодный run"
            timing = row.get("timing")
            if median_value is None and isinstance(timing, dict):
                median_value = timing.get("host_run_us")
                label = "ShuffleNet · run в output-capture прогоне"
            if median_value is None:
                continue
            median_us = _positive(median_value, f"{run_id}.host_run_us")
            quality_value = row.get("quality")
            if isinstance(quality_value, dict):
                correctness_verified = quality_value.get("correctness_verified") is True
            else:
                output = row.get("output")
                correctness_verified = isinstance(output, dict) and output.get("correctness_verified") is True
            points.append(OperatorPoint(
                run_id, label, median_us / 1000.0,
                1_000_000.0 / median_us, "run-only",
                correctness_verified,
            ))
        elif schema == "vip9000-viplite-phase-profile/v1":
            steady = _object(_object(row.get("stats_us"), f"{run_id}.stats_us").get("steady"),
                             f"{run_id}.stats_us.steady")
            end_to_end = _object(steady.get("end_to_end"), f"{run_id}.end_to_end")
            median_us = _positive(end_to_end.get("median"), f"{run_id}.end_to_end.median")
            correctness = _object(row.get("correctness"), f"{run_id}.correctness")
            points.append(OperatorPoint(
                run_id, "ShuffleNet · полный вход→NPU→выход", median_us / 1000.0,
                1_000_000.0 / median_us, "end-to-end",
                correctness.get("golden_checked") is True,
            ))
    if not points:
        raise ChartError("results directory has no plottable VIPLite operator points")
    return points, missing


def load_golden(path: Path) -> tuple[str, list[dict[str, Any]]]:
    row = _json(path)
    if row.get("schema_version") != "q1-vip-cpu-golden-tests/v1":
        raise ChartError("schema error: unsupported golden summary")
    run_id = row.get("run_id")
    tests = row.get("tests")
    if not isinstance(run_id, str) or not run_id or not isinstance(tests, list) or not tests:
        raise ChartError("schema error: incomplete golden summary")
    parsed: list[dict[str, Any]] = []
    for index, test in enumerate(tests, 1):
        item = _object(test, f"golden.tests[{index}]")
        if not isinstance(item.get("name"), str) or item.get("kind") not in ("numeric", "guard"):
            raise ChartError(f"schema error at golden.tests[{index}]")
        if not isinstance(item.get("passed"), bool):
            raise ChartError(f"schema error at golden.tests[{index}].passed")
        parsed.append(item)
    tests_run = row.get("tests_run")
    failures = row.get("failures")
    errors = row.get("errors")
    for value, context in ((tests_run, "tests_run"), (failures, "failures"), (errors, "errors")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ChartError(f"schema error at golden.{context}")
    failed_count = sum(1 for item in parsed if not item["passed"])
    if tests_run != len(parsed) or failures + errors != failed_count:
        raise ChartError("schema error: golden aggregate counts do not match tests")
    return run_id, parsed


def _text(parent: ET.Element, value: str, x: float, y: float, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, f"{{{SVG}}}text", {"x": f"{x:.3f}", "y": f"{y:.3f}", **attrs})
    node.text = value
    return node


def _line(parent: ET.Element, x1: float, y1: float, x2: float, y2: float, **attrs: str) -> None:
    ET.SubElement(parent, f"{{{SVG}}}line", {
        "x1": f"{x1:.3f}", "y1": f"{y1:.3f}", "x2": f"{x2:.3f}", "y2": f"{y2:.3f}", **attrs,
    })


def _domain(values: list[float], ratio: float = 0.12) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        pad = max(abs(low) * ratio, 1.0)
    else:
        pad = (high - low) * ratio
    return max(0.0, low - pad), high + pad


def _scale(value: float, domain: tuple[float, float], start: float, end: float) -> float:
    low, high = domain
    return start + (value - low) / (high - low) * (end - start)


def _axes(root: ET.Element, left: float, top: float, width: float, height: float,
          x_domain: tuple[float, float], y_domain: tuple[float, float], x_title: str,
          y_title: str) -> None:
    _line(root, left, top + height, left + width, top + height, stroke="#6b7280", **{"stroke-width": "1"})
    _line(root, left, top, left, top + height, stroke="#6b7280", **{"stroke-width": "1"})
    for index in range(5):
        fraction = index / 4
        x = left + width * fraction
        y = top + height * (1 - fraction)
        x_value = x_domain[0] + (x_domain[1] - x_domain[0]) * fraction
        y_value = y_domain[0] + (y_domain[1] - y_domain[0]) * fraction
        _line(root, x, top + height, x, top + height + 5, stroke="#6b7280")
        _line(root, left - 5, y, left, y, stroke="#6b7280")
        _text(root, f"{x_value:.2f}".replace(".", ","), x, top + height + 21,
              **{"font-size": "11", "text-anchor": "middle", "fill": "#4b5563"})
        _text(root, f"{y_value:.3f}".replace(".", ","), left - 10, y + 4,
              **{"font-size": "11", "text-anchor": "end", "fill": "#4b5563"})
    _text(root, x_title, left + width / 2, top + height + 47,
          **{"font-size": "13", "text-anchor": "middle", "font-weight": "600", "fill": "#111827"})
    _text(root, y_title, left - 62, top + height / 2,
          transform=f"rotate(-90 {left - 62:.3f} {top + height / 2:.3f})",
          **{"font-size": "13", "text-anchor": "middle", "font-weight": "600", "fill": "#111827"})


def render(model_points: list[ModelPoint], failed: list[str], operators: list[OperatorPoint],
           golden_run: str, golden_tests: list[dict[str, Any]]) -> bytes:
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1200 1080", "width": "1200", "height": "1080",
        "role": "img", "aria-labelledby": "title description",
        "font-family": "Inter, DejaVu Sans, sans-serif",
    })
    title = ET.SubElement(root, f"{{{SVG}}}title", {"id": "title"})
    title.text = "XY-карта реальных тестов Bonsai 27B, VIPLite и CPU golden"
    desc = ET.SubElement(root, f"{{{SVG}}}desc", {"id": "description"})
    desc.text = "Две XY-системы и отдельная correctness-панель не смешивают токены языковой модели и запуски ShuffleNet."
    ET.SubElement(root, f"{{{SVG}}}rect", {"width": "1200", "height": "1080", "fill": "#ffffff"})
    _text(root, "Что мы действительно измерили", 55, 48,
          **{"font-size": "28", "font-weight": "700", "fill": "#111827"})
    _text(root, "Токены Bonsai, служебные NPU-запуски и golden показаны в разных системах.", 55, 76,
          **{"font-size": "14", "fill": "#4b5563"})

    # Panel 1: model token generation.
    _text(root, "1. Bonsai 27B: средняя стоимость decode в серии tg128", 55, 118,
          **{"font-size": "20", "font-weight": "700", "fill": "#111827"})
    _text(root, "X = 1000 / median tok/s: это среднее по серии, не профиль отдельного токена. Цвет: PASS / FAIL / не проверено.",
          55, 142, **{"font-size": "13", "fill": "#4b5563"})
    left, top, width, height = 150.0, 165.0, 960.0, 205.0
    x_domain = _domain([point.latency_ms for point in model_points])
    y_domain = _domain([
        value for point in model_points for value in (point.p10, point.decode_tps, point.p90)
    ])
    _axes(root, left, top, width, height, x_domain, y_domain, "средняя стоимость decode, мс/токен",
          "скорость, токен/с")
    for index, point in enumerate(model_points):
        x = _scale(point.latency_ms, x_domain, left, left + width)
        y = _scale(point.decode_tps, y_domain, top + height, top)
        p10_y = _scale(point.p10, y_domain, top + height, top)
        p90_y = _scale(point.p90, y_domain, top + height, top)
        _line(root, x, p10_y, x, p90_y, stroke="#d97706", **{"stroke-width": "2"})
        quality_color = "#16a34a" if point.golden is True else "#dc2626" if point.golden is False else "#d97706"
        quality_fill = "#86efac" if point.golden is True else "#fca5a5" if point.golden is False else "#fbbf24"
        attrs = {"data-role": "model-point", "data-run-id": point.run_id,
                 "stroke": quality_color, "stroke-width": "3"}
        if point.comparable:
            ET.SubElement(root, f"{{{SVG}}}circle", {
                "cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "8", "fill": quality_fill, **attrs,
            })
        else:
            ET.SubElement(root, f"{{{SVG}}}polygon", {
                "points": f"{x:.3f},{y-9:.3f} {x+9:.3f},{y:.3f} {x:.3f},{y+9:.3f} {x-9:.3f},{y:.3f}",
                "fill": "#ffffff", **attrs,
            })
        label_y = y - 15 if index % 2 == 0 else y + 23
        label_on_left = x > left + width * 0.68
        _text(root, f"{point.label}: {point.decode_tps:.3f} ток/с · {point.latency_ms:.1f} мс/ток",
              x - 12 if label_on_left else x + 12, label_y,
              **{"font-size": "12", "text-anchor": "end" if label_on_left else "start",
                 "fill": "#111827"})

    # Panel 2: NPU operator path.
    _text(root, "2. VIPLite operator: один запуск NPU-сети изображений", 55, 442,
          **{"font-size": "20", "font-weight": "700", "fill": "#111827"})
    _text(root, "Это не токены и не Bonsai. Y = 1000 / latency: другая запись той же метрики, не независимый throughput.",
          55, 466, **{"font-size": "13", "fill": "#4b5563"})
    left2, top2, width2, height2 = 150.0, 488.0, 960.0, 190.0
    ox = _domain([point.latency_ms for point in operators])
    oy = _domain([point.inference_ps for point in operators])
    _axes(root, left2, top2, width2, height2, ox, oy, "время одного запуска, мс/inference",
          "скорость, inference/с")
    for index, point in enumerate(operators):
        x = _scale(point.latency_ms, ox, left2, left2 + width2)
        y = _scale(point.inference_ps, oy, top2 + height2, top2)
        operator_color = "#16a34a" if point.golden else "#2563eb"
        operator_fill = "#86efac" if point.golden else "#93c5fd"
        ET.SubElement(root, f"{{{SVG}}}circle", {
            "cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "8", "fill": operator_fill,
            "stroke": operator_color, "stroke-width": "3", "data-role": "operator-point",
            "data-run-id": point.run_id,
        })
        label_y = y - 15
        label_on_left = x > left2 + width2 * 0.55
        _text(root, f"{point.label}: {point.latency_ms:.3f} мс · {point.inference_ps:.1f}/с · golden {'PASS' if point.golden else 'нет'}",
              x - 12 if label_on_left else x + 12, label_y,
              **{"font-size": "12", "text-anchor": "end" if label_on_left else "start",
                 "fill": "#111827"})

    # Panel 3: correctness, deliberately not performance.
    _text(root, "3. CPU golden Q1: правильность математики", 55, 750,
          **{"font-size": "20", "font-weight": "700", "fill": "#111827"})
    passed_count = sum(1 for test in golden_tests if test["passed"])
    numeric_count = sum(1 for test in golden_tests if test["kind"] == "numeric")
    guard_count = len(golden_tests) - numeric_count
    _text(root, f"{golden_run}: {passed_count}/{len(golden_tests)} PASS · арифметика {numeric_count} · защита входов {guard_count}. Это CPU reference, не скорость NPU.",
          55, 774, **{"font-size": "13", "fill": "#4b5563"})
    gx0, gx1, gy_pass, gy_fail, gy_missing = 150.0, 1010.0, 816.0, 853.0, 890.0
    _line(root, gx0, gy_pass, gx1, gy_pass, stroke="#d1d5db")
    _line(root, gx0, gy_fail, gx1, gy_fail, stroke="#d1d5db")
    _line(root, gx0, gy_missing, gx1, gy_missing, stroke="#d1d5db")
    _text(root, "PASS", 132, gy_pass + 4, **{"font-size": "12", "text-anchor": "end", "fill": "#166534"})
    _text(root, "FAIL", 132, gy_fail + 4, **{"font-size": "12", "text-anchor": "end", "fill": "#991b1b"})
    _text(root, "нет проверки", 132, gy_missing + 4,
          **{"font-size": "12", "text-anchor": "end", "fill": "#6b7280"})
    slots = len(golden_tests) + 2
    for index, test in enumerate(golden_tests):
        x = gx0 + (gx1 - gx0) * index / max(1, slots - 1)
        passed = test["passed"]
        y = gy_pass if passed else gy_fail
        ET.SubElement(root, f"{{{SVG}}}circle", {
            "cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "6",
            "fill": "#22c55e" if passed else "#ef4444", "data-role": "golden-point",
            "data-test": test["name"],
        })
        _text(root, str(index + 1), x, 922, **{"font-size": "10", "text-anchor": "middle", "fill": "#4b5563"})
    missing_labels = ("Q1 custom NPU", "ShuffleNet golden")
    for offset, label in enumerate(missing_labels, len(golden_tests)):
        x = gx0 + (gx1 - gx0) * offset / max(1, slots - 1)
        ET.SubElement(root, f"{{{SVG}}}circle", {
            "cx": f"{x:.3f}", "cy": f"{gy_missing:.3f}", "r": "7", "fill": "#ffffff",
            "stroke": "#6b7280", "stroke-width": "2", "data-role": "golden-missing",
        })
        _text(root, label, x, 940 + (offset - len(golden_tests)) * 17,
              **{"font-size": "11", "text-anchor": "middle", "fill": "#374151"})

    _text(root, "Запуски без координат скорости (failed/no metric):", 55, 985,
          **{"font-size": "12", "font-weight": "600", "fill": "#991b1b"})
    midpoint = (len(failed) + 1) // 2
    failed_lines = (failed[:midpoint], failed[midpoint:]) if failed else (("нет",), ())
    for line_index, identifiers in enumerate(failed_lines):
        if identifiers:
            _text(root, ", ".join(identifiers), 55, 1004 + line_index * 19,
                  **{"font-size": "11", "fill": "#991b1b"})
    _text(root, "Вывод: сейчас измерена только CPU-генерация Bonsai; Q1 custom NPU ещё не имеет точки latency/speed.",
          55, 1055, **{"font-size": "13", "font-weight": "600", "fill": "#111827"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить раздельную XY-карту всех экспериментов")
    parser.add_argument("--ledger", type=Path, default=Path("benchmarks/results/model-runs.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--golden-summary", type=Path,
                        default=Path("benchmarks/results/q1-vip-cpu-golden-tests-001/summary.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/charts/experiment-xy-overview.svg"))
    args = parser.parse_args(argv)
    try:
        models, failed = load_model_points(args.ledger)
        operators, failed_operators = load_operator_points(args.results_dir)
        failed.extend(failed_operators)
        golden_run, golden_tests = load_golden(args.golden_summary)
        payload = render(models, failed, operators, golden_run, golden_tests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        print(f"generate_experiment_xy_chart: refusing to overwrite: {error.filename}", file=sys.stderr)
        return 2
    except ChartError as error:
        print(f"generate_experiment_xy_chart: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
