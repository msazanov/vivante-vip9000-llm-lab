#!/usr/bin/env python3
"""Generate a deterministic Russian SVG from a VIPLite phase summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "vip9000-viplite-phase-profile/v1"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


class ChartError(ValueError):
    pass


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError(f"schema error at {context}: expected number")
    return float(value)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChartError(f"schema error at {context}: expected object")
    return value


def _text(parent: ET.Element, value: str, x: float, y: float, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, f"{{{SVG_NS}}}text", {"x": str(x), "y": str(y), **attrs})
    element.text = value
    return element


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def render(summary: dict[str, Any]) -> bytes:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ChartError("schema error: unsupported schema_version")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ChartError("schema error: invalid run_id")
    stats = _mapping(summary.get("stats_us"), "stats_us")
    first = _mapping(stats.get("first"), "stats_us.first")
    steady = _mapping(stats.get("steady"), "stats_us.steady")
    iterations = _mapping(summary.get("iterations"), "iterations")
    correctness = _mapping(summary.get("correctness"), "correctness")
    thermal = _mapping(summary.get("thermal"), "thermal")

    phases = (
        ("h2d", "H2D", "#2563eb"),
        ("device", "NPU device", "#16a34a"),
        ("run_minus_device", "драйвер/API", "#f59e0b"),
        ("d2h", "D2H", "#9333ea"),
    )
    rows: list[tuple[str, dict[str, Any]]] = [("Первый запуск", first), ("Steady-state", steady)]
    maxima = []
    for _, row in rows:
        maxima.append(sum(_number(_mapping(row.get(key), key).get("median"), f"{key}.median")
                          for key, _, _ in phases))
    scale_max = max(maxima) * 1.08

    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "viewBox": "0 0 1200 740", "width": "1200", "height": "740",
        "role": "img", "aria-labelledby": "title description",
        "font-family": "Inter, DejaVu Sans, sans-serif",
    })
    title = ET.SubElement(root, f"{{{SVG_NS}}}title", {"id": "title"})
    title.text = f"Профиль VIPLite: {run_id}"
    description = ET.SubElement(root, f"{{{SVG_NS}}}desc", {"id": "description"})
    description.text = "Сравнение медианного времени первого и установившегося запуска по фазам."
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"width": "1200", "height": "740", "fill": "#ffffff"})
    _text(root, "Профиль VIPLite: H2D → NPU → D2H", 55, 54,
          **{"font-size": "28", "font-weight": "700", "fill": "#111827"})
    _text(root, run_id, 55, 82, **{"font-size": "14", "fill": "#4b5563"})

    legend_x = 55
    for _, label, color in phases:
        ET.SubElement(root, f"{{{SVG_NS}}}rect", {
            "x": str(legend_x), "y": "108", "width": "14", "height": "14", "fill": color,
        })
        _text(root, label, legend_x + 21, 120, **{"font-size": "13", "fill": "#374151"})
        legend_x += 165

    left, width = 250.0, 860.0
    for row_index, (label, row) in enumerate(rows):
        y = 190.0 + row_index * 135.0
        _text(root, label, 55, y + 30, **{"font-size": "16", "font-weight": "600", "fill": "#111827"})
        cursor = left
        for key, phase_label, color in phases:
            median = _number(_mapping(row.get(key), key).get("median"), f"{key}.median")
            segment = width * median / scale_max
            ET.SubElement(root, f"{{{SVG_NS}}}rect", {
                "x": f"{cursor:.3f}", "y": f"{y:.3f}", "width": f"{segment:.3f}",
                "height": "45", "fill": color, "data-phase": key, "data-row": str(row_index),
            })
            if segment >= 85:
                _text(root, f"{_fmt(median, 1)} мкс", cursor + segment / 2, y + 28,
                      **{"font-size": "12", "font-weight": "600", "text-anchor": "middle", "fill": "#ffffff"})
            cursor += segment
        end_to_end = _number(_mapping(row.get("end_to_end"), "end_to_end").get("median"),
                             "end_to_end.median")
        sum_of_phase_medians = sum(
            _number(_mapping(row.get(key), key).get("median"), f"{key}.median")
            for key, _, _ in phases
        )
        phase_values = " · ".join(
            f"{phase_label} {_fmt(_number(_mapping(row.get(key), key).get('median'), f'{key}.median'), 1)} мкс"
            for key, phase_label, _ in phases
        )
        _text(root, phase_values, left, y + 70,
              **{"font-size": "13", "fill": "#374151"})
        _text(root, f"Σ медиан фаз {_fmt(sum_of_phase_medians / 1000.0, 3)} мс · медиана E2E {_fmt(end_to_end / 1000.0, 3)} мс",
              left, y + 92, **{"font-size": "13", "fill": "#374151"})
        if row_index == 1:
            p95 = _number(_mapping(row.get("end_to_end"), "end_to_end").get("p95"), "end_to_end.p95")
            _text(root, f"p95 E2E {_fmt(p95 / 1000.0, 3)} мс · {_fmt(1_000_000.0 / end_to_end, 1)} инференса/с",
                  left, y + 114, **{"font-size": "13", "fill": "#374151"})

    ET.SubElement(root, f"{{{SVG_NS}}}line", {
        "x1": "55", "y1": "465", "x2": "1145", "y2": "465", "stroke": "#d1d5db",
    })
    peak = _number(thermal.get("npu_peak_c"), "thermal.npu_peak_c")
    start = _number(thermal.get("npu_start_c"), "thermal.npu_start_c")
    end = _number(thermal.get("npu_end_c"), "thermal.npu_end_c")
    min_hz = _number(thermal.get("npu_clock_min_hz"), "thermal.npu_clock_min_hz")
    max_hz = _number(thermal.get("npu_clock_max_hz"), "thermal.npu_clock_max_hz")
    throttling = thermal.get("throttling_evidence")
    if not isinstance(throttling, bool):
        raise ChartError("schema error at thermal.throttling_evidence")
    _text(root, "Тепловой контроль", 55, 510,
          **{"font-size": "20", "font-weight": "700", "fill": "#111827"})
    _text(root, f"NPU: {_fmt(start)} → пик {_fmt(peak)} → {_fmt(end)} °C", 55, 548,
          **{"font-size": "16", "fill": "#111827"})
    _text(root, f"частота: {_fmt(min_hz / 1e6, 0)}–{_fmt(max_hz / 1e6, 0)} МГц · признаки троттлинга: {'да' if throttling else 'нет'}",
          55, 578, **{"font-size": "15", "fill": "#374151"})

    failures = correctness.get("repeat_equal_failures")
    golden = correctness.get("golden_checked")
    if not isinstance(failures, int) or isinstance(failures, bool) or not isinstance(golden, bool):
        raise ChartError("schema error at correctness")
    total = iterations.get("total")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ChartError("schema error at iterations.total")
    _text(root, "Проверка результата", 660, 510,
          **{"font-size": "20", "font-weight": "700", "fill": "#111827"})
    _text(root, f"повторяемость: {total - failures}/{total} · CPU golden: {'проверен' if golden else 'не проверен'}",
          660, 548, **{"font-size": "16", "fill": "#111827"})
    _text(root, "Одинаковый выход подтверждает повторяемость,", 660, 578,
          **{"font-size": "13", "fill": "#4b5563"})
    _text(root, "но не математическую корректность.", 660, 599,
          **{"font-size": "13", "fill": "#4b5563"})

    _text(root, "H2D/D2H на A733 включают map, memcpy и синхронизацию кэша общей DDR; это не PCIe transfer.",
          55, 670, **{"font-size": "13", "fill": "#4b5563"})
    _text(root, "run содержит device time и overhead драйвера/API. Скорость здесь — инференсы/с, не токены/с.",
          55, 697, **{"font-size": "13", "fill": "#4b5563"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить SVG фаз VIPLite")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.summary.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict):
            raise ChartError("schema error: root must be an object")
        payload = render(summary)
        with args.output.open("xb") as stream:
            stream.write(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ChartError) as error:
        print(f"generate_viplite_phase_chart: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
