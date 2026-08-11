#!/usr/bin/env python3
"""Построить понятный SVG по экспериментам Q8 reuse на VIP9000."""

from __future__ import annotations

import argparse
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


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChartError(f"schema error at {context}: expected object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ChartError(f"schema error at {context}: expected non-empty array")
    return value


def _positive(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError(f"schema error at {context}: expected number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ChartError(f"schema error at {context}: expected positive finite number")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChartError(f"schema error at {context}: expected positive integer")
    return value


def _text(root: ET.Element, value: str, x: float, y: float, **attributes: str) -> None:
    node = ET.SubElement(
        root,
        f"{{{SVG}}}text",
        {"x": f"{x:.3f}", "y": f"{y:.3f}", **attributes},
    )
    node.text = value


def _rect(root: ET.Element, x: float, y: float, width: float, height: float,
          **attributes: str) -> None:
    ET.SubElement(
        root,
        f"{{{SVG}}}rect",
        {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "width": f"{width:.3f}",
            "height": f"{height:.3f}",
            **attributes,
        },
    )


def _line(root: ET.Element, x1: float, y1: float, x2: float, y2: float,
          **attributes: str) -> None:
    ET.SubElement(
        root,
        f"{{{SVG}}}line",
        {
            "x1": f"{x1:.3f}",
            "y1": f"{y1:.3f}",
            "x2": f"{x2:.3f}",
            "y2": f"{y2:.3f}",
            **attributes,
        },
    )


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _fmt_int(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _variant_cycles(item: dict[str, Any], context: str) -> tuple[float, str]:
    if "device_cycles_median" in item:
        value = _positive(item["device_cycles_median"], f"{context}.device_cycles_median")
        return value, _fmt_int(value)
    raw_range = item.get("device_cycles_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ChartError(f"schema error at {context}: cycles median or two-value range required")
    low = _positive(raw_range[0], f"{context}.device_cycles_range[0]")
    high = _positive(raw_range[1], f"{context}.device_cycles_range[1]")
    if high < low:
        raise ChartError(f"schema error at {context}.device_cycles_range: descending range")
    return (low + high) / 2.0, f"{_fmt_int(low)}–{_fmt_int(high)}"


def render(payload: dict[str, Any]) -> bytes:
    if payload.get("schema_version") != "vip9000-q1-q8-row-reuse/v1":
        raise ChartError("schema error: expected vip9000-q1-q8-row-reuse/v1")

    synthetic = _object(payload.get("synthetic_r64_k128"), "synthetic_r64_k128")
    variants: list[tuple[str, str, float, str]] = []
    for index, raw in enumerate(_array(synthetic.get("variants"), "synthetic_r64_k128.variants")):
        item = _object(raw, f"variants[{index}]")
        variant_id = item.get("id")
        status = item.get("status")
        if not isinstance(variant_id, str) or not isinstance(status, str):
            raise ChartError(f"schema error at variants[{index}]: id/status required")
        if item.get("golden_passed") is not True:
            raise ChartError(f"schema error at variants[{index}].golden_passed: expected true")
        value, label = _variant_cycles(item, f"variants[{index}]")
        variants.append((variant_id, status, value, label))

    tile = _object(payload.get("bonsai_real_tile_1024x5120"), "bonsai_real_tile_1024x5120")
    tile_old = _object(tile.get("baseline_e003"), "bonsai_real_tile_1024x5120.baseline_e003")
    tile_e011 = _object(tile.get("e011_r4_shared_q8"), "bonsai_real_tile_1024x5120.e011_r4_shared_q8")
    tile_new = _object(tile.get("e014_r4_vector_accum"), "bonsai_real_tile_1024x5120.e014_r4_vector_accum")
    for context, item in (
        ("baseline_e003", tile_old),
        ("e011_r4_shared_q8", tile_e011),
        ("e014_r4_vector_accum", tile_new),
    ):
        if item.get("golden_passed") is not True:
            raise ChartError(f"schema error at {context}.golden_passed: expected true")
    old_ms = _positive(tile_old.get("end_to_end_ms_median"), "baseline_e003.end_to_end_ms_median")
    e011_ms = _positive(tile_e011.get("end_to_end_ms_median"), "e011_r4_shared_q8.end_to_end_ms_median")
    new_ms = _positive(
        tile_new.get("end_to_end_ms_median"),
        "e014_r4_vector_accum.end_to_end_ms_median",
    )

    full = _object(payload.get("full_layer_context"), "full_layer_context")
    rows = _integer(full.get("rows"), "full_layer_context.rows")
    cpu_ms = _positive(full.get("measured_cpu_full_layer_ms_median"), "measured_cpu_full_layer_ms_median")
    projected_ms = _positive(full.get("npu_e014_linear_projection_ms"), "npu_e014_linear_projection_ms")
    if full.get("projection_is_measurement") is not False:
        raise ChartError("projection_is_measurement must be false")
    if full.get("tokens_per_second_improvement_demonstrated") is not False:
        raise ChartError("tokens_per_second_improvement_demonstrated must be false")

    labels = {
        "e003-row-local": "E003 · Q8 читается заново",
        "e011-r4-shared-q8": "E011 · Q8 на 4 строки",
        "e011-v3-direct-signs": "E011-v3 · direct signs",
        "e012-r4-direct-signed-read": "E012 · signed read",
        "e013-r8-shared-q8": "E013 · Q8 на 8 строк",
        "e014-r4-vector-accum": "E014 · vector FP32",
    }
    root = ET.Element(
        f"{{{SVG}}}svg",
        {
            "width": "1200",
            "height": "1070",
            "viewBox": "0 0 1200 1070",
            "role": "img",
            "aria-labelledby": "title description",
            "font-family": "Inter, DejaVu Sans, sans-serif",
        },
    )
    title = ET.SubElement(root, f"{{{SVG}}}title", {"id": "title"})
    title.text = "Переиспользование Q8 в бинарном ядре Bonsai на VIP9000"
    description = ET.SubElement(root, f"{{{SVG}}}desc", {"id": "description"})
    description.text = (
        "Три отдельные панели показывают cycles микротеста, измеренный tile Bonsai "
        "и честное сравнение полного CPU слоя с NPU-прогнозом."
    )
    _rect(root, 0, 0, 1200, 1070, fill="#f8fafc")
    _text(root, "Q1×Q8 Bonsai на VIP9000", 55, 48,
          **{"font-size": "28", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Меньше — быстрее · точность всех показанных вариантов прошла golden", 55, 76,
          **{"font-size": "14", "fill": "#475569"})

    # Panel 1: small-shape kernel cost.
    _rect(root, 40, 100, 1120, 390, rx="14", fill="#ffffff", stroke="#cbd5e1")
    _text(root, "1. Микротест 64×128 · аппаратные cycles NPU", 65, 138,
          **{"font-size": "20", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Сравнивается только стоимость kernel; fixed overhead драйвера сюда не входит.", 65, 163,
          **{"font-size": "13", "fill": "#64748b"})
    maximum_cycles = max(item[2] for item in variants)
    bar_x, bar_width = 350.0, 650.0
    for index, (variant_id, status, cycles, value_label) in enumerate(variants):
        y = 190.0 + index * 45.0
        accepted = status.startswith("accepted")
        baseline = status == "baseline"
        color = "#16a34a" if variant_id == "e014-r4-vector-accum" else (
            "#0284c7" if accepted else ("#64748b" if baseline else "#f97316")
        )
        _text(root, labels.get(variant_id, variant_id), 65, y + 18,
              **{"font-size": "13", "font-weight": "600", "fill": "#334155"})
        _rect(root, bar_x, y, bar_width * cycles / maximum_cycles, 22, rx="4", fill=color)
        _text(root, value_label, 1020, y + 17,
              **{"font-size": "13", "font-weight": "600", "fill": color})
    _text(root, "Лучший результат: E014, Q8 reuse R4 + vector FP32", 65, 475,
          **{"font-size": "15", "font-weight": "700", "fill": "#15803d"})

    # Panel 2: exact same real Bonsai tile.
    _rect(root, 40, 515, 1120, 250, rx="14", fill="#ffffff", stroke="#cbd5e1")
    _text(root, "2. Реальный tile Bonsai 1024×5120 · end-to-end", 65, 553,
          **{"font-size": "20", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Одна форма, одни Q1_0 weights и Q8_0 activation, 19 steady запусков.", 65, 578,
          **{"font-size": "13", "fill": "#64748b"})
    tile_max = max(old_ms, e011_ms, new_ms)
    for index, (label, value, color) in enumerate((
        ("E003 · прежнее ядро", old_ms, "#64748b"),
        ("E011 · Q8 reuse R4", e011_ms, "#0284c7"),
        ("E014 · vector FP32", new_ms, "#16a34a"),
    )):
        y = 605.0 + index * 42.0
        _text(root, label, 65, y + 18,
              **{"font-size": "14", "font-weight": "600", "fill": "#334155"})
        _rect(root, 310, y, 650 * value / tile_max, 24, rx="4", fill=color)
        _text(root, f"{_fmt(value, 3)} ms", 980, y + 18,
              **{"font-size": "13", "font-weight": "600", "fill": color})
    speedup = old_ms / new_ms
    _text(root, f"E014: {_fmt(speedup)}× быстрее исходного E003 на измеренном NPU tile", 65, 747,
          **{"font-size": "16", "font-weight": "700", "fill": "#15803d"})

    # Panel 3: full layer, log scale because the gap is 62x.
    _rect(root, 40, 790, 1120, 205, rx="14", fill="#ffffff", stroke="#cbd5e1")
    _text(root, f"3. Полный слой {rows}×5120 · CPU измерен, NPU оценён", 65, 828,
          **{"font-size": "20", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Логарифмическая шкала времени позволяет видеть обе величины; короче — лучше.", 65, 853,
          **{"font-size": "13", "fill": "#64748b"})
    axis_left, axis_right = 330.0, 840.0
    domain_min = min(cpu_ms, projected_ms) * 0.7
    domain_max = max(cpu_ms, projected_ms) * 1.3

    def log_width(value: float) -> float:
        return (axis_right - axis_left) * (
            (math.log(value) - math.log(domain_min)) /
            (math.log(domain_max) - math.log(domain_min))
        )

    for index, (label, value, color, note) in enumerate((
        ("CPU full layer", cpu_ms, "#16a34a", "измерено"),
        ("NPU E014 ×17 tiles", projected_ms, "#f97316", "прогноз, не измерение"),
    )):
        y = 880.0 + index * 47.0
        _text(root, label, 65, y + 18,
              **{"font-size": "14", "font-weight": "600", "fill": "#334155"})
        _line(root, axis_left, y + 12, axis_right, y + 12, stroke="#e2e8f0", **{"stroke-width": "2"})
        _rect(root, axis_left, y, max(4.0, log_width(value)), 24, rx="4", fill=color)
        _text(root, f"{_fmt(value, 3)} ms · {note}", 1135, y + 18,
              **{"font-size": "12", "font-weight": "600", "text-anchor": "end", "fill": color})

    slowdown = projected_ms / cpu_ms
    _text(root, f"Ускорение tokens/s пока не доказано: прогноз NPU в {_fmt(slowdown)}× медленнее CPU.", 55, 1040,
          **{"font-size": "16", "font-weight": "700", "fill": "#991b1b"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить SVG эксперимента Q8 reuse")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.summary.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ChartError("schema error: root must be an object")
        svg = render(payload)
        with args.output.open("xb") as stream:
            stream.write(svg)
    except (OSError, UnicodeError, json.JSONDecodeError, ChartError) as error:
        print(f"generate_q1_reuse_chart: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
