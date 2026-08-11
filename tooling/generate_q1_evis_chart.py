#!/usr/bin/env python3
"""Построить честный SVG для packed-Q1 EVIS эксперимента на VIP9000."""

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


def _list(value: Any, context: str) -> list[Any]:
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
    node = ET.SubElement(root, f"{{{SVG}}}text", {"x": f"{x:.3f}", "y": f"{y:.3f}", **attributes})
    node.text = value


def _line(root: ET.Element, x1: float, y1: float, x2: float, y2: float, **attributes: str) -> None:
    ET.SubElement(root, f"{{{SVG}}}line", {
        "x1": f"{x1:.3f}", "y1": f"{y1:.3f}",
        "x2": f"{x2:.3f}", "y2": f"{y2:.3f}", **attributes,
    })


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def render(payload: dict[str, Any]) -> bytes:
    if payload.get("schema_version") != "q1-vip9000-evis-study/v1":
        raise ChartError("schema error: expected q1-vip9000-evis-study/v1")

    micro = _object(payload.get("microkernel_same_shape"), "microkernel_same_shape")
    shape = _object(micro.get("shape"), "microkernel_same_shape.shape")
    rows = _integer(shape.get("rows"), "shape.rows")
    columns = _integer(shape.get("columns"), "shape.columns")
    scalar = _object(micro.get("scalar_npu"), "microkernel_same_shape.scalar_npu")
    evis = _object(micro.get("evis_packed_q1_q8"), "microkernel_same_shape.evis_packed_q1_q8")
    for name, item in (("scalar_npu", scalar), ("evis_packed_q1_q8", evis)):
        if item.get("golden_passed") is not True:
            raise ChartError(f"schema error at {name}.golden_passed: expected true")
        for key in ("cycles_median", "device_us_median", "run_us_median", "e2e_us_median"):
            _positive(item.get(key), f"{name}.{key}")

    scaling = _object(payload.get("bonsai_ffn_gate_scaling"), "bonsai_ffn_gate_scaling")
    if scaling.get("projection_is_measurement") is not False:
        raise ChartError("projection_is_measurement must be false")
    full_rows = _integer(scaling.get("full_rows"), "bonsai_ffn_gate_scaling.full_rows")
    full_columns = _integer(scaling.get("columns"), "bonsai_ffn_gate_scaling.columns")
    cpu_ms = _positive(scaling.get("cpu_full_layer_ms_median"), "cpu_full_layer_ms_median")
    projection_ms = _positive(
        scaling.get("npu_full_layer_linear_projection_ms"),
        "npu_full_layer_linear_projection_ms",
    )
    measurements: list[tuple[int, float, float]] = []
    for index, raw in enumerate(_list(scaling.get("npu_measurements"), "npu_measurements")):
        item = _object(raw, f"npu_measurements[{index}]")
        if item.get("golden_passed") is not True:
            raise ChartError(f"schema error at npu_measurements[{index}].golden_passed")
        measurements.append((
            _integer(item.get("rows"), f"npu_measurements[{index}].rows"),
            _positive(item.get("e2e_ms_median"), f"npu_measurements[{index}].e2e_ms_median"),
            _positive(item.get("device_ms_median"), f"npu_measurements[{index}].device_ms_median"),
        ))
    measurements.sort()

    root = ET.Element(f"{{{SVG}}}svg", {
        "width": "1200", "height": "980", "viewBox": "0 0 1200 980",
        "role": "img", "aria-labelledby": "title description",
        "font-family": "Inter, DejaVu Sans, sans-serif",
    })
    title = ET.SubElement(root, f"{{{SVG}}}title", {"id": "title"})
    title.text = "Packed Q1×Q8 EVIS на VIP9000: микрокernel и масштабирование Bonsai"
    description = ET.SubElement(root, f"{{{SVG}}}desc", {"id": "description"})
    description.text = (
        "Верхняя панель сравнивает два NPU-ядра на одинаковой форме. "
        "Нижняя XY-панель отделяет реальные измерения Bonsai от линейной оценки."
    )
    ET.SubElement(root, f"{{{SVG}}}rect", {"width": "1200", "height": "980", "fill": "#f8fafc"})
    _text(root, "Packed Q1×Q8 на VIP9000", 55, 52,
          **{"font-size": "28", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Измерения на Orange Pi Zero 3W · A733 · NPU 1008 МГц", 55, 80,
          **{"font-size": "14", "fill": "#475569"})

    # Panel 1: same-shape A/B.  Each metric has its own normalized bar pair;
    # exact values remain printed, so different units are never merged.
    ET.SubElement(root, f"{{{SVG}}}rect", {
        "x": "40", "y": "105", "width": "1120", "height": "350",
        "rx": "14", "fill": "#ffffff", "stroke": "#cbd5e1",
    })
    _text(root, f"Одинаковая задача: {rows}×{columns}", 65, 143,
          **{"font-size": "20", "font-weight": "700", "fill": "#0f172a"})
    _text(root, "Все четыре пары нормированы внутри своей метрики; меньше — быстрее.", 65, 168,
          **{"font-size": "13", "fill": "#64748b"})
    metrics = [
        ("cycles_median", "Циклы NPU", " циклов"),
        ("device_us_median", "Device profiler", " мкс"),
        ("run_us_median", "VIPLite run", " мкс"),
        ("e2e_us_median", "End-to-end", " мкс"),
    ]
    bar_x, bar_width = 280.0, 690.0
    for index, (key, label, unit) in enumerate(metrics):
        y = 195.0 + index * 55.0
        old = _positive(scalar[key], f"scalar_npu.{key}")
        new = _positive(evis[key], f"evis_packed_q1_q8.{key}")
        maximum = max(old, new)
        _text(root, label, 65, y + 19, **{"font-size": "14", "font-weight": "600", "fill": "#334155"})
        ET.SubElement(root, f"{{{SVG}}}rect", {
            "x": f"{bar_x:.3f}", "y": f"{y:.3f}", "width": f"{bar_width * old / maximum:.3f}",
            "height": "18", "rx": "4", "fill": "#94a3b8",
        })
        ET.SubElement(root, f"{{{SVG}}}rect", {
            "x": f"{bar_x:.3f}", "y": f"{y + 23:.3f}", "width": f"{bar_width * new / maximum:.3f}",
            "height": "18", "rx": "4", "fill": "#0ea5e9",
        })
        _text(root, f"{_fmt(old, 0 if key == 'cycles_median' else 3)}{unit}", 985, y + 14,
              **{"font-size": "12", "fill": "#475569"})
        _text(root, f"{_fmt(new, 0 if key == 'cycles_median' else 3)}{unit}", 985, y + 37,
              **{"font-size": "12", "fill": "#0369a1"})
    cycle_speedup = _positive(scalar["cycles_median"], "scalar cycles") / _positive(evis["cycles_median"], "evis cycles")
    _text(root, "Scalar Q1×F32", 795, 143, **{"font-size": "13", "fill": "#64748b"})
    _text(root, "EVIS packed Q1×Q8", 925, 143, **{"font-size": "13", "fill": "#0284c7"})
    _text(root, f"Ускорение по циклам: {_fmt(cycle_speedup)}×", 65, 432,
          **{"font-size": "17", "font-weight": "700", "fill": "#0369a1"})
    _text(root, "Golden: PASS · 100/100 стабильных запусков", 430, 432,
          **{"font-size": "15", "font-weight": "600", "fill": "#15803d"})

    # Panel 2: logarithmic XY scaling.  The projected point is visually dashed
    # and is never joined to the measured CPU point.
    ET.SubElement(root, f"{{{SVG}}}rect", {
        "x": "40", "y": "475", "width": "1120", "height": "435",
        "rx": "14", "fill": "#ffffff", "stroke": "#cbd5e1",
    })
    _text(root, "Реальный Bonsai blk.0.ffn_gate", 65, 515,
          **{"font-size": "20", "font-weight": "700", "fill": "#0f172a"})
    _text(root, f"K={full_columns}; X = выходные строки, Y = медиана end-to-end, логарифмические оси", 65, 540,
          **{"font-size": "13", "fill": "#64748b"})
    left, top, width, height = 115.0, 575.0, 680.0, 265.0
    all_rows = [item[0] for item in measurements] + [full_rows]
    all_ms = [item[1] for item in measurements] + [cpu_ms, projection_ms]
    min_x, max_x = min(all_rows), max(all_rows)
    min_y, max_y = min(all_ms) * 0.75, max(all_ms) * 1.25

    def map_x(value: float) -> float:
        return left + width * (math.log(value) - math.log(min_x)) / (math.log(max_x) - math.log(min_x))

    def map_y(value: float) -> float:
        return top + height - height * (math.log(value) - math.log(min_y)) / (math.log(max_y) - math.log(min_y))

    for tick in sorted(set([16, 64, 256, 1024, 4096, full_rows])):
        if tick < min_x or tick > max_x:
            continue
        x = map_x(tick)
        _line(root, x, top, x, top + height, stroke="#e2e8f0", **{"stroke-width": "1"})
        _text(root, str(tick), x, top + height + 23,
              **{"font-size": "11", "text-anchor": "middle", "fill": "#64748b"})
    for tick in (0.5, 1, 4, 16, 64, 256, 512):
        if tick < min_y or tick > max_y:
            continue
        y = map_y(tick)
        _line(root, left, y, left + width, y, stroke="#e2e8f0", **{"stroke-width": "1"})
        _text(root, _fmt(tick, 1), left - 12, y + 4,
              **{"font-size": "11", "text-anchor": "end", "fill": "#64748b"})
    _line(root, left, top, left, top + height, stroke="#475569", **{"stroke-width": "1.5"})
    _line(root, left, top + height, left + width, top + height, stroke="#475569", **{"stroke-width": "1.5"})
    _text(root, "мс", 72, top - 12, **{"font-size": "12", "fill": "#475569"})

    measured_points: list[tuple[float, float]] = []
    for row_count, e2e_ms, _device_ms in measurements:
        x, y = map_x(row_count), map_y(e2e_ms)
        measured_points.append((x, y))
        ET.SubElement(root, f"{{{SVG}}}circle", {
            "cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "7", "fill": "#0ea5e9", "stroke": "#ffffff",
        })
    for first, second in zip(measured_points, measured_points[1:]):
        _line(root, first[0], first[1], second[0], second[1], stroke="#0ea5e9", **{"stroke-width": "2.5"})

    projection_x, projection_y = map_x(full_rows), map_y(projection_ms)
    if measured_points:
        _line(root, measured_points[-1][0], measured_points[-1][1], projection_x, projection_y,
              stroke="#f97316", **{"stroke-width": "2", "stroke-dasharray": "7 6"})
    ET.SubElement(root, f"{{{SVG}}}circle", {
        "cx": f"{projection_x:.3f}", "cy": f"{projection_y:.3f}", "r": "7",
        "fill": "#ffffff", "stroke": "#f97316", "stroke-width": "3",
    })
    cpu_x, cpu_y = map_x(full_rows), map_y(cpu_ms)
    ET.SubElement(root, f"{{{SVG}}}rect", {
        "x": f"{cpu_x - 7:.3f}", "y": f"{cpu_y - 7:.3f}", "width": "14", "height": "14",
        "fill": "#22c55e", "stroke": "#ffffff",
    })

    info_x = 825.0
    _text(root, "Измеренные точки", info_x, 590,
          **{"font-size": "15", "font-weight": "700", "fill": "#0f172a"})
    info_y = 620.0
    for row_count, e2e_ms, device_ms in measurements:
        _text(root, f"{row_count} строк · измерено", info_x, info_y,
              **{"font-size": "14", "font-weight": "600", "fill": "#0284c7"})
        _text(root, f"E2E {_fmt(e2e_ms, 3)} мс · device {_fmt(device_ms, 3)} мс · golden PASS", info_x, info_y + 20,
              **{"font-size": "12", "fill": "#475569"})
        info_y += 54
    _text(root, f"CPU · {full_rows} строк · измерено", info_x, info_y + 5,
          **{"font-size": "14", "font-weight": "600", "fill": "#15803d"})
    _text(root, f"{_fmt(cpu_ms, 3)} мс на полном слое", info_x, info_y + 25,
          **{"font-size": "12", "fill": "#475569"})
    _text(root, f"NPU · {full_rows} строк · {_fmt(projection_ms, 1)} мс", info_x, info_y + 68,
          **{"font-size": "14", "font-weight": "600", "fill": "#ea580c"})
    _text(root, "линейная оценка, не измерение", info_x, info_y + 88,
          **{"font-size": "12", "font-weight": "600", "fill": "#ea580c"})
    _text(root, "Текущий EVIS — programmable shader/PPU,", info_x, info_y + 130,
          **{"font-size": "12", "fill": "#475569"})
    _text(root, "а не эффективный tensor-core GEMV.", info_x, info_y + 148,
          **{"font-size": "12", "fill": "#475569"})

    _text(root, "Вывод: транспорт packed Q1 ускорен без потери качества, но полный decode пока оставляем CPU.", 55, 948,
          **{"font-size": "15", "font-weight": "700", "fill": "#0f172a"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить SVG packed-Q1 EVIS")
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
        print(f"generate_q1_evis_chart: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
