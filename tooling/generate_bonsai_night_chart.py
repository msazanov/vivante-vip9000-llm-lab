#!/usr/bin/env python3
"""Generate a deterministic SVG for the Bonsai heterogeneous night series.

The input is deliberately small and explicit: only rows whose metric is a
full-model decode in tokens/second and whose exact golden comparison passed
are allowed onto the performance axis.  Operator and accelerator timings are
retained as labelled observations in a separate section of the SVG.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
SERIES = "Bonsai-27B Q1_0 / A733 heterogeneous night"
MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
RUNTIME_COMMIT = "38c66ad0241da4f9fcce541cda8edc219086cec5"
GOLDEN_SHA256 = "a9d86b7d298be35cf27a156be87f9aeedbae3602b877e74ced0fdc75b41c4a5f"


class ChartError(ValueError):
    """Raised when summary evidence cannot be safely interpreted."""


@dataclass(frozen=True)
class Row:
    run_id: str
    label: str
    backend: str
    metric_scope: str
    unit: str
    samples: tuple[float, ...]
    median: float | None
    exact_match: bool
    reference: bool
    rejection_reason: str | None

    @property
    def eligible(self) -> bool:
        return (
            self.metric_scope == "full_model_decode"
            and self.unit == "tokens_per_second"
            and self.exact_match
            and self.median is not None
            and self.median > 0
        )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChartError(f"schema error at {location}: expected object")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChartError(f"schema error at {location}: expected non-empty string")
    return value


def _number(value: Any, location: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError(f"schema error at {location}: expected finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        expectation = "positive finite number" if positive else "non-negative finite number"
        raise ChartError(f"schema error at {location}: expected {expectation}")
    return result


def _pinned_provenance(root: dict[str, Any]) -> None:
    if root.get("series") != SERIES:
        raise ChartError("schema error at series: unexpected Bonsai series")
    if root.get("model_sha256") != MODEL_SHA256:
        raise ChartError("schema error at model_sha256: unexpected pinned Bonsai model")
    if root.get("runtime_commit") != RUNTIME_COMMIT:
        raise ChartError("schema error at runtime_commit: unexpected pinned runtime")
    quality = _mapping(root.get("quality_evidence"), "quality_evidence")
    _string(quality.get("method"), "quality_evidence.method")
    reference_run_id = _string(
        quality.get("reference_run_id"), "quality_evidence.reference_run_id"
    )
    candidate_run_id = _string(
        quality.get("candidate_run_id"), "quality_evidence.candidate_run_id"
    )
    if reference_run_id == candidate_run_id:
        raise ChartError("golden quality evidence requires distinct run ids")
    if quality.get("exact_match") is not True:
        raise ChartError("golden quality evidence requires exact_match=true")
    reference_sha256 = quality.get("reference_sha256")
    candidate_sha256 = quality.get("candidate_sha256")
    if reference_sha256 != GOLDEN_SHA256 or candidate_sha256 != GOLDEN_SHA256:
        raise ChartError("golden quality evidence does not match the pinned stdout SHA-256")


def _quantile(samples: tuple[float, ...], q: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(row: dict[str, Any], location: str) -> tuple[tuple[float, ...], float | None]:
    raw_samples = row.get("samples")
    if raw_samples is None:
        return (), None
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ChartError(f"schema error at {location}.samples: expected non-empty array")
    samples = tuple(_number(value, f"{location}.samples[{index}]") for index, value in enumerate(raw_samples))
    raw_median = row.get("median")
    computed_median = _quantile(samples, 0.5)
    if raw_median is None:
        return samples, computed_median
    supplied_median = _number(raw_median, f"{location}.median")
    if not math.isclose(supplied_median, computed_median, rel_tol=1e-9, abs_tol=1e-12):
        raise ChartError(f"schema error at {location}.median: does not match samples")
    return samples, computed_median


def _reference_marker(row: dict[str, Any]) -> bool:
    for key in ("reference", "is_reference", "fresh_reference"):
        if key in row:
            if not isinstance(row[key], bool):
                raise ChartError(f"schema error at row.{key}: expected boolean")
            if row[key]:
                return True
    return row.get("role") in ("reference", "fresh_reference")


def parse_summary(summary: Any) -> tuple[list[Row], str | None]:
    root = _mapping(summary, "root")
    if root.get("schema_version") != 1 or isinstance(root.get("schema_version"), bool):
        raise ChartError("schema error: unsupported schema_version; expected 1")
    _pinned_provenance(root)
    raw_rows = root.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ChartError("schema error at rows: expected a non-empty array")

    rows: list[Row] = []
    for index, raw in enumerate(raw_rows):
        location = f"rows[{index}]"
        row = _mapping(raw, location)
        run_id = _string(row.get("run_id"), f"{location}.run_id")
        label = _string(row.get("label"), f"{location}.label")
        backend = _string(row.get("backend", "unknown"), f"{location}.backend")
        metric_scope = _string(row.get("metric_scope"), f"{location}.metric_scope")
        unit = _string(row.get("unit"), f"{location}.unit")
        quality = _mapping(row.get("quality"), f"{location}.quality")
        exact_match = quality.get("exact_match")
        if not isinstance(exact_match, bool):
            raise ChartError(f"schema error at {location}.quality.exact_match: expected boolean")
        samples, median = _stats(row, location)
        eligible_shape = metric_scope == "full_model_decode" and unit == "tokens_per_second"
        if eligible_shape and exact_match:
            if median is None or median <= 0 or not samples:
                raise ChartError(f"schema error at {location}: eligible row needs samples and median")
        reason = row.get("rejection_reason")
        if reason is not None:
            reason = _string(reason, f"{location}.rejection_reason")
        if not eligible_shape:
            reason = reason or "operator/device timing is not full-model decode tokens/s"
        elif not exact_match:
            reason = reason or "exact golden не подтверждён"
        rows.append(Row(
            run_id=run_id,
            label=label,
            backend=backend,
            metric_scope=metric_scope,
            unit=unit,
            samples=samples,
            median=median,
            exact_match=exact_match,
            reference=_reference_marker(row),
            rejection_reason=reason,
        ))

    explicit_reference = root.get("reference_run_id")
    if explicit_reference is not None:
        explicit_reference = _string(explicit_reference, "reference_run_id")
        matches = [row for row in rows if row.run_id == explicit_reference]
        if len(matches) != 1:
            raise ChartError("schema error: reference_run_id must resolve exactly once")
        rows = [
            Row(**{**row.__dict__, "reference": row.run_id == explicit_reference})
            for row in rows
        ]
    refs = [row for row in rows if row.reference]
    if len(refs) > 1:
        raise ChartError("schema error: more than one fresh reference row")
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        raise ChartError("schema error: no eligible full-model tokens/s rows")
    if not refs:
        raise ChartError("schema error: fresh reference row is required")
    if not refs[0].eligible:
        raise ChartError("schema error: fresh reference must be eligible full-model tokens/s")
    return rows, refs[0].run_id


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "−").replace(".", ",")


def _percent(value: float) -> str:
    if abs(value) < 0.05:
        value = 0.0
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}{abs(value):.1f}%".replace(".", ",")


def _milliseconds_per_token(tokens_per_second: float) -> str:
    rendered = f"{1000.0 / tokens_per_second:,.1f}"
    return rendered.replace(",", " ").replace(".", ",")


def _text(parent: ET.Element, value: str, x: float, y: float, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, f"{{{SVG_NS}}}text", {"x": f"{x:.3f}", "y": f"{y:.3f}", **attrs})
    element.text = value
    return element


def render(summary: Any) -> bytes:
    rows, reference_id = parse_summary(summary)
    eligible = [row for row in rows if row.eligible]
    screens = [
        row for row in rows
        if not row.eligible
        and row.metric_scope == "full_model_decode"
        and row.unit == "tokens_per_second"
        and row.median is not None
        and row.median > 0
        and row.samples
    ]
    rejected = [row for row in rows if not row.eligible and row not in screens]
    reference = next(row for row in eligible if row.run_id == reference_id)
    reference_median = reference.median
    assert reference_median is not None

    plotted = eligible + screens
    max_value = max(max(_quantile(row.samples, 0.9), row.median or 0.0) for row in plotted)
    max_value = max(max_value, reference_median) * 1.25
    if max_value <= 0:
        max_value = 1.0
    width = float(max(1200, 340 + 320 * len(eligible)))
    left, right = 265.0, width - 75.0
    plot_width = right - left
    top = 150.0
    bar_height = 48.0
    row_gap = 92.0
    plot_bottom = top + max(1, len(eligible)) * row_gap + 30.0
    screen_height = 82.0 + len(screens) * 46.0 if screens else 0.0
    rejected_height = 110.0 + max(0, len(rejected) - 1) * 32.0 if rejected else 0.0
    height = int(plot_bottom + screen_height + rejected_height + 100.0)

    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "viewBox": f"0 0 {int(width)} {height}",
        "width": str(int(width)), "height": str(height), "role": "img",
        "aria-labelledby": "title description",
        "font-family": "Inter, DejaVu Sans, sans-serif",
    })
    title = ET.SubElement(root, f"{{{SVG_NS}}}title", {"id": "title"})
    title.text = "Ночная серия Bonsai-27B: полный decode"
    desc = ET.SubElement(root, f"{{{SVG_NS}}}desc", {"id": "description"})
    desc.text = "Подтверждённые full-model tests и отдельная визуальная панель неподтверждённых screens; operator и device timings не смешиваются с tokens/s."
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"width": str(int(width)), "height": str(height), "fill": "#ffffff"})
    _text(root, "Bonsai-27B Q1_0 — полный decode", 48, 54, **{"font-size": "28", "font-weight": "700", "fill": "#111827"})
    _text(root, "X: backend / topology · Y: tokens/s · выше — лучше", 48, 83, **{"font-size": "15", "fill": "#374151"})
    _text(root, "Верхняя панель: exact golden · нижняя: screens без заявления качества", 48, 109, **{"font-size": "14", "fill": "#4b5563"})

    axis_y = plot_bottom
    ET.SubElement(root, f"{{{SVG_NS}}}line", {"x1": f"{left:.3f}", "y1": f"{axis_y:.3f}", "x2": f"{right:.3f}", "y2": f"{axis_y:.3f}", "stroke": "#111827", "stroke-width": "1.5"})
    ET.SubElement(root, f"{{{SVG_NS}}}line", {"x1": f"{left:.3f}", "y1": f"{top:.3f}", "x2": f"{left:.3f}", "y2": f"{axis_y:.3f}", "stroke": "#111827", "stroke-width": "1.5"})
    for tick in range(6):
        value = max_value * tick / 5.0
        y = axis_y - (axis_y - top) * tick / 5.0
        ET.SubElement(root, f"{{{SVG_NS}}}line", {"x1": f"{left:.3f}", "y1": f"{y:.3f}", "x2": f"{right:.3f}", "y2": f"{y:.3f}", "stroke": "#e5e7eb"})
        _text(root, _fmt(value), left - 18, y + 5, **{"font-size": "12", "text-anchor": "end", "fill": "#4b5563"})
    _text(root, "токенов/с", 48, top - 10, **{"font-size": "14", "font-weight": "600", "fill": "#111827"})
    ref_y = axis_y - (axis_y - top) * reference_median / max_value
    ET.SubElement(root, f"{{{SVG_NS}}}line", {"x1": f"{left:.3f}", "y1": f"{ref_y:.3f}", "x2": f"{right:.3f}", "y2": f"{ref_y:.3f}", "stroke": "#b91c1c", "stroke-width": "2", "stroke-dasharray": "8 5", "data-reference": "fresh"})
    _text(root, f"свежий эталон: {_fmt(reference_median)} ток/с", right, ref_y - 7, **{"font-size": "13", "text-anchor": "end", "fill": "#991b1b"})

    colors = ("#2563eb", "#059669", "#7c3aed", "#d97706")
    for index, row in enumerate(eligible):
        x = left + (index + 0.5) * plot_width / len(eligible)
        median = row.median
        assert median is not None
        y = axis_y - (axis_y - top) * median / max_value
        q10, q90 = _quantile(row.samples, 0.1), _quantile(row.samples, 0.9)
        low_y = axis_y - (axis_y - top) * q10 / max_value
        high_y = axis_y - (axis_y - top) * q90 / max_value
        group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"data-panel": "tokens_per_second", "data-run-id": row.run_id, "data-backend": row.backend, "data-metric": "full_model_decode", "data-status": "qualified", "data-cohort-id": "bonsai-night-20260810"})
        ET.SubElement(group, f"{{{SVG_NS}}}line", {"x1": f"{x:.3f}", "y1": f"{low_y:.3f}", "x2": f"{x:.3f}", "y2": f"{high_y:.3f}", "stroke": "#111827", "stroke-width": "3"})
        ET.SubElement(group, f"{{{SVG_NS}}}line", {"x1": f"{x - 9:.3f}", "y1": f"{low_y:.3f}", "x2": f"{x + 9:.3f}", "y2": f"{low_y:.3f}", "stroke": "#111827", "stroke-width": "2"})
        ET.SubElement(group, f"{{{SVG_NS}}}line", {"x1": f"{x - 9:.3f}", "y1": f"{high_y:.3f}", "x2": f"{x + 9:.3f}", "y2": f"{high_y:.3f}", "stroke": "#111827", "stroke-width": "2"})
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {"cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "11", "fill": colors[index % len(colors)]})
        _text(group, f"{_fmt(median)} ток/с · {_milliseconds_per_token(median)} мс/токен", x, y - 18, **{"font-size": "13", "font-weight": "700", "text-anchor": "middle", "fill": "#111827"})
        _text(group, f"разброс p10–p90: ±{_fmt((q90 - q10) / 2)} ток/с", x, y - 36, **{"font-size": "11", "text-anchor": "middle", "fill": "#4b5563"})
        delta = (median / reference_median - 1.0) * 100.0
        _text(group, _percent(delta), x, axis_y + 22, **{"font-size": "14", "font-weight": "700", "text-anchor": "middle", "fill": "#111827"})
        _text(group, f"{row.backend} · {row.label}", x, axis_y + 43, **{"font-size": "12", "text-anchor": "middle", "fill": "#374151"})

    cursor = plot_bottom + 58.0
    if screens:
        _text(root, "Исследовательские screens", 48, cursor, **{"font-size": "20", "font-weight": "700", "fill": "#92400e"})
        _text(root, "Скорость измерена, но exact golden отдельно не подтверждён; полые маркеры не являются qualified-результатом.", 48, cursor + 25, **{"font-size": "13", "fill": "#92400e"})
        for index, row in enumerate(screens):
            median = row.median
            assert median is not None
            q10, q90 = _quantile(row.samples, 0.1), _quantile(row.samples, 0.9)
            line_y = cursor + 56 + index * 46
            low_x = left + plot_width * q10 / max_value
            high_x = left + plot_width * q90 / max_value
            value_x = left + plot_width * median / max_value
            group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
                "data-panel": "screen_tokens_per_second",
                "data-run-id": row.run_id,
                "data-status": "unqualified",
                "data-backend": row.backend,
            })
            _text(group, row.label, left - 18, line_y + 5, **{"font-size": "12", "text-anchor": "end", "fill": "#374151"})
            ET.SubElement(group, f"{{{SVG_NS}}}line", {"x1": f"{left:.3f}", "y1": f"{line_y:.3f}", "x2": f"{right:.3f}", "y2": f"{line_y:.3f}", "stroke": "#f3f4f6", "stroke-width": "8"})
            ET.SubElement(group, f"{{{SVG_NS}}}line", {"x1": f"{low_x:.3f}", "y1": f"{line_y:.3f}", "x2": f"{high_x:.3f}", "y2": f"{line_y:.3f}", "stroke": "#b45309", "stroke-width": "3"})
            ET.SubElement(group, f"{{{SVG_NS}}}circle", {"cx": f"{value_x:.3f}", "cy": f"{line_y:.3f}", "r": "8", "fill": "#ffffff", "stroke": "#b45309", "stroke-width": "3"})
            _text(group, f"{_fmt(median)} ток/с · {_milliseconds_per_token(median)} мс/токен", min(value_x + 14, right - 2), line_y + 5, **{"font-size": "12", "font-weight": "700", "fill": "#78350f"})
        cursor += screen_height
    if rejected:
        _text(root, "Отклонены от сравнения с model tokens/s", 48, cursor, **{"font-size": "20", "font-weight": "700", "fill": "#7f1d1d"})
        _text(root, "Эти наблюдения показаны отдельно: их миллисекунды, run/device time и fallback не являются скоростью Bonsai.", 48, cursor + 25, **{"font-size": "13", "fill": "#7f1d1d"})
        for index, row in enumerate(rejected):
            group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"data-panel": "rejected", "data-run-id": row.run_id, "data-status": "rejected"})
            detail = f"{row.label} — {row.rejection_reason} [{row.metric_scope}, {row.unit}]"
            if row.median is not None:
                detail += f"; median {_fmt(row.median)} {row.unit}"
            _text(group, "• " + detail, 62, cursor + 53 + index * 32, **{"font-size": "13", "fill": "#374151"})
    _text(root, "Русское пояснение: golden = точное совпадение токенов/текста; tg32 — screen, tg128 — подтверждение; H2D/run/D2H и GEMV не смешиваются с tok/s.", 48, height - 36, **{"font-size": "12", "fill": "#4b5563"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить SVG ночной серии Bonsai")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.summary.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        payload = render(summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as stream:
            stream.write(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ChartError) as error:
        print(f"generate_bonsai_night_chart: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
