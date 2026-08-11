#!/usr/bin/env python3
"""Parse the phase-aware VIPLite runner log into deterministic JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA_VERSION = "vip9000-viplite-phase-profile/v1"
DRIVER_PREFIX = "VIPLite driver software version "
FLOAT = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


class ProfileError(ValueError):
    """The runner log is incomplete or violates the emitted schema."""


def _fields(line: str, prefix: str, line_number: int) -> dict[str, str]:
    parts = line.split(",")
    if parts[0] != prefix:
        raise ProfileError(f"line {line_number}: expected {prefix}")
    result: dict[str, str] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ProfileError(f"line {line_number}: malformed field {item!r}")
        key, value = item.split("=", 1)
        if not key or not value or key in result:
            raise ProfileError(f"line {line_number}: malformed or duplicate field {key!r}")
        result[key] = value
    return result


def _require(fields: dict[str, str], names: set[str], line_number: int) -> None:
    if set(fields) != names:
        raise ProfileError(
            f"line {line_number}: fields differ: expected={sorted(names)} got={sorted(fields)}"
        )


def _float(value: str, context: str) -> float:
    if FLOAT.fullmatch(value) is None:
        raise ProfileError(f"{context}: expected finite non-negative number")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileError(f"{context}: expected finite number")
    return result


def _int(value: str, context: str, *, non_negative: bool = False) -> int:
    if INTEGER.fullmatch(value) is None:
        raise ProfileError(f"{context}: expected integer")
    result = int(value)
    if non_negative and result < 0:
        raise ProfileError(f"{context}: expected non-negative integer")
    return result


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    rank = 0.95 * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {
        "min": round(ordered[0], 3),
        "median": round(median, 3),
        "p95": round(p95, 3),
        "max": round(ordered[-1], 3),
    }


def _tensor(fields: dict[str, str], line_number: int) -> dict[str, Any]:
    common = {"kind", "index", "format", "quant", "dims", "shape"}
    affine = {"scale", "zero_point"}
    if set(fields) not in (common, common | affine):
        raise ProfileError(f"line {line_number}: unexpected tensor fields")
    dims = _int(fields["dims"], f"line {line_number}.dims", non_negative=True)
    shape = [_int(item, f"line {line_number}.shape", non_negative=True)
             for item in fields["shape"].split("x")]
    if len(shape) != dims or any(value == 0 for value in shape):
        raise ProfileError(f"line {line_number}: shape does not match dims")
    result: dict[str, Any] = {
        "index": _int(fields["index"], f"line {line_number}.index", non_negative=True),
        "format": _int(fields["format"], f"line {line_number}.format"),
        "quant": _int(fields["quant"], f"line {line_number}.quant"),
        "shape": shape,
    }
    if affine <= set(fields):
        result["scale"] = _float(fields["scale"], f"line {line_number}.scale")
        result["zero_point"] = _int(fields["zero_point"], f"line {line_number}.zero_point")
    return result


def summarize(path: Path, run_id: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, UnicodeError) as error:
        raise ProfileError(f"cannot read runner log: {path}") from error
    if not lines:
        raise ProfileError("runner log is empty")

    driver: str | None = None
    setup: dict[str, float] = {}
    declared_counts: tuple[int, int] | None = None
    tensors: dict[str, dict[int, dict[str, Any]]] = {"input": {}, "output": {}}
    buffer_sizes: dict[str, dict[int, int]] = {"input": {}, "output": {}}
    iterations: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines, 1):
        if line.startswith(DRIVER_PREFIX):
            if driver is not None:
                raise ProfileError(f"line {line_number}: duplicate driver version")
            driver = line.removeprefix(DRIVER_PREFIX)
            if not driver:
                raise ProfileError(f"line {line_number}: empty driver version")
        elif line.startswith("phase,"):
            fields = _fields(line, "phase", line_number)
            _require(fields, {"name", "time_us"}, line_number)
            if fields["name"] in setup:
                raise ProfileError(f"line {line_number}: duplicate phase {fields['name']}")
            setup[fields["name"]] = _float(fields["time_us"], f"line {line_number}.time_us")
        elif line.startswith("network,"):
            fields = _fields(line, "network", line_number)
            _require(fields, {"inputs", "outputs"}, line_number)
            if declared_counts is not None:
                raise ProfileError(f"line {line_number}: duplicate network declaration")
            declared_counts = (
                _int(fields["inputs"], f"line {line_number}.inputs", non_negative=True),
                _int(fields["outputs"], f"line {line_number}.outputs", non_negative=True),
            )
        elif line.startswith("tensor,"):
            fields = _fields(line, "tensor", line_number)
            kind = fields.get("kind")
            if kind not in tensors:
                raise ProfileError(f"line {line_number}: invalid tensor kind")
            parsed = _tensor(fields, line_number)
            index = parsed["index"]
            if index in tensors[kind]:
                raise ProfileError(f"line {line_number}: duplicate {kind} tensor {index}")
            tensors[kind][index] = parsed
        elif line.startswith("buffer,"):
            fields = _fields(line, "buffer", line_number)
            _require(fields, {"kind", "index", "bytes"}, line_number)
            kind = fields["kind"]
            if kind not in buffer_sizes:
                raise ProfileError(f"line {line_number}: invalid buffer kind")
            index = _int(fields["index"], f"line {line_number}.index", non_negative=True)
            if index in buffer_sizes[kind]:
                raise ProfileError(f"line {line_number}: duplicate {kind} buffer {index}")
            size = _int(fields["bytes"], f"line {line_number}.bytes", non_negative=True)
            if size == 0:
                raise ProfileError(f"line {line_number}: zero-sized buffer")
            buffer_sizes[kind][index] = size
        elif line.startswith("iteration,"):
            fields = _fields(line, "iteration", line_number)
            _require(fields, {
                "index", "kind", "h2d_us", "run_us", "d2h_us",
                "device_profile_status", "device_us", "cycles", "repeat_equal",
            }, line_number)
            index = _int(fields["index"], f"line {line_number}.index", non_negative=True)
            if index != len(iterations):
                raise ProfileError(f"line {line_number}: iteration indices must be contiguous")
            expected_kind = "first" if index == 0 else "steady"
            if fields["kind"] != expected_kind:
                raise ProfileError(f"line {line_number}: invalid iteration kind")
            profile_status = _int(fields["device_profile_status"],
                                  f"line {line_number}.device_profile_status")
            if profile_status != 0:
                raise ProfileError(f"line {line_number}: device profiling failed: {profile_status}")
            repeat_equal = _int(fields["repeat_equal"], f"line {line_number}.repeat_equal")
            if repeat_equal not in (0, 1):
                raise ProfileError(f"line {line_number}: repeat_equal must be 0 or 1")
            h2d = _float(fields["h2d_us"], f"line {line_number}.h2d_us")
            run = _float(fields["run_us"], f"line {line_number}.run_us")
            d2h = _float(fields["d2h_us"], f"line {line_number}.d2h_us")
            device = float(_int(fields["device_us"], f"line {line_number}.device_us",
                                non_negative=True))
            if run < device:
                raise ProfileError(f"line {line_number}: host run time is below device time")
            iterations.append({
                "kind": expected_kind,
                "h2d": h2d,
                "run": run,
                "d2h": d2h,
                "device": device,
                "cycles": float(_int(fields["cycles"], f"line {line_number}.cycles",
                                      non_negative=True)),
                "repeat_equal": repeat_equal,
            })

    required_phases = {"init", "create_network", "create_buffers", "prepare"}
    if driver is None or declared_counts is None or set(setup) != required_phases or not iterations:
        raise ProfileError("runner log is missing driver, setup, network, or iteration evidence")
    for kind, expected in zip(("input", "output"), declared_counts):
        required_indices = set(range(expected))
        if set(tensors[kind]) != required_indices or set(buffer_sizes[kind]) != required_indices:
            raise ProfileError(f"{kind} tensor/buffer metadata is incomplete")
        for index in required_indices:
            tensors[kind][index]["bytes"] = buffer_sizes[kind][index]

    grouped = {
        "first": [row for row in iterations if row["kind"] == "first"],
        "steady": [row for row in iterations if row["kind"] == "steady"],
    }
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for kind, rows in grouped.items():
        stats[kind] = {}
        derived = {
            "h2d": [row["h2d"] for row in rows],
            "run": [row["run"] for row in rows],
            "d2h": [row["d2h"] for row in rows],
            "device": [row["device"] for row in rows],
            "cycles": [row["cycles"] for row in rows],
            "run_minus_device": [row["run"] - row["device"] for row in rows],
            "end_to_end": [row["h2d"] + row["run"] + row["d2h"] for row in rows],
            "end_to_end_minus_device": [
                row["h2d"] + row["run"] + row["d2h"] - row["device"] for row in rows
            ],
        }
        for name, values in derived.items():
            stats[kind][name] = _stats(values)

    h2d_bytes = sum(buffer_sizes["input"].values())
    d2h_bytes = sum(buffer_sizes["output"].values())
    bandwidth: dict[str, dict[str, float | None]] = {}
    for kind in ("first", "steady"):
        bandwidth[kind] = {
            "h2d_at_median": h2d_bytes / stats[kind]["h2d"]["median"] / 1000.0
            if stats[kind]["h2d"] else None,
            "d2h_at_median": d2h_bytes / stats[kind]["d2h"]["median"] / 1000.0
            if stats[kind]["d2h"] else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "performance-observed-unqualified",
        "run_id": run_id,
        "driver_software": driver,
        "source_stdout_sha256": source_hash,
        "iterations": {
            "first": len(grouped["first"]),
            "steady": len(grouped["steady"]),
            "total": len(iterations),
        },
        "bytes": {"h2d": h2d_bytes, "d2h": d2h_bytes},
        "setup_us": setup,
        "inputs": [tensors["input"][index] for index in sorted(tensors["input"])],
        "outputs": [tensors["output"][index] for index in sorted(tensors["output"])],
        "stats_us": stats,
        "bandwidth_gbps": bandwidth,
        "device_profile_failures": 0,
        "correctness": {
            "golden_checked": False,
            "repeat_equal_failures": sum(1 for row in iterations if not row["repeat_equal"]),
            "repeat_equal_is_not_golden": True,
        },
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ProfileError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProfileError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot parse JSONL evidence: {path}") from error
    return rows


def merge_thermal(result: dict[str, Any], guard_path: Path, telemetry_path: Path) -> None:
    samples = [row for row in _jsonl(guard_path) if row.get("kind") == "sample"]
    if not samples:
        raise ProfileError("thermal guard contains no samples")
    temperatures: list[int] = []
    fan_states: list[int] = []
    cooling_states: list[int] = []
    sample_times: list[int] = []
    for expected_seq, sample in enumerate(samples):
        if sample.get("sample_seq") != expected_seq:
            raise ProfileError("thermal sample_seq must be contiguous")
        monotonic_ns = sample.get("monotonic_ns")
        if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
            raise ProfileError("thermal sample has invalid monotonic_ns")
        if sample_times and monotonic_ns < sample_times[-1]:
            raise ProfileError("thermal sample timestamps must be monotonic")
        sample_times.append(monotonic_ns)
        zones = sample.get("thermal_zones")
        cooling = sample.get("cooling_devices")
        if not isinstance(zones, list) or not isinstance(cooling, list):
            raise ProfileError("thermal sample is missing zone or cooling arrays")
        npu = [zone for zone in zones if isinstance(zone, dict)
               and zone.get("type") == "npu_thermal_zone"]
        fans = [device for device in cooling if isinstance(device, dict)
                and device.get("type") == "pwm-fan"]
        npu_cooling = [device for device in cooling if isinstance(device, dict)
                       and device.get("type") == "devfreq-3600000.npu"]
        if len(npu) != 1 or len(fans) != 1 or len(npu_cooling) != 1:
            raise ProfileError("thermal sample has ambiguous NPU/fan cooling evidence")
        for target, value, context in (
            (temperatures, npu[0].get("millidegrees_c"), "NPU temperature"),
            (fan_states, fans[0].get("cur_state"), "fan state"),
            (cooling_states, npu_cooling[0].get("cur_state"), "NPU cooling state"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProfileError(f"thermal sample has invalid {context}")
            target.append(value)

    frequencies: list[int] = []
    for row in _jsonl(telemetry_path):
        timestamp = row.get("monotonic_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ProfileError("telemetry has invalid monotonic_ns")
        if timestamp < sample_times[0] or timestamp > sample_times[-1]:
            continue
        system = row.get("system")
        if not isinstance(system, dict):
            continue
        observed = system.get("npu_frequencies")
        if not isinstance(observed, dict):
            continue
        for name, value in observed.items():
            if not isinstance(name, str) or not name.endswith(".npu_hz"):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProfileError("telemetry has invalid NPU frequency")
            frequencies.append(value)
    if not frequencies:
        raise ProfileError("telemetry contains no NPU frequency samples")

    result["thermal"] = {
        "sample_count": len(samples),
        "npu_start_c": temperatures[0] / 1000.0,
        "npu_peak_c": max(temperatures) / 1000.0,
        "npu_end_c": temperatures[-1] / 1000.0,
        "npu_clock_min_hz": min(frequencies),
        "npu_clock_max_hz": max(frequencies),
        "pwm_fan_min_state": min(fan_states),
        "npu_cooling_max_state": max(cooling_states),
        "clock_variation_evidence": min(frequencies) != max(frequencies),
        "throttling_evidence": max(cooling_states) > 0,
    }
    result["thermal_evidence_sha256"] = {
        guard_path.name: hashlib.sha256(guard_path.read_bytes()).hexdigest(),
        telemetry_path.name: hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a phase-aware VIPLite runner log")
    parser.add_argument("stdout_log", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--thermal-guard", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, help="exclusive output path; stdout by default")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = summarize(args.stdout_log, args.run_id or args.stdout_log.parent.name)
        if (args.thermal_guard is None) != (args.telemetry is None):
            raise ProfileError("--thermal-guard and --telemetry must be supplied together")
        if args.thermal_guard is not None and args.telemetry is not None:
            merge_thermal(result, args.thermal_guard, args.telemetry)
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except FileExistsError as error:
        print(f"summarize_viplite_profile: refusing to overwrite: {error.filename}", file=sys.stderr)
        return 2
    except ProfileError as error:
        print(f"summarize_viplite_profile: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
