#!/usr/bin/env python3
"""Run one command while retaining raw output and lightweight Linux telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any


SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_meminfo() -> dict[str, int]:
    selected = {"MemAvailable", "MemFree", "Cached", "Buffers", "Dirty", "SwapFree"}
    values: dict[str, int] = {}
    text = read_text(Path("/proc/meminfo"))
    if text is None:
        return values
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in selected:
            continue
        fields = remainder.split()
        if fields and fields[0].isdigit():
            values[f"{key}_kib"] = int(fields[0])
    return values


def read_process(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid, "available": False}
    status = read_text(Path(f"/proc/{pid}/status"))
    if status is not None:
        result["available"] = True
        wanted = {
            "VmRSS": "rss_kib",
            "VmHWM": "rss_hwm_kib",
            "Threads": "threads",
            "voluntary_ctxt_switches": "voluntary_context_switches",
            "nonvoluntary_ctxt_switches": "nonvoluntary_context_switches",
        }
        for line in status.splitlines():
            key, separator, remainder = line.partition(":")
            if not separator or key not in wanted:
                continue
            fields = remainder.split()
            if fields:
                try:
                    result[wanted[key]] = int(fields[0])
                except ValueError:
                    pass

    stat = read_text(Path(f"/proc/{pid}/stat"))
    if stat is not None and ")" in stat:
        fields = stat.rsplit(")", 1)[1].strip().split()
        if len(fields) > 12:
            try:
                result["user_ticks"] = int(fields[11])
                result["system_ticks"] = int(fields[12])
            except ValueError:
                pass
    return result


def read_cpu_frequencies() -> dict[str, int]:
    frequencies: dict[str, int] = {}
    root = Path("/sys/devices/system/cpu")
    for cpu_dir in sorted(root.glob("cpu[0-9]*")):
        value = read_int(cpu_dir / "cpufreq" / "scaling_cur_freq")
        if value is not None:
            frequencies[f"{cpu_dir.name}_khz"] = value
    return frequencies


def read_npu_frequencies() -> dict[str, int]:
    frequencies: dict[str, int] = {}
    root = Path("/sys/class/devfreq")
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return frequencies
    for entry in entries:
        if "npu" not in entry.name.lower():
            continue
        value = read_int(entry / "cur_freq")
        if value is not None:
            frequencies[f"{entry.name}_hz"] = value
    return frequencies


def read_thermals() -> dict[str, dict[str, Any]]:
    zones: dict[str, dict[str, Any]] = {}
    root = Path("/sys/class/thermal")
    for entry in sorted(root.glob("thermal_zone*")):
        zone: dict[str, Any] = {}
        zone_type = read_text(entry / "type")
        temperature = read_int(entry / "temp")
        if zone_type is not None:
            zone["type"] = zone_type
        if temperature is not None:
            zone["millidegrees_c"] = temperature
        if zone:
            zones[entry.name] = zone
    return zones


def snapshot(pid: int) -> dict[str, Any]:
    def optional(read: Any, default: Any) -> Any:
        try:
            return read()
        except Exception:
            return default

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "process": optional(lambda: read_process(pid), {"pid": pid, "available": False}),
        "system": {
            "memory": optional(read_meminfo, {}),
            "cpu_frequencies": optional(read_cpu_frequencies, {}),
            "npu_frequencies": optional(read_npu_frequencies, {}),
            "thermal_zones": optional(read_thermals, {}),
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a command and retain raw Linux telemetry.",
    epilog="Example: profile_command.py --output-dir results/run-001 -- llama-bench -m model.gguf",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--label", default="")
    parser.add_argument("--phase-file", default="phases.jsonl")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if args.output_dir.name != args.run_id:
        parser.error("output directory basename must equal --run-id")
    phase_file = Path(args.phase_file)
    if not args.phase_file or phase_file.is_absolute() or phase_file == Path(".") or ".." in phase_file.parts:
        parser.error("--phase-file must be a relative path without '..'")
    if args.interval_ms < 10:
        parser.error("--interval-ms must be at least 10")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"refusing to overwrite existing output directory: {args.output_dir}", file=sys.stderr)
        return 2

    start_utc = datetime.now(timezone.utc).isoformat()
    start_ns = time.monotonic_ns()
    launch_error: str | None = None
    profiler_error: str | None = None
    exit_code = 127
    child_return_code: int | None = None
    sample_count = 0

    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"
    telemetry_path = args.output_dir / "telemetry.jsonl"
    phase_path = args.output_dir / Path(args.phase_file)
    phase_path.parent.mkdir(parents=True, exist_ok=True)

    def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        with telemetry_path.open("w") as telemetry_file, phase_path.open("ab"):
            process: subprocess.Popen[bytes] | None = None
            try:
                child_env = os.environ.copy()
                child_env["VIP9000_PHASE_FILE"] = str(phase_path.resolve())
                try:
                    process = subprocess.Popen(
                        args.command,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        env=child_env,
                        start_new_session=True,
                    )
                except OSError as error:
                    launch_error = f"{type(error).__name__}: {error}"
                    stderr_file.write(launch_error + "\n")
                if process is not None:
                    while True:
                        telemetry_file.write(json.dumps(snapshot(process.pid), sort_keys=True) + "\n")
                        telemetry_file.flush()
                        sample_count += 1
                        polled = process.poll()
                        if polled is not None:
                            child_return_code = polled
                            break
                        time.sleep(args.interval_ms / 1000)
            except Exception as error:
                profiler_error = f"{type(error).__name__}: {error}"
            finally:
                if process is not None:
                    if process.poll() is None:
                        terminate_process_group(process)
                    if child_return_code is None:
                        try:
                            child_return_code = process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            terminate_process_group(process)
                            try:
                                child_return_code = process.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                child_return_code = process.returncode

    if child_return_code is not None:
        exit_code = 128 - child_return_code if child_return_code < 0 else child_return_code
    if profiler_error is not None:
        exit_code = 1

    end_ns = time.monotonic_ns()
    end_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "label": args.label,
        "command": args.command,
        "working_directory": str(Path.cwd()),
        "profiler": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "interval_ms": args.interval_ms,
            "scope": "direct child process plus board-wide readable /proc and /sys telemetry",
        },
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_ns": end_ns - start_ns,
        "exit_code": exit_code,
        "child_return_code": child_return_code,
        "signal": -child_return_code if child_return_code is not None and child_return_code < 0 else None,
        "launch_error": launch_error,
        "profiler_error": profiler_error,
        "telemetry_samples": sample_count,
        "phase_file": {
            "path": phase_path.relative_to(args.output_dir).as_posix(),
            "size_bytes": phase_path.stat().st_size if phase_path.exists() else 0,
        },
        "files": {
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "telemetry": telemetry_path.name,
            "phases": phase_path.relative_to(args.output_dir).as_posix(),
        },
    }
    atomic_json(args.output_dir / "metadata.json", metadata)

    if exit_code < 0:
        return 128 - exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
