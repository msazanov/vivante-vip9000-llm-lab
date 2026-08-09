#!/usr/bin/env python3
"""Fail-safe command wrapper with thermal/cpufreq/cooling telemetry.

The production interface is intentionally small and standard-library-only:
``thermal_exec_guard.py [--limit-mc N] [--interval-ms N] --trace PATH -- CMD``.
The ``sys_root`` constructor argument and ``--sys-root`` CLI option exist only
to make fixture-backed tests possible; production defaults to ``/sys``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable


EXIT_ABORT = 86
EXIT_PREFLIGHT = 87
DEFAULT_LIMIT_MC = 85_000
DEFAULT_INTERVAL_MS = 100
TERM_WAIT_SECONDS = 1.0
KILL_WAIT_SECONDS = 1.0


class GuardFailure(RuntimeError):
    """A fail-closed sensor, trace, or process-control failure."""


@dataclass(frozen=True)
class Inventory:
    zones: tuple[dict[str, Any], ...]
    policies: tuple[dict[str, Any], ...]
    cooling_devices: tuple[dict[str, Any], ...]


def read_process(pid: int) -> dict[str, Any]:
    """Best-effort direct-child telemetry; process exit is not a guard failure."""
    result: dict[str, Any] = {"pid": pid, "available": False}
    try:
        status = Path(f"/proc/{pid}/status").read_text(errors="replace")
    except (OSError, UnicodeError):
        status = ""
    if status:
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

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="replace").strip()
    except (OSError, UnicodeError):
        stat = ""
    if ")" in stat:
        fields = stat.rsplit(")", 1)[1].strip().split()
        if len(fields) > 12:
            try:
                result["user_ticks"] = int(fields[11])
                result["system_ticks"] = int(fields[12])
            except ValueError:
                pass

    try:
        io_text = Path(f"/proc/{pid}/io").read_text(errors="replace")
    except (OSError, UnicodeError):
        io_text = ""
    for line in io_text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in {"rchar", "wchar", "read_bytes", "write_bytes"}:
            continue
        try:
            result[f"io_{key}"] = int(value.strip())
        except ValueError:
            pass
    return result


class SysfsReader:
    def __init__(self, sys_root: Path = Path("/sys")) -> None:
        self.sys_root = sys_root
        self.thermal_root = sys_root / "class" / "thermal"
        self.policy_root = sys_root / "devices" / "system" / "cpu" / "cpufreq"

    @staticmethod
    def _read(path: Path, category: str) -> str:
        try:
            value = path.read_text(errors="replace").strip()
        except (OSError, UnicodeError) as error:
            raise GuardFailure(f"{category}_loss:{path}") from error
        if not value:
            raise GuardFailure(f"{category}_loss:{path}")
        return value

    @classmethod
    def _read_int(cls, path: Path, category: str) -> int:
        value = cls._read(path, category)
        try:
            return int(value)
        except ValueError as error:
            raise GuardFailure(f"{category}_loss:{path}") from error

    @staticmethod
    def _canonical(path: Path) -> str:
        try:
            return str(path.resolve())
        except OSError:
            return str(path.absolute())

    @staticmethod
    def _identity(path: Path, category: str) -> str:
        try:
            stat_result = path.stat()
        except OSError as error:
            raise GuardFailure(f"{category}_loss:{path}") from error
        return f"{stat_result.st_dev}:{stat_result.st_ino}"

    def inventory(self) -> Inventory:
        zones: list[dict[str, Any]] = []
        for temp_path in sorted(self.thermal_root.glob("thermal_zone*/temp")):
            zone_path = temp_path.parent
            zone_type = self._read(zone_path / "type", "sensor")
            self._read_int(temp_path, "sensor")
            zones.append(
                {
                    "path": self._canonical(temp_path),
                    "type": zone_type,
                    "identity": self._identity(zone_path, "sensor"),
                }
            )
        if not zones:
            raise GuardFailure("no_thermal_zones")

        policies: list[dict[str, Any]] = []
        for policy_path in sorted(self.policy_root.glob("policy*")):
            current = self._read_int(policy_path / "scaling_cur_freq", "policy")
            maximum = self._read_int(policy_path / "scaling_max_freq", "policy")
            governor = self._read(policy_path / "scaling_governor", "policy")
            affected_path = policy_path / "affected_cpus"
            affected = self._read(affected_path, "policy") if affected_path.exists() else None
            policies.append(
                {
                    "path": self._canonical(policy_path),
                    "cur_khz": current,
                    "max_khz": maximum,
                    "governor": governor,
                    "affected_cpus": affected,
                    "identity": self._identity(policy_path, "policy"),
                }
            )
        if not policies:
            raise GuardFailure("no_cpu_policies")

        cooling_devices: list[dict[str, Any]] = []
        for cooling_path in sorted(self.thermal_root.glob("cooling_device*")):
            cooling_type = self._read(cooling_path / "type", "cooling")
            current = self._read_int(cooling_path / "cur_state", "cooling")
            maximum = self._read_int(cooling_path / "max_state", "cooling")
            cooling_devices.append(
                {
                    "path": self._canonical(cooling_path),
                    "type": cooling_type,
                    "cur_state": current,
                    "max_state": maximum,
                    "identity": self._identity(cooling_path, "cooling"),
                }
            )
        if not cooling_devices:
            raise GuardFailure("no_cooling_devices")

        return Inventory(tuple(zones), tuple(policies), tuple(cooling_devices))

    def sample(self, inventory: Inventory) -> dict[str, Any]:
        zones: list[dict[str, Any]] = []
        for baseline in inventory.zones:
            temp_path = Path(baseline["path"])
            zone_path = temp_path.parent
            if self._identity(zone_path, "sensor") != baseline["identity"]:
                raise GuardFailure(f"sensor_loss:{zone_path}")
            zone_type = self._read(zone_path / "type", "sensor")
            if zone_type != baseline["type"]:
                raise GuardFailure(f"sensor_loss:{zone_path / 'type'}")
            zones.append(
                {
                    "path": baseline["path"],
                    "type": zone_type,
                    "millidegrees_c": self._read_int(temp_path, "sensor"),
                }
            )

        policies: list[dict[str, Any]] = []
        for baseline in inventory.policies:
            policy_path = Path(baseline["path"])
            if self._identity(policy_path, "policy") != baseline["identity"]:
                raise GuardFailure(f"policy_loss:{policy_path}")
            policies.append(
                {
                    **baseline,
                    "cur_khz": self._read_int(policy_path / "scaling_cur_freq", "policy"),
                    "max_khz": self._read_int(policy_path / "scaling_max_freq", "policy"),
                    "governor": self._read(policy_path / "scaling_governor", "policy"),
                }
            )

        cooling_devices: list[dict[str, Any]] = []
        for baseline in inventory.cooling_devices:
            cooling_path = Path(baseline["path"])
            if self._identity(cooling_path, "cooling") != baseline["identity"]:
                raise GuardFailure(f"cooling_loss:{cooling_path}")
            cooling_type = self._read(cooling_path / "type", "cooling")
            if cooling_type != baseline["type"]:
                raise GuardFailure(f"cooling_loss:{cooling_path / 'type'}")
            cooling_devices.append(
                {
                    **baseline,
                    "type": cooling_type,
                    "cur_state": self._read_int(cooling_path / "cur_state", "cooling"),
                    "max_state": self._read_int(cooling_path / "max_state", "cooling"),
                }
            )

        return {
            "thermal_zones": zones,
            "cpu_policies": policies,
            "cooling_devices": cooling_devices,
        }


class Trace:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("x", encoding="utf-8")
        except OSError as error:
            raise GuardFailure(f"trace_open:{path}") from error

    def write(self, row: dict[str, Any]) -> None:
        try:
            self.stream.write(json.dumps(row, sort_keys=True) + "\n")
            self.stream.flush()
        except (OSError, UnicodeError) as error:
            raise GuardFailure(f"trace_write:{self.path}") from error

    def close(self) -> None:
        self.stream.close()


class Guard:
    def __init__(
        self,
        *,
        trace: Path,
        limit_mc: int = DEFAULT_LIMIT_MC,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        sys_root: Path = Path("/sys"),
        clock: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if limit_mc <= 0:
            raise ValueError("limit_mc must be positive")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.trace_path = trace
        self.limit_mc = limit_mc
        self.interval_ns = interval_ms * 1_000_000
        self.reader = SysfsReader(sys_root)
        self.clock = clock
        self.sleeper = sleeper
        self.signal_number: int | None = None
        self.trace: Trace | None = None

    def emit(self, event: str, **fields: Any) -> None:
        if self.trace is None:
            return
        self.trace.write({"event": event, "monotonic_ns": self.clock(), **fields})

    def safe_emit(self, event: str, **fields: Any) -> None:
        try:
            self.emit(event, **fields)
        except GuardFailure:
            # A trace failure must never prevent process-group cleanup.
            pass

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        self.signal_number = signum

    @staticmethod
    def _status_code(returncode: int | None) -> int:
        if returncode is None:
            return EXIT_ABORT
        return 128 + (-returncode) if returncode < 0 else returncode

    def _terminate_group(self, process: subprocess.Popen[bytes], reason: str) -> None:
        group_id = process.pid

        def group_exists() -> bool:
            try:
                os.killpg(group_id, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        if process.poll() is not None and not group_exists():
            return
        term_sent = False
        kill_sent = False
        try:
            os.killpg(group_id, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=TERM_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        if group_exists():
            try:
                os.killpg(group_id, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        self.safe_emit("terminated", reason=reason, term_sent=term_sent, kill_sent=kill_sent)

    def _sample_loop(self, process: subprocess.Popen[bytes], inventory: Inventory) -> int | None:
        sequence = 0
        deadline = self.clock()
        while True:
            if self.signal_number is not None:
                self.safe_emit("signal", signal=self.signal_number)
                self._terminate_group(process, f"signal:{self.signal_number}")
                return 128 + self.signal_number
            try:
                sample = self.reader.sample(inventory)
                now = self.clock()
                sample.update(
                    {
                        "kind": "sample",
                        "sample_seq": sequence,
                        "monotonic_ns": now,
                        "scheduled_deadline_ns": deadline,
                        "lateness_ns": max(0, now - deadline),
                        "process": read_process(process.pid),
                    }
                )
                self.trace.write(sample)  # type: ignore[union-attr]
                sequence += 1
                peak = max(zone["millidegrees_c"] for zone in sample["thermal_zones"])
                if peak >= self.limit_mc:
                    reason = f"temperature_abort:{peak}"
                    self.safe_emit("abort", reason=reason)
                    self._terminate_group(process, reason)
                    return EXIT_ABORT
            except GuardFailure as error:
                if self.signal_number is not None:
                    reason = f"signal:{self.signal_number}"
                    self.safe_emit("signal", signal=self.signal_number)
                    self._terminate_group(process, reason)
                    return 128 + self.signal_number
                reason = str(error)
                self.safe_emit("abort", reason=reason)
                self._terminate_group(process, reason)
                return EXIT_ABORT

            returncode = process.poll()
            if returncode is not None:
                return self._status_code(returncode)
            deadline += self.interval_ns
            delay_ns = deadline - self.clock()
            if delay_ns > 0:
                try:
                    self.sleeper(delay_ns / 1_000_000_000)
                except InterruptedError:
                    pass

    def run(self, command: Iterable[str]) -> int:
        argv = list(command)
        try:
            self.trace = Trace(self.trace_path)
        except GuardFailure:
            return EXIT_PREFLIGHT
        try:
            self.emit("start", command=argv, limit_mc=self.limit_mc, interval_ms=self.interval_ns // 1_000_000)
            inventory = self.reader.inventory()
            self.emit(
                "inventory",
                thermal_zones=list(inventory.zones),
                cpu_policies=list(inventory.policies),
                cooling_devices=list(inventory.cooling_devices),
            )
        except GuardFailure as error:
            self.safe_emit("abort", reason=str(error), phase="preflight")
            self.safe_emit("exit", status=EXIT_PREFLIGHT)
            try:
                self.trace.close()
            except OSError:
                pass
            return EXIT_PREFLIGHT
        if not argv:
            self.safe_emit("abort", reason="empty_command", phase="preflight")
            self.safe_emit("exit", status=EXIT_PREFLIGHT)
            try:
                self.trace.close()
            except OSError:
                pass
            return EXIT_PREFLIGHT

        process: subprocess.Popen[bytes] | None = None
        previous_handlers: dict[int, Any] = {}
        result = EXIT_ABORT
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
            # No stdout/stderr arguments: the child inherits both streams.
            process = subprocess.Popen(argv, start_new_session=True)
            self.emit("child_started", pid=process.pid)
            result = self._sample_loop(process, inventory)
        except OSError as error:
            self.safe_emit("abort", reason=f"launch:{error}")
            result = EXIT_PREFLIGHT
        except GuardFailure as error:
            self.safe_emit("abort", reason=str(error))
            result = EXIT_ABORT
        finally:
            if process is not None:
                self._terminate_group(process, "cleanup")
                try:
                    returncode = process.wait(timeout=KILL_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    returncode = process.returncode
                if returncode is not None:
                    self.safe_emit("child_exit", returncode=returncode)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self.safe_emit("exit", status=result)
            try:
                self.trace.close()
            except OSError:
                pass
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with fail-safe thermal telemetry")
    parser.add_argument("--limit-mc", type=int, default=DEFAULT_LIMIT_MC)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sys-root", type=Path, default=Path("/sys"), help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.limit_mc <= 0:
        parser.error("--limit-mc must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return Guard(
        trace=args.trace,
        limit_mc=args.limit_mc,
        interval_ms=args.interval_ms,
        sys_root=args.sys_root,
    ).run(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
