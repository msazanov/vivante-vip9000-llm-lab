"""Fail-closed, mock-testable A733 DDR PLL A/B guard.

The default CLI operation is read-only inspection.  The apply transaction is
deliberately narrow: it accepts only the current full word ``0xf9005400`` and
the exact approved candidate ``0x54 -> 0x63``.  No board command is executed by the
tests or by this implementation task.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import json
import os
from pathlib import Path
import mmap
import re
import signal
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Mapping


PLL_DDR_CTRL = 0x02002020
DRAM_CLK_REG = 0x02002C00
CCU_BASE = 0x02002000
DEFAULT_LOCK_PATH = Path("/run/lock/a733-ddr-pll-guard.lock")
EXPECTED_ORIGINAL = 0xF9005400
TARGET_RAW = 0x63
CURRENT_RAW = 0x54
RAW_SHIFT = 8
RAW_MASK = 0xFF << RAW_SHIFT
LOCK_BIT = 28
LDO_BIT = 30
OUTPUT_BIT = 27
ENABLE_BIT = 31
REQUIRED_BITS = (1 << LOCK_BIT) | (1 << LDO_BIT) | (1 << OUTPUT_BIT) | (1 << ENABLE_BIT)
SCHEMA_VERSION = "a733-ddr-pll-ab/v1"
CONFIRMATION = "APPLY-A733-DDR-510-TO-600"
DEFAULT_TEMPERATURE_LIMIT_MC = 85000
DEFAULT_SAMPLE_INTERVAL_S = 0.250
WDIOC_SETTIMEOUT = 0xC0045706
WDIOC_GETSUPPORT = 0x80285700
WDIOC_KEEPALIVE = 0x80045705
WDIOF_MAGICCLOSE = 0x0100
WDIOF_KEEPALIVEPING = 0x8000
ALLOWED_EXECUTABLES = frozenset({"taskset", "mbw", "memtester", "q1_cpu_operator_runner", "llama-bench"})
PINNED_EXECUTABLE_PATHS = frozenset({
    "/usr/bin/taskset",
    "/usr/bin/mbw",
    "/usr/local/bin/memtester",
    "/usr/local/bin/q1_cpu_operator_runner",
    "/usr/local/bin/llama-bench",
})
SHELL_NAMES = frozenset({"sh", "bash", "dash", "zsh", "fish", "env"})


class SafetyError(Exception):
    """Raised when a requested operation is outside the immutable policy."""


class SafetyAbort(RuntimeError):
    """Raised when a live safety condition requires restoration."""


def validate_watchdog_options(options: int) -> bool:
    required = WDIOF_MAGICCLOSE | WDIOF_KEEPALIVEPING
    if options & required != required:
        raise SafetyError("watchdog must advertise MAGICCLOSE and KEEPALIVEPING")
    return True


def raw_n(word: int) -> int:
    return (word >> RAW_SHIFT) & 0xFF


def validate_address(address: int, *, writable: bool) -> None:
    if address == PLL_DDR_CTRL:
        return
    if address == DRAM_CLK_REG and not writable:
        return
    raise SafetyError(f"address 0x{address:x} is not allowed for this operation")


def register_offset(address: int) -> int:
    if address not in (PLL_DDR_CTRL, DRAM_CLK_REG):
        raise SafetyError(f"address 0x{address:x} is not whitelisted")
    return address - CCU_BASE


def _validate_word_bits(word: int) -> None:
    if word & REQUIRED_BITS != REQUIRED_BITS:
        raise SafetyError(f"PLL word lacks required enable/lock bits: 0x{word:08x}")


def compute_target_word(original_word: int, target_raw: int) -> int:
    if target_raw != TARGET_RAW:
        raise SafetyError("this experiment has only the approved raw target 0x63")
    _validate_word_bits(original_word)
    return (original_word & ~RAW_MASK) | (target_raw << RAW_SHIFT)


def poll_lock(
    backend: RegisterBackend,
    expected_word: int,
    *,
    timeout_s: float = 2.0,
    interval_s: float = 0.050,
) -> int:
    if timeout_s <= 0 or interval_s <= 0:
        raise SafetyError("lock polling bounds must be positive")
    deadline = time.monotonic() + timeout_s
    while True:
        value = backend.read32(PLL_DDR_CTRL)
        if value == expected_word and value & (1 << LOCK_BIT):
            return value
        if not value & (1 << LOCK_BIT):
            raise SafetyAbort("PLL lock lost during bounded polling")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SafetyAbort("PLL lock/readback polling timeout")
        time.sleep(min(interval_s, remaining))


def parse_clk_summary(summary: str | None) -> dict[str, int]:
    if not isinstance(summary, str):
        raise SafetyAbort("clk_summary missing")
    values: dict[str, int] = {}
    names = ("pll-ddr", "dram0")
    for index, name in enumerate(names):
        start = summary.find(name)
        if start < 0:
            raise SafetyAbort(f"clk_summary missing {name}")
        end = min((summary.find(other, start + len(name)) for other in names if other != name and summary.find(other, start + len(name)) >= 0), default=len(summary))
        numbers = [int(item) for item in re.findall(r"\b\d{6,}\b", summary[start:end])]
        if not numbers:
            raise SafetyAbort(f"clk_summary has no rate for {name}")
        values[name] = max(numbers)
    return values


EXPECTED_CLOCKS = {
    EXPECTED_ORIGINAL: {"pll-ddr": 2040000000, "dram0": 510000000},
    0xF9006300: {"pll-ddr": 2400000000, "dram0": 600000000},
}


def validate_apply(
    original_word: int,
    *,
    target_raw: int = TARGET_RAW,
    explicit_apply: bool,
    expected_original: int,
    confirmation: str,
) -> int:
    if not explicit_apply:
        raise SafetyError("apply requires --apply")
    if expected_original != EXPECTED_ORIGINAL or original_word != EXPECTED_ORIGINAL:
        raise SafetyError("full original PLL word must be exactly 0xf9005400")
    if target_raw != TARGET_RAW or raw_n(original_word) != CURRENT_RAW:
        raise SafetyError("approved experiment permits only raw 0x54 -> 0x63")
    if confirmation != CONFIRMATION:
        raise SafetyError(f"confirmation must equal {CONFIRMATION!r}")
    return compute_target_word(original_word, target_raw)


class RegisterBackend:
    def read32(self, address: int) -> int:
        raise NotImplementedError

    def write32(self, address: int, value: int) -> None:
        raise NotImplementedError


class MockBackend(RegisterBackend):
    def __init__(self, registers: Mapping[int, int]):
        self.registers = dict(registers)
        self.writes: list[int] = []

    def read32(self, address: int) -> int:
        validate_address(address, writable=False)
        return int(self.registers[address])

    def write32(self, address: int, value: int) -> None:
        validate_address(address, writable=True)
        if address != PLL_DDR_CTRL:
            raise SafetyError("mock refuses writes outside PLL_DDR_CTRL")
        self.registers[address] = value & 0xFFFFFFFF
        self.writes.append(value & 0xFFFFFFFF)


class DevMemBackend(RegisterBackend):
    """Small explicit backend; construction is read-only unless requested."""

    def __init__(self, path: str = "/dev/mem", *, read_only: bool = True):
        flags = os.O_RDONLY if read_only else os.O_RDWR | os.O_SYNC
        self._fd = os.open(path, flags | getattr(os, "O_CLOEXEC", 0))
        self.read_only = read_only
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._map_base = CCU_BASE & ~(self._page_size - 1)
        self._mapping = mmap.mmap(
            self._fd,
            self._page_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | (0 if read_only else mmap.PROT_WRITE),
            offset=self._map_base,
        )

    def close(self) -> None:
        try:
            self._mapping.close()
        finally:
            os.close(self._fd)

    def read32(self, address: int) -> int:
        validate_address(address, writable=False)
        return struct.unpack_from("<I", self._mapping, register_offset(address))[0]

    def write32(self, address: int, value: int) -> None:
        if self.read_only:
            raise SafetyError("/dev/mem backend is read-only")
        validate_address(address, writable=True)
        struct.pack_into("<I", self._mapping, register_offset(address), value & 0xFFFFFFFF)
        # Do not call mmap.flush() for MMIO: some kernels reject msync on a
        # device mapping after the store.  The immediate readback and bounded
        # poll_lock are the ordering and acceptance gates.


class EventLog:
    """One-record-per-write JSONL with a sibling-process-safe append."""

    def __init__(self, path: Path, run_id: str, host: str | None = None):
        self.path = Path(path)
        self.run_id = run_id
        self.host = host or socket.gethostname()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, *, phase: str | None = None, **fields: Any) -> None:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "host": self.host,
            "event": event,
            "wall_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
        }
        if phase is not None:
            record["phase"] = phase
        record.update(fields)
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class RunLock:
    """Exclusive process lock held for the complete apply transaction."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._fd)
            self._fd = None
            raise SafetyError(f"another A733 DDR run holds {self.path}") from exc

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()


class SystemSampler:
    """Read-only target sampler for thermal, CPU, clock and register state."""

    def __init__(self, backend: RegisterBackend, *, sys_root: Path = Path("/sys"), proc_root: Path = Path("/proc")):
        self.backend = backend
        self.sys_root = Path(sys_root)
        self.proc_root = Path(proc_root)

    @staticmethod
    def _text(path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except (OSError, UnicodeError):
            return None

    def sample(self, phase: str) -> dict[str, Any]:
        thermal: list[dict[str, Any]] = []
        thermal_root = self.sys_root / "class" / "thermal"
        for path in sorted(thermal_root.glob("thermal_zone*")):
            if not path.is_dir():
                continue
            kind = self._text(path / "type") or path.name
            value = self._text(path / "temp")
            if value is None:
                continue
            try:
                thermal.append({"path": str(path), "type": kind, "temp_mC": int(value)})
            except ValueError:
                continue

        cpufreq: list[dict[str, Any]] = []
        cpu_root = self.sys_root / "devices" / "system" / "cpu" / "cpufreq"
        for path in sorted(cpu_root.glob("policy*")):
            if not path.is_dir():
                continue
            row: dict[str, Any] = {"policy": path.name}
            for name in ("scaling_cur_freq", "scaling_max_freq", "scaling_governor", "affected_cpus"):
                value = self._text(path / name)
                if value is not None:
                    row[name] = value
            cpufreq.append(row)

        clk_summary = self._text(self.sys_root / "kernel" / "debug" / "clk" / "clk_summary")
        npu_devfreq: list[dict[str, Any]] = []
        for path in sorted((self.sys_root / "class" / "devfreq").glob("*")):
            if not path.is_dir():
                continue
            row = {"name": path.name}
            for name in ("cur_freq", "available_frequencies", "governor"):
                value = self._text(path / name)
                if value is not None:
                    row[name] = value
            npu_devfreq.append(row)

        cooling_devices: list[dict[str, Any]] = []
        for path in sorted(thermal_root.glob("cooling_device*")):
            if not path.is_dir():
                continue
            row = {"name": path.name}
            for name in ("type", "cur_state", "max_state"):
                value = self._text(path / name)
                if value is not None:
                    row[name] = value
            cooling_devices.append(row)

        memory: dict[str, int] = {}
        meminfo = self._text(self.proc_root / "meminfo") or ""
        for line in meminfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in ("MemAvailable", "SwapFree"):
                match = re.search(r"\d+", value)
                if match:
                    memory_key = "mem_available_kB" if key == "MemAvailable" else "swap_free_kB"
                    memory[memory_key] = int(match.group())
        return {
            "phase": phase,
            "thermal_zones": thermal,
            "cpufreq": cpufreq,
            "clk_summary": clk_summary,
            "npu_devfreq": npu_devfreq,
            "cooling_devices": cooling_devices,
            "memory": memory,
            "registers": {
                "pll_ddr_ctrl": self.backend.read32(PLL_DDR_CTRL),
                "dram_clk_reg": self.backend.read32(DRAM_CLK_REG),
            },
        }


class WatchdogDevice:
    """Linux watchdog with bounded timeout and magic-close on clean success."""

    def __init__(self, path: str = "/dev/watchdog", timeout_s: int = 15):
        self.path = path
        self.timeout_s = timeout_s
        self._fd: int | None = None

    def arm(self, timeout_s: int | None = None) -> int:
        if self._fd is not None:
            raise SafetyError("watchdog already armed")
        self._fd = os.open(self.path, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
        import struct

        requested = int(timeout_s or self.timeout_s)
        support = bytearray(40)
        try:
            fcntl.ioctl(self._fd, WDIOC_GETSUPPORT, support, True)
            options = struct.unpack_from("I", support)[0]
            validate_watchdog_options(options)
            timeout_buffer = bytearray(struct.pack("I", requested))
            fcntl.ioctl(self._fd, WDIOC_SETTIMEOUT, timeout_buffer, True)
            self.actual_timeout_s = struct.unpack_from("I", timeout_buffer)[0]
            return self.actual_timeout_s
        except Exception:
            os.close(self._fd)
            self._fd = None
            raise

    def keepalive(self) -> None:
        if self._fd is None:
            raise SafetyError("watchdog is not armed")
        fcntl.ioctl(self._fd, WDIOC_KEEPALIVE, 0)

    def disarm(self) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, b"V")
        finally:
            os.close(self._fd)
            self._fd = None


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1.0)


def run_command(
    argv: list[str],
    timeout_s: float,
    abort_event: threading.Event,
    event_log: EventLog,
) -> dict[str, Any]:
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise SafetyError("workload argv must be a non-empty list of strings")
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        shell=False,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    termination_reason: str | None = None
    while True:
        if abort_event.is_set():
            termination_reason = "safety monitor requested workload termination"
            _kill_process_group(process)
            break
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            termination_reason = "workload timeout"
            _kill_process_group(process)
            break
        try:
            stdout, stderr = process.communicate(timeout=min(0.25, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if termination_reason is not None:
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            stdout, stderr = process.communicate()
    result = {
        "argv": list(argv),
        "stdout": stdout,
        "stderr": stderr,
        "returncode": process.returncode,
        "elapsed_s": time.monotonic() - started,
    }
    event_log.append("workload", phase="during", **result)
    if termination_reason is not None:
        raise SafetyAbort(termination_reason)
    return result


def build_workloads() -> list[list[str]]:
    return [
        ["/usr/bin/taskset", "-c", "0-5", "/usr/bin/mbw", "-n", "10", "256"],
        ["/usr/bin/taskset", "-c", "0-5", "/usr/local/bin/memtester", "128M", "1"],
    ]


def load_workload_specs(paths: list[Path]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyError(f"cannot read workload JSON {path}: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"name", "argv", "timeout_s"}:
            raise SafetyError("workload JSON must contain exactly name, argv, timeout_s")
        name = payload["name"]
        argv = payload["argv"]
        timeout_s = payload["timeout_s"]
        if not isinstance(name, str) or not name:
            raise SafetyError("workload name must be non-empty string")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
            or not os.path.isabs(argv[0])
            or argv[0] not in PINNED_EXECUTABLE_PATHS
        ):
            raise SafetyError("workload argv[0] must be an absolute allowlisted executable")
        executable = Path(argv[0]).name
        if any(Path(value).name in SHELL_NAMES for value in argv):
            raise SafetyError("shell interpreters are forbidden in workload argv")
        if executable == "taskset":
            if len(argv) < 4 or argv[1:3] != ["-c", "0-5"]:
                raise SafetyError("taskset workload must pin exactly CPUs 0-5")
            child = Path(argv[3])
            if str(child) not in (PINNED_EXECUTABLE_PATHS - {"/usr/bin/taskset"}):
                raise SafetyError("taskset child must be an absolute allowlisted executable")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 3600:
            raise SafetyError("workload timeout_s must be in (0, 3600]")
        specs.append({"name": name, "argv": argv, "timeout_s": float(timeout_s)})
    return specs


class GuardRunner:
    def __init__(
        self,
        *,
        backend: RegisterBackend,
        event_log: EventLog,
        sampler: Any | None = None,
        watchdog: Any | None = None,
        command_runner: Callable[..., dict[str, Any]] = run_command,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        temperature_limit_mC: int = DEFAULT_TEMPERATURE_LIMIT_MC,
        workload_timeout_s: float = 60.0,
        lock_path: Path | None = None,
    ):
        self.backend = backend
        self.event_log = event_log
        self.sampler = sampler or SystemSampler(backend)
        self.watchdog = watchdog
        self.command_runner = command_runner
        self.sample_interval_s = sample_interval_s
        self.temperature_limit_mC = temperature_limit_mC
        self.workload_timeout_s = workload_timeout_s
        self.lock_path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
        self.abort_event = threading.Event()
        self._abort_reason: str | None = None

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self._abort_reason = f"signal {signum}"
        self.abort_event.set()

    def _registers(self) -> dict[str, int]:
        return {
            "pll_ddr_ctrl": self.backend.read32(PLL_DDR_CTRL),
            "dram_clk_reg": self.backend.read32(DRAM_CLK_REG),
        }

    def inspect(self) -> dict[str, int]:
        registers = self._registers()
        self.event_log.append("inspect", phase="before", registers=registers)
        return {"pll_ddr_ctrl": registers["pll_ddr_ctrl"], "dram_clk_reg": registers["dram_clk_reg"]}

    def _check_sample(self, observation: Mapping[str, Any], expected_word: int, baseline_dram: int) -> None:
        if not observation.get("thermal_zones"):
            raise SafetyAbort("required thermal telemetry missing")
        if not observation.get("cpufreq"):
            raise SafetyAbort("required cpufreq telemetry missing")
        if not observation.get("clk_summary"):
            raise SafetyAbort("required clk_summary telemetry missing")
        if not observation.get("npu_devfreq"):
            raise SafetyAbort("required NPU devfreq telemetry missing")
        if not observation.get("cooling_devices"):
            raise SafetyAbort("required cooling telemetry missing")
        memory = observation.get("memory")
        if not isinstance(memory, Mapping) or not all(key in memory for key in ("mem_available_kB", "swap_free_kB")):
            raise SafetyAbort("required memory/swap preflight telemetry missing")
        clocks = parse_clk_summary(observation.get("clk_summary"))
        expected_clocks = EXPECTED_CLOCKS.get(expected_word)
        if expected_clocks is None or clocks != expected_clocks:
            raise SafetyAbort(f"unexpected DDR clock summary: {clocks!r}")
        for zone in observation.get("thermal_zones", []):
            value = zone.get("temp_mC")
            if isinstance(value, (int, float)) and value >= self.temperature_limit_mC:
                raise SafetyAbort(f"temperature limit reached: {zone.get('type')}={value}mC")
        registers = observation.get("registers", {})
        pll = registers.get("pll_ddr_ctrl")
        dram = registers.get("dram_clk_reg")
        if pll != expected_word:
            if isinstance(pll, int) and not (pll & (1 << LOCK_BIT)):
                raise SafetyAbort("PLL lock lost")
            raise SafetyAbort("PLL register drift")
        if dram != baseline_dram:
            raise SafetyAbort("DRAM clock register drift")

    def _keepalive(self) -> None:
        if self.watchdog is None or not hasattr(self.watchdog, "keepalive"):
            return
        try:
            self.watchdog.keepalive()
        except Exception as exc:
            raise SafetyAbort(f"watchdog heartbeat failed: {exc}") from exc

    def _sample_and_check(self, phase: str, expected_word: int, baseline_dram: int) -> None:
        observation = self.sampler.sample(phase)
        self.event_log.append("sample", **observation)
        self._keepalive()
        self._check_sample(observation, expected_word, baseline_dram)

    def _monitor(self, stop_event: threading.Event, expected_word: int, baseline_dram: int) -> None:
        while not stop_event.is_set() and not self.abort_event.is_set():
            try:
                self._sample_and_check("during", expected_word, baseline_dram)
            except SafetyAbort as exc:
                self._abort_reason = str(exc)
                self.abort_event.set()
                return
            except Exception as exc:  # fail closed on sampler failure
                self._abort_reason = f"sampler failure: {exc}"
                self.abort_event.set()
                return
            stop_event.wait(self.sample_interval_s)

    def apply(
        self,
        *,
        explicit_apply: bool,
        expected_original: int,
        target_raw: int,
        confirmation: str,
        workloads: list[Any] | None = None,
    ) -> dict[str, Any]:
        with RunLock(self.lock_path):
            return self._apply_locked(
                explicit_apply=explicit_apply,
                expected_original=expected_original,
                target_raw=target_raw,
                confirmation=confirmation,
                workloads=workloads,
            )

    def _apply_locked(
        self,
        *,
        explicit_apply: bool,
        expected_original: int,
        target_raw: int,
        confirmation: str,
        workloads: list[Any] | None = None,
    ) -> dict[str, Any]:
        original = self.backend.read32(PLL_DDR_CTRL)
        baseline_dram = self.backend.read32(DRAM_CLK_REG)
        target = validate_apply(
            original,
            target_raw=target_raw,
            explicit_apply=explicit_apply,
            expected_original=expected_original,
            confirmation=confirmation,
        )
        if self.watchdog is None:
            raise SafetyError("apply requires a watchdog backend")
        self.event_log.append("start", phase="before", original=original, target=target, baseline_dram=baseline_dram)
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        stop_monitor = threading.Event()
        monitor_thread: threading.Thread | None = None
        restore_needed = False
        success = False
        try:
            if self.abort_event.is_set():
                raise SafetyAbort(self._abort_reason or "abort requested before apply")
            actual_watchdog_timeout = self.watchdog.arm(15)
            self.event_log.append(
                "watchdog",
                phase="before",
                action="arm",
                requested_timeout_s=15,
                actual_timeout_s=actual_watchdog_timeout or 15,
            )
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            if hasattr(self.sampler, "target_word"):
                self.sampler.target_word = original
            self._sample_and_check("before", original, baseline_dram)
            restore_needed = True
            self.backend.write32(PLL_DDR_CTRL, target)
            self.event_log.append("write", phase="during", address=PLL_DDR_CTRL, value=target)
            poll_lock(self.backend, target)
            if hasattr(self.sampler, "target_word"):
                self.sampler.target_word = target
            self._sample_and_check("during", target, baseline_dram)
            monitor_thread = threading.Thread(
                target=self._monitor,
                args=(stop_monitor, target, baseline_dram),
                daemon=True,
            )
            monitor_thread.start()
            selected_workloads = workloads if workloads is not None else build_workloads()
            for workload in selected_workloads:
                if isinstance(workload, dict):
                    argv = workload["argv"]
                    timeout_s = workload["timeout_s"]
                    workload_name = workload["name"]
                else:
                    argv = workload
                    timeout_s = self.workload_timeout_s
                    workload_name = Path(argv[0]).name
                if self.abort_event.is_set():
                    raise SafetyAbort(self._abort_reason or "safety monitor abort")
                result = self.command_runner(argv, timeout_s, self.abort_event, self.event_log)
                self.event_log.append("workload_spec", phase="during", name=workload_name, timeout_s=timeout_s, argv=argv)
                if result.get("returncode") != 0:
                    raise SafetyAbort(f"workload failed rc={result.get('returncode')}")
            if self.abort_event.is_set():
                raise SafetyAbort(self._abort_reason or "safety monitor abort")
            self._sample_and_check("after", target, baseline_dram)
            success = True
            return {"status": "restored", "original": original, "target": target}
        except SafetyAbort as exc:
            self._abort_reason = str(exc)
            self.event_log.append("abort", phase="during", reason=self._abort_reason)
            raise
        except Exception as exc:
            self._abort_reason = str(exc)
            self.event_log.append("failure", phase="during", reason=self._abort_reason)
            raise SafetyAbort(self._abort_reason) from exc
        finally:
            stop_monitor.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=max(1.0, self.sample_interval_s * 4))
            restore_error: Exception | None = None
            restored: int | None = None
            try:
                if restore_needed:
                    self.backend.write32(PLL_DDR_CTRL, original)
                    poll_lock(self.backend, original)
                    restored = self.backend.read32(PLL_DDR_CTRL)
                    if hasattr(self.sampler, "target_word"):
                        self.sampler.target_word = original
                    self._sample_and_check("restore", original, baseline_dram)
                else:
                    restored = self.backend.read32(PLL_DDR_CTRL)
            except Exception as exc:
                restore_error = exc
            self.event_log.append("restore", phase="restore", value=restored, expected=original)
            if restored != original and restore_error is None:
                restore_error = SafetyAbort("restore readback mismatch")
            signal.signal(signal.SIGINT, old_handlers[signal.SIGINT])
            signal.signal(signal.SIGTERM, old_handlers[signal.SIGTERM])
            if restore_error is not None:
                self.event_log.append("failure", phase="restore", reason=str(restore_error))
                raise SafetyAbort(f"restore failed: {restore_error}") from restore_error
            if success:
                self.watchdog.disarm()
                self.event_log.append("watchdog", phase="after", action="magic_close")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded A733 DDR PLL inspect/apply runner")
    parser.add_argument("command", choices=("inspect", "dry-run", "apply"))
    parser.add_argument("--run-id", default="E005-a733-ddr-pll-ab")
    parser.add_argument("--event-log", type=Path, default=Path("a733-ddr-pll-events.jsonl"))
    parser.add_argument("--dev-mem", default="/dev/mem")
    parser.add_argument("--apply", action="store_true", help="required in addition to the apply subcommand")
    parser.add_argument("--expected-original", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--target-raw", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--confirm", default=None)
    parser.add_argument("--workload-json", action="append", type=Path, default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "apply":
        if not args.apply:
            raise SystemExit("apply requires --apply")
        if args.expected_original is None or args.target_raw is None or args.confirm is None:
            raise SystemExit("apply requires --expected-original, --target-raw and --confirm")
        backend = DevMemBackend(args.dev_mem, read_only=False)
        try:
            runner = GuardRunner(
                backend=backend,
                event_log=EventLog(args.event_log, args.run_id),
                watchdog=WatchdogDevice(),
            )
            workload_specs = build_workloads() + load_workload_specs(args.workload_json)
            runner.apply(
                explicit_apply=args.apply,
                expected_original=args.expected_original,
                target_raw=args.target_raw,
                confirmation=args.confirm,
                workloads=workload_specs,
            )
        finally:
            backend.close()
        return 0
    backend = DevMemBackend(args.dev_mem, read_only=True)
    try:
        runner = GuardRunner(backend=backend, event_log=EventLog(args.event_log, args.run_id))
        result = runner.inspect()
        result["workloads"] = build_workloads() + load_workload_specs(args.workload_json)
        print(json.dumps(result, sort_keys=True))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
