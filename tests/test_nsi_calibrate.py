"""Host-only tests for the fail-closed A733 NSI calibration helper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tooling.nsi_calibrate import (
    NSIError,
    SysfsNSI,
    TimerRestoreFailure,
    classify_unit_hypothesis,
    fit_line,
    timer_window,
)


def make_fake_sysfs(root: Path, timer: int = 0) -> Path:
    """Create only the files that the helper is allowed to read/write."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pmu_timer").write_text(f"{timer}\n", encoding="utf-8")
    for name in SysfsNSI.PMU_READ_FILES:
        if name == "available_pmu":
            value = "npu cpu0 cpu1 total\n"
        elif name == "pmu_timer":
            continue
        else:
            value = "1 2 3 4\n"
        (root / name).write_text(value, encoding="utf-8")
    # A sentinel proves that the helper does not discover or write port files.
    (root / "port_mode").write_text("sentinel\n", encoding="utf-8")
    return root


class FailingRestoreNSI(SysfsNSI):
    """Fail on the second timer write, which is the restoration write."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.write_count = 0

    def _write_timer_raw(self, value: int) -> None:
        self.write_count += 1
        if self.write_count == 2:
            raise OSError("simulated restore I/O failure")
        super()._write_timer_raw(value)


class NsiCalibrationUnitTest(unittest.TestCase):
    def test_timer_window_restores_exact_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fake_sysfs(Path(tmp), timer=37)
            nsi = SysfsNSI(root)
            with timer_window(nsi, 250):
                self.assertEqual(nsi.read_timer(), 250)
            self.assertEqual(nsi.read_timer(), 37)
            self.assertEqual((root / "port_mode").read_text(encoding="utf-8"),
                             "sentinel\n")

    def test_restore_failure_is_explicit_and_not_silenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nsi = FailingRestoreNSI(make_fake_sysfs(Path(tmp), timer=11))
            with self.assertRaises(TimerRestoreFailure) as raised:
                with timer_window(nsi, 100):
                    self.assertEqual(nsi.read_timer(), 100)
            self.assertIn("восстанов", str(raised.exception).lower())
            # The first write was successful; the failed second write must not
            # be replaced with a best-effort write to any other sysfs control.
            self.assertEqual(nsi.write_count, 2)

    def test_rejects_a_timer_path_that_is_not_exactly_pmu_timer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "port_mode").write_text("0\n", encoding="utf-8")
            with self.assertRaises(NSIError):
                SysfsNSI(root / "port_mode")

    def test_parse_snapshot_has_numeric_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nsi = SysfsNSI(make_fake_sysfs(Path(tmp)))
            snapshot = nsi.snapshot()
            self.assertEqual(snapshot["pmu_bandwidth_rd"], [1, 2, 3, 4])
            self.assertEqual(snapshot["available_pmu"], ["npu", "cpu0", "cpu1", "total"])

    def test_fit_and_unit_classification_prefers_rate_when_timer_matters(self) -> None:
        # The synthetic counter is proportional to bytes / measurement window.
        points = []
        for size in (32, 64, 128, 256):
            for timer_ms in (100, 250, 500, 1000):
                x = float(size * 1024 * 1024)
                y = 7.0 * x / (timer_ms / 1000.0)
                points.append({"bytes": x, "timer_ms": timer_ms, "value": y})
        fit = fit_line([p["bytes"] for p in points], [p["value"] for p in points])
        self.assertGreaterEqual(fit["r2"], 0.0)
        result = classify_unit_hypothesis(points)
        self.assertEqual(result["classification"], "rate")
        self.assertGreater(result["rate_fit"]["r2"], 0.99)

    def test_fit_line_rejects_degenerate_x(self) -> None:
        with self.assertRaises(NSIError):
            fit_line([1.0, 1.0], [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
