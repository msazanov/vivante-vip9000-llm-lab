#!/usr/bin/env python3
"""Минимальный TDD/regression test для генератора E044."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "plot_e044.py"


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


class E044PlotTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        with_mpl = Path(tempfile.gettempdir()) / "e044-mpl-test"
        env["MPLCONFIGDIR"] = str(with_mpl)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=HERE,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_schema_and_safety_invariants(self) -> None:
        result = self._run("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("screen=4 rows, full=95 rows, 3 reset runs", result.stdout)

    def test_regeneration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e044-plot-a-") as first, tempfile.TemporaryDirectory(prefix="e044-plot-b-") as second:
            result_a = self._run("--output-dir", first)
            result_b = self._run("--output-dir", second)
            self.assertEqual(result_a.returncode, 0, result_a.stderr)
            self.assertEqual(result_b.returncode, 0, result_b.stderr)
            self.assertEqual(_hashes(Path(first)), _hashes(Path(second)))
            self.assertEqual(
                set(_hashes(Path(first))),
                {
                    "e044_screen_throughput.png",
                    "e044_screen_throughput.svg",
                    "e044_full_reset_temperature.png",
                    "e044_full_reset_temperature.svg",
                },
            )


if __name__ == "__main__":
    unittest.main()
