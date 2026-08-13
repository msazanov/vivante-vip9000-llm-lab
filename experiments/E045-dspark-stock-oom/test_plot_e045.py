#!/usr/bin/env python3
"""Регрессионные тесты для детерминированной последовательности E045."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "plot_e045.py"


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


class E045PlotTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "e045-mpl-test")
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=HERE,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_schema_config_and_safety_invariants(self) -> None:
        result = self._run("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("198/31/32/43 bins", result.stdout)
        self.assertIn("smoke-only point", result.stdout)

    def test_help_describes_completed_smoke_and_control(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("E045e", result.stdout)
        self.assertIn("0.263", result.stdout)
        self.assertIn("E045f", result.stdout)
        self.assertIn("1.120", result.stdout)
        self.assertIn("разные timer", result.stdout)
        self.assertIn("scopes", result.stdout)
        self.assertIn("намеренно не строится", result.stdout)

    def test_sequence_has_no_unmeasured_speed_or_acceptance(self) -> None:
        result = self._run("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = json.loads((HERE / "data" / "metrics.json").read_text(encoding="utf-8"))
        sequence = metrics["sequence"]
        self.assertEqual(sequence["pending"], "E045g")
        self.assertEqual({entry["id"] for entry in sequence["runs"]}, {"E045a", "E045b", "E045c", "E045d", "E045e", "E045f"})
        for entry in sequence["runs"]:
            if entry["id"] == "E045e":
                self.assertEqual(entry["throughput_status"], "measured_smoke_only")
                self.assertEqual(entry["acceptance_status"], "measured_smoke_only")
                self.assertEqual(entry["quality_status"], "not_compared")
                self.assertEqual(entry["comparison_scope"], "SMOKE_NOT_FULL_BENCHMARK_DIFFERENT_TIMER_FROM_E045F")
                self.assertEqual(entry["tok_s"], 0.263)
                self.assertEqual(entry["actual_tokens"], 2)
                self.assertEqual(entry["drafted"], 8)
                self.assertEqual(entry["accepted"], 0)
                continue
            if entry["id"] == "E045f":
                self.assertEqual(entry["throughput_status"], "measured_target_eval_only")
                self.assertEqual(entry["acceptance_status"], "not_applicable")
                self.assertEqual(entry["quality_status"], "visible_prefix_only")
                self.assertEqual(entry["tok_s"], 1.120)
                continue
            self.assertEqual(entry["throughput_status"], "unavailable")
            self.assertEqual(entry["acceptance_status"], "unavailable")
            self.assertEqual(entry["quality_status"], "not_run")
            self.assertNotIn("tok_s", entry)
            self.assertNotIn("acceptance_rate", entry)

    def test_each_status_and_normalized_run_are_distinct(self) -> None:
        metrics = json.loads((HERE / "data" / "metrics.json").read_text(encoding="utf-8"))
        statuses = {entry["id"]: entry["classification"] for entry in metrics["sequence"]["runs"]}
        self.assertEqual(statuses["E045a"], "CONFIGURATION_FAIL")
        self.assertEqual(statuses["E045b"], "LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_N0")
        self.assertEqual(statuses["E045c"], "LOAD_MEMORY_PASS + FUNCTIONAL_FAIL_EMPTY_PREFILL")
        self.assertEqual(statuses["E045d"], "WRAPPER_FAIL_NOT_RUN")
        self.assertEqual(statuses["E045e"], "FUNCTIONAL_SMOKE_PASS")
        self.assertEqual(statuses["E045f"], "TARGET_ONLY_BASELINE_PASS")
        for run, expected in (("b", 31), ("c", 32), ("e", 43), ("f", 33)):
            with (HERE / "data" / f"memory_thermal_{run}_1s.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected)
            self.assertEqual([int(row["elapsed_s"]) for row in rows], list(range(expected)))

    def test_regeneration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e045-plot-a-") as first, tempfile.TemporaryDirectory(prefix="e045-plot-b-") as second:
            result_a = self._run("--output-dir", first)
            result_b = self._run("--output-dir", second)
            self.assertEqual(result_a.returncode, 0, result_a.stderr)
            self.assertEqual(result_b.returncode, 0, result_b.stderr)
            self.assertEqual(_hashes(Path(first)), _hashes(Path(second)))
            self.assertEqual(
                set(_hashes(Path(first))),
                {
                    "e045_memory_oom.png",
                    "e045_memory_oom.svg",
                    "e045_temperature.png",
                    "e045_temperature.svg",
                    "e045_memory_comparison.png",
                    "e045_memory_comparison.svg",
                    "e045_temperature_comparison.png",
                    "e045_temperature_comparison.svg",
                },
            )

            memory_svg = (Path(first) / "e045_memory_comparison.svg").read_text(encoding="utf-8")
            self.assertIn("a/b/c/e/f", memory_svg)


if __name__ == "__main__":
    unittest.main()
