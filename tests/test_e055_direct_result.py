"""Regression gates for the direct-OpenSSH E055 target result.

The result parser deliberately accepts only the bounded 20-run microgate.  It
normalizes every run to nanoseconds per traversal before it constructs paired
cold/hot penalties, and it refuses to promote the intentionally incomplete
matrix.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tooling.e055_direct_result import (
    analyze_phase,
    audit_phase,
    validate_phase,
    validate_run,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments/E055-q1-hot-cold/results"
    / "e055a-direct-native-o3-core-20260814t161316z"
)
FRESH_RESULT = (
    ROOT
    / "experiments/E055-q1-hot-cold/results"
    / "e055a-direct-native-rerun2-o3-core-20260814t173636z"
)


class E055DirectResultTest(unittest.TestCase):
    def test_fresh_no_tty_microgate_passes_but_cannot_promote(self) -> None:
        audit = audit_phase(FRESH_RESULT / "raw")
        self.assertEqual(audit["phase"], FRESH_RESULT.name)
        self.assertEqual(len(audit["qualified_rows"]), 20)
        self.assertEqual(audit["invalid_runs"], [])

        analysis = analyze_phase(audit)
        self.assertEqual(analysis["status"], "PASS_BOUNDED_MICROGATE")
        self.assertEqual(analysis["phase"], FRESH_RESULT.name)
        self.assertEqual(analysis["valid_run_count"], 20)
        self.assertEqual(analysis["invalid_run_count"], 0)
        self.assertEqual(analysis["cpus"]["0"]["qualified_pair_count"], 5)
        self.assertEqual(analysis["cpus"]["6"]["qualified_pair_count"], 5)
        self.assertIn("20/20 qualified", analysis["chart_annotation"])
        self.assertNotIn("X =", analysis["chart_annotation"])
        self.assertFalse(analysis["promotion"]["eligible"])
        self.assertEqual(analysis["promotion"]["qualified_matrix_rows"], 20)
        self.assertEqual(analysis["promotion"]["required_matrix_rows"], 1260)

    def test_published_csv_uses_repository_lf_line_endings(self) -> None:
        samples = (FRESH_RESULT / "samples.csv").read_bytes()
        self.assertNotIn(b"\r", samples)
        self.assertTrue(samples.endswith(b"\n"))

    def test_live_gate_requires_sched_getcpu_evidence_not_cpu_label(self) -> None:
        source = RESULT / "raw/cpu0-pair1-hot"
        with self.assertRaisesRegex(ValueError, "requested CPU label"):
            validate_run(source)

        with tempfile.TemporaryDirectory(prefix="e055-observed-cpu-") as raw:
            run = Path(raw) / source.name
            shutil.copytree(source, run)
            pmu_path = run / "e049c.json"
            pmu = json.loads(pmu_path.read_text(encoding="utf-8"))
            label_index = pmu["command"].index("--cpu")
            del pmu["command"][label_index : label_index + 2]
            pmu_path.write_text(json.dumps(pmu), encoding="utf-8")

            qualified = validate_run(run)
            self.assertEqual(qualified["requested_cpu"], 0)
            self.assertEqual(qualified["observed_cpu"], 0)

            harness_path = run / "harness.stdout.raw"
            harness = json.loads(harness_path.read_text(encoding="utf-8"))
            harness["cpu"] = 6
            harness_path.write_text(json.dumps(harness) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observed CPU"):
                validate_run(run)

    def test_live_single_run_gate_accepts_valid_and_rejects_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested CPU label"):
            validate_run(RESULT / "raw/cpu6-pair2-hot")
        with self.assertRaisesRegex(ValueError, "PMU event window"):
            validate_run(
                RESULT / "raw/cpu6-pair1-hot", require_observed_cpu=False
            )

    def test_published_phase_fails_closed_on_two_short_pmu_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, r"2 raw sample\(s\) failed"):
            validate_phase(RESULT / "raw", require_observed_cpu=False)
        audit = audit_phase(RESULT / "raw", require_observed_cpu=False)
        self.assertEqual(audit["raw_run_count"], 20)
        self.assertEqual(len(audit["qualified_rows"]), 18)
        self.assertEqual(
            [issue["run_id"] for issue in audit["invalid_runs"]],
            ["cpu6-pair1-hot", "cpu6-pair3-hot"],
        )
        self.assertTrue(
            all("PMU event window" in issue["reason"] for issue in audit["invalid_runs"])
        )
        analysis = analyze_phase(audit)
        self.assertEqual(analysis["valid_run_count"], 18)
        self.assertEqual(analysis["normalization"], "elapsed_ns / calls")
        self.assertAlmostEqual(
            analysis["cpus"]["0"]["median_pair_cold_hot_penalty"],
            1.0687418423090615,
        )
        self.assertAlmostEqual(
            analysis["cpus"]["6"]["median_pair_cold_hot_penalty"],
            1.561660783813544,
        )
        self.assertEqual(analysis["cpus"]["6"]["qualified_pair_count"], 3)
        self.assertIn("X = invalid", analysis["chart_annotation"])
        self.assertFalse(analysis["promotion"]["eligible"])
        self.assertEqual(analysis["promotion"]["qualified_matrix_rows"], 18)
        self.assertEqual(analysis["promotion"]["required_matrix_rows"], 1260)

    def test_mutated_raw_pmu_fails_closed(self) -> None:
        source = RESULT / "raw/cpu0-pair1-hot/e049c.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["events"][0]["running_ratio"] = 0.999
        with tempfile.TemporaryDirectory(prefix="e055-direct-mutation-") as raw:
            phase = Path(raw)
            run = phase / "cpu0-pair1-hot"
            run.mkdir()
            for name in (
                "harness.stdout.raw",
                "harness.stderr.raw",
                "wrapper.stdout.raw",
                "wrapper.stderr.raw",
            ):
                (run / name).write_bytes(
                    (RESULT / "raw/cpu0-pair1-hot" / name).read_bytes()
                )
            (run / "e049c.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "running ratio"):
                validate_phase(
                    phase,
                    require_complete=False,
                    require_observed_cpu=False,
                )

    def test_valid_looking_but_wrong_kernel_checksum_fails_closed(self) -> None:
        source = FRESH_RESULT / "raw/cpu6-pair1-hot"
        with tempfile.TemporaryDirectory(prefix="e055-direct-checksum-") as raw:
            run = Path(raw) / source.name
            shutil.copytree(source, run)
            harness_path = run / "harness.stdout.raw"
            harness = json.loads(harness_path.read_text(encoding="utf-8"))
            harness["checksum"] = "0x1111111111111111"
            harness_path.write_text(json.dumps(harness) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output checksum"):
                validate_run(run)

    def test_total_window_ratio_is_not_used(self) -> None:
        rows = audit_phase(
            RESULT / "raw", require_observed_cpu=False
        )["qualified_rows"]
        cpu0_pair1 = [
            row for row in rows if row["cpu"] == 0 and row["pair_index"] == 1
        ]
        hot = next(row for row in cpu0_pair1 if row["cache_state"] == "hot_repeat")
        cold = next(
            row for row in cpu0_pair1 if row["cache_state"] == "cold_conditioned"
        )
        normalized = cold["ns_per_traversal"] / hot["ns_per_traversal"]
        wrong_total_ratio = cold["elapsed_ns"] / hot["elapsed_ns"]
        self.assertAlmostEqual(normalized, 1.0687418423090615)
        self.assertLess(wrong_total_ratio, 0.01)


if __name__ == "__main__":
    unittest.main()
