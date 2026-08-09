import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "summarize_target_run.py"


class SummarizeTargetRunTests(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[Path, dict[str, str]]:
        run = root / "run-summary"
        run.mkdir()
        metadata = {
            "schema_version": 1,
            "run_id": run.name,
            "elapsed_ns": 123456789,
            "exit_code": 86,
            "files": {"phases": "phases.jsonl"},
        }
        telemetry = [
            {
                "monotonic_ns": 100,
                "process": {"pid": 41, "available": True, "rss_kib": 10, "rss_hwm_kib": 20, "threads": 2},
                "system": {
                    "memory": {"MemAvailable_kib": 900, "SwapTotal_kib": 2000, "SwapFree_kib": 1500},
                    "thermal_zones": {
                        "zone0": {"type": "cpu", "millidegrees_c": 40000},
                        "zone1": {"type": "gpu", "millidegrees_c": 45000},
                    },
                },
            },
            {
                "monotonic_ns": 200,
                "process": {"pid": 41, "available": True, "rss_kib": 30, "rss_hwm_kib": 40, "threads": 3},
                "system": {
                    "memory": {"MemAvailable_kib": 700, "SwapTotal_kib": 2000, "SwapFree_kib": 1200},
                    "thermal_zones": {
                        "zone0": {"type": "cpu", "millidegrees_c": 50000},
                        "zone1": {"type": "gpu", "millidegrees_c": 43000},
                    },
                },
            },
        ]
        guard = [
            {
                "kind": "sample",
                "sample_seq": 0,
                "monotonic_ns": 110,
                "thermal_zones": [{"type": "cpu", "millidegrees_c": 42000, "path": "cpu0"}],
                "cpu_policies": [
                    {"path": "policy0", "cur_khz": 1000, "max_khz": 2000, "governor": "ondemand"}
                ],
                "cooling_devices": [{"type": "pwm-fan", "cur_state": 1, "max_state": 4}],
                "process": {"pid": 77, "available": True, "rss_kib": 100, "rss_hwm_kib": 120, "threads": 4},
            },
            {
                "kind": "sample",
                "sample_seq": 1,
                "monotonic_ns": 210,
                "thermal_zones": [{"type": "cpu", "millidegrees_c": 48000, "path": "cpu0"}],
                "cpu_policies": [
                    {"path": "policy0", "cur_khz": 1500, "max_khz": 2000, "governor": "performance"}
                ],
                "cooling_devices": [{"type": "pwm-fan", "cur_state": 4, "max_state": 4}],
                "process": {"pid": 77, "available": True, "rss_kib": 80, "rss_hwm_kib": 130, "threads": 5},
            },
            {"event": "exit", "status": 86},
        ]
        phases = [
            {"event": "idle", "monotonic_ns": 90, "step": 0},
            {"event": "measured", "monotonic_ns": 105, "step": 1},
            {"event": "complete", "monotonic_ns": 220, "step": 1},
        ]
        contents = {
            "metadata.json": json.dumps(metadata, sort_keys=True) + "\n",
            "telemetry.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in telemetry),
            "thermal-guard.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in guard),
            "phases.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in phases),
        }
        for name, content in contents.items():
            (run / name).write_text(content)
        return run, {name: hashlib.sha256(content.encode()).hexdigest() for name, content in contents.items()}

    def invoke(self, run: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(run), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_summary_is_deterministic_and_aggregates_both_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, hashes = self.make_run(Path(tmp))

            first = self.invoke(run)
            second = self.invoke(run)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            summary = json.loads(first.stdout)
            self.assertEqual(summary["run_id"], "run-summary")
            self.assertEqual(summary["exit_code"], 86)
            self.assertEqual(summary["elapsed_ns"], 123456789)
            self.assertEqual(summary["sample_counts"], {"telemetry": 2, "thermal_guard": 2, "phases": 3})
            self.assertEqual(summary["profiler_child"], {"peak_rss_kib": 30, "peak_rss_hwm_kib": 40, "peak_threads": 3})
            self.assertEqual(summary["workload_child"], {"peak_rss_kib": 100, "peak_rss_hwm_kib": 130, "peak_threads": 5})
            self.assertEqual(summary["thermal_by_type"]["cpu"], {"min": 40000, "median": 45000, "max": 50000})
            self.assertEqual(summary["thermal_by_type"]["gpu"], {"min": 43000, "median": 44000, "max": 45000})
            self.assertEqual(summary["cpu_policies"]["policy0"]["cur_khz"], {"min": 1000, "median": 1250, "max": 1500})
            self.assertEqual(summary["cpu_policies"]["policy0"]["max_khz"], {"min": 2000, "median": 2000, "max": 2000})
            self.assertEqual(summary["cpu_policies"]["policy0"]["governors"], ["ondemand", "performance"])
            self.assertEqual(summary["cooling_by_type"]["pwm-fan"], {"min": 1, "median": 2.5, "max": 4})
            self.assertEqual(
                summary["memory"],
                {
                    "mem_available_kib_min": 700,
                    "swap_free_kib_min": 1200,
                    "swap_used_kib_delta": 300,
                    "swap_used_kib_peak": 800,
                },
            )
            self.assertEqual(summary["thermal_guard_events"], {"exit": 1})
            self.assertEqual(summary["phase_events"], {"complete": 1, "idle": 1, "measured": 1})
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)

    def test_output_is_exclusive_and_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(Path(tmp))
            output = Path(tmp) / "summary.json"
            first = self.invoke(run, "--output", str(output))
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_text()
            second = self.invoke(run, "--output", str(output))
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("overwrite", second.stderr.lower())
            self.assertEqual(output.read_text(), original)

    def test_missing_or_malformed_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(Path(tmp))
            (run / "telemetry.jsonl").write_text("not-json\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("malformed", completed.stderr.lower())

            (run / "telemetry.jsonl").write_text("{}\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("schema", completed.stderr.lower())

            (run / "thermal-guard.jsonl").unlink()
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("thermal-guard.jsonl", completed.stderr)

    def test_phase_evidence_is_required_nonempty_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(Path(tmp))

            (run / "phases.jsonl").write_text("")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("empty", completed.stderr.lower())

            metadata = json.loads((run / "metadata.json").read_text())
            metadata["files"]["phases"] = "../outside.jsonl"
            (run / "metadata.json").write_text(json.dumps(metadata) + "\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unsafe", completed.stderr.lower())

            metadata.pop("files")
            (run / "metadata.json").write_text(json.dumps(metadata) + "\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("metadata.files", completed.stderr)

    def test_phase_path_rejects_noncanonical_components_and_outside_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for unsafe in (
                "/tmp/phases.jsonl",
                "./phases.jsonl",
                "foo//phases.jsonl",
                "C:/phases.jsonl",
                "bad\nname.jsonl",
            ):
                case_root = root / unsafe.replace("/", "_").replace("\n", "_")
                case_root.mkdir()
                run, _ = self.make_run(case_root)
                metadata = json.loads((run / "metadata.json").read_text())
                metadata["files"]["phases"] = unsafe
                (run / "metadata.json").write_text(json.dumps(metadata) + "\n")
                completed = self.invoke(run)
                self.assertNotEqual(completed.returncode, 0, unsafe)
                self.assertIn("unsafe", completed.stderr.lower(), unsafe)

            case_root = root / "symlink-case"
            case_root.mkdir()
            run, _ = self.make_run(case_root)
            outside = root / "outside.jsonl"
            outside.write_text('{"event":"complete","monotonic_ns":1,"step":0}\n')
            (run / "phases.jsonl").unlink()
            (run / "phases.jsonl").symlink_to(outside)
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unsafe", completed.stderr.lower())

    def test_phase_timestamps_must_be_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(Path(tmp))
            rows = [
                {"event": "idle", "monotonic_ns": 100, "step": 0},
                {"event": "complete", "monotonic_ns": 99, "step": 0},
            ]
            (run / "phases.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )

            completed = self.invoke(run)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("monotonic", completed.stderr.lower())

    def test_exact_statistics_use_a_bounded_insert_batch(self) -> None:
        namespace = runpy.run_path(str(SCRIPT))
        store_type = namespace["DiskStatsStore"]
        batch_size = namespace["STATS_BATCH_SIZE"]

        with store_type() as store:
            series = store.new_series()
            for value in range(20001):
                store.add(series, value, "synthetic.value")
                self.assertLessEqual(len(store._pending), batch_size)

            self.assertEqual(store.stats(series), {"min": 0, "median": 10000, "max": 20000})

    def test_swap_delta_is_derived_when_only_swap_free_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(Path(tmp))
            rows = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
            for row in rows:
                row["system"]["memory"].pop("SwapTotal_kib")
            (run / "telemetry.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )

            completed = self.invoke(run)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            memory = json.loads(completed.stdout)["memory"]
            self.assertEqual(memory["swap_free_kib_min"], 1200)
            self.assertEqual(memory["swap_used_kib_delta"], 300)
            self.assertNotIn("swap_used_kib_peak", memory)


if __name__ == "__main__":
    unittest.main()
