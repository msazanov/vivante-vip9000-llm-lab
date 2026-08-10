import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "summarize_bonsai_night.py"
CHART_SCRIPT = ROOT / "tooling" / "generate_bonsai_night_chart.py"
MODEL_PATH = "/home/orangepi/vip9000-lab/models/Bonsai-27B-Q1_0.gguf"
GOLDEN_STDOUT = (
    b"\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n"
    b"   - **Topic:** Memory bandwidth\n   - **Context\n\n"
)


class SummarizeBonsaiNightTests(unittest.TestCase):
    def make_run(self, root: Path, name: str = "run-a", *, abort: bool = False) -> Path:
        run = root / name
        run.mkdir()
        metadata = {
            "schema_version": 1,
            "run_id": name,
            "command": [
                "llama-bench", "-m", MODEL_PATH, "-p", "0", "-n", "128",
                "-r", "3", "-t", "6",
            ],
            "elapsed_ns": 2_000_000_000,
            "exit_code": 0,
            "child_return_code": 0,
            "build_commit": "38c66ad",
            "build_number": 9594,
        }
        stdout = [
            {
                "build_commit": "38c66ad",
                "build_number": 9594,
                "model_filename": MODEL_PATH,
                "model_type": "qwen35 27B Q1_0",
                "model_size": 3792459776,
                "model_n_params": 26895998464,
                "n_prompt": 0,
                "n_gen": 128,
                "backends": "CPU",
                "n_threads": 6,
                "cpu_mask": "0x3f",
                "poll": 50,
                "n_gpu_layers": 0,
                "devices": ["CPU"],
                "samples_ts": [1.0, 2.0, 3.0],
            }
        ]
        telemetry = [
            {
                "system": {
                    "thermal_zones": {
                        "zone0": {"type": "cpu", "millidegrees_c": 40000},
                        "zone1": {"type": "npu", "millidegrees_c": 45000},
                    },
                    "cpu_frequencies": {"cpu0_khz": 1000000, "cpu1_khz": 1200000},
                    "npu_frequencies": {"npu_hz": 500000000},
                    "memory": {"SwapFree_kib": 900},
                },
                "process": {"rss_kib": 100},
            },
            {
                "system": {
                    "thermal_zones": {
                        "zone0": {"type": "cpu", "millidegrees_c": 50000},
                        "zone1": {"type": "npu", "millidegrees_c": 46000},
                    },
                    "cpu_frequencies": {"cpu0_khz": 1400000, "cpu1_khz": 1300000},
                    "npu_frequencies": {"npu_hz": 600000000},
                    "memory": {"SwapFree_kib": 800},
                },
                "process": {"rss_kib": 120},
            },
        ]
        guard = [
            {"event": "start", "monotonic_ns": 1},
            {"event": "inventory", "monotonic_ns": 2},
            {"event": "child_started", "pid": 42, "monotonic_ns": 3},
            {
                "kind": "sample",
                "thermal_zones": [],
                "process": {"rss_kib": 600, "rss_hwm_kib": 700},
                "sample_seq": 0,
                "monotonic_ns": 4,
            },
            {"event": "child_exit", "returncode": 0, "monotonic_ns": 5},
            {"event": "abort", "reason": "temperature_abort:90000", "monotonic_ns": 6}
            if abort
            else {"event": "exit", "status": 0, "monotonic_ns": 6},
        ]
        (run / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        (run / "stdout.log").write_text(
            "".join(json.dumps(row) + "\n" for row in stdout), encoding="utf-8"
        )
        (run / "telemetry.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
        )
        (run / "thermal-guard.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in guard), encoding="utf-8"
        )
        return run

    def invoke(self, *runs: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(run) for run in runs)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_aggregates_full_model_decode_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_run(root, "run-a")
            second = self.make_run(root, "run-b")
            one = self.invoke(first, second)
            two = self.invoke(first, second)

        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(one.stdout, two.stdout)
        summary = json.loads(one.stdout)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual([run["run_id"] for run in summary["rows"]], ["run-a", "run-b"])
        result = summary["rows"][0]
        self.assertEqual(result["metric_scope"], "full_model_decode")
        self.assertEqual(result["unit"], "tokens_per_second")
        self.assertEqual(result["backend"], "CPU")
        self.assertEqual(result["backends"], ["CPU"])
        self.assertEqual(result["samples"], [1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["stddev"], 0.816496580927726)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["cv"], 0.408248290463863)
        self.assertEqual(result["telemetry_samples"], 2)
        self.assertEqual(result["peak_temperatures_millidegrees_c"], {"cpu": 50000, "npu": 46000})
        self.assertEqual(result["min_cpu_frequencies_khz"], 1000000)
        self.assertEqual(result["max_cpu_frequencies_khz"], 1400000)
        self.assertEqual(result["min_npu_frequency_hz"], 500000000)
        self.assertEqual(result["max_npu_frequency_hz"], 600000000)
        self.assertEqual(result["min_swap_free_kib"], 800)
        self.assertEqual(result["max_swap_free_kib"], 900)
        self.assertEqual(result["peak_rss_kib"], 600)
        self.assertEqual(result["peak_rss_hwm_kib"], 700)
        self.assertEqual(result["quality"], {"status": "not_checked", "exact_match": False})

    def test_rejects_nonzero_metadata_and_thermal_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary), abort=True)
            metadata = json.loads((run / "metadata.json").read_text())
            metadata["exit_code"] = 1
            (run / "metadata.json").write_text(json.dumps(metadata) + "\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exit_code", completed.stderr)

    def test_rejects_missing_decode_samples_and_nonempty_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            (run / "stdout.log").write_text(
                json.dumps({"n_prompt": 128, "n_gen": 0, "samples_ts": [1.0]}) + "\n"
            )
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("decode", completed.stderr.lower())

            (run / "stdout.log").write_text(
                json.dumps({"n_prompt": 0, "n_gen": 1, "samples_ts": ["nan"]}) + "\n"
            )
            (run / "telemetry.jsonl").write_text("")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("telemetry", completed.stderr.lower())

    def test_combined_poll_bundle_emits_one_row_per_decode_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary), "poll-screen")
            metadata_path = run / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["command"][metadata["command"].index("-r") + 1] = "2"
            metadata_path.write_text(json.dumps(metadata) + "\n")
            rows = [
                {
                    "build_commit": "38c66ad",
                    "build_number": 9594,
                    "model_filename": MODEL_PATH,
                    "model_type": "qwen35 27B Q1_0",
                    "model_size": 3792459776,
                    "model_n_params": 26895998464,
                    "n_prompt": 0,
                    "n_gen": 128,
                    "backends": ["CPU"],
                    "n_threads": 6,
                    "cpu_mask": "0x3f",
                    "poll": 0,
                    "n_gpu_layers": 0,
                    "devices": ["CPU"],
                    "samples_ts": [1.0, 2.0],
                },
                {
                    "build_commit": "38c66ad",
                    "build_number": 9594,
                    "model_filename": MODEL_PATH,
                    "model_type": "qwen35 27B Q1_0",
                    "model_size": 3792459776,
                    "model_n_params": 26895998464,
                    "n_prompt": 0,
                    "n_gen": 128,
                    "backends": ["CPU"],
                    "n_threads": 6,
                    "cpu_mask": "0x3f",
                    "poll": 50,
                    "n_gpu_layers": 0,
                    "devices": ["CPU"],
                    "samples_ts": [3.0, 5.0],
                },
            ]
            (run / "stdout.log").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            completed = self.invoke(run)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual([row["run_id"] for row in summary["rows"]], [
            "poll-screen--case-001",
            "poll-screen--case-002",
        ])
        first, second = summary["rows"]
        self.assertEqual(first["poll"], 0)
        self.assertEqual(second["poll"], 50)
        self.assertEqual(first["median"], 1.5)
        self.assertEqual(second["median"], 4.0)
        self.assertIn("poll=0", first["label"])
        self.assertIn("threads=6", first["label"])
        self.assertIn("mask=0x3f", first["label"])
        self.assertIn("ngl=0", first["label"])
        for row in (first, second):
            self.assertEqual(row["telemetry_scope"], "bundle")
            self.assertEqual(row["bundle_run_id"], "poll-screen")
            self.assertEqual(row["bundle_elapsed_seconds"], 2.0)
            self.assertNotIn("elapsed", row)
            self.assertEqual(row["peak_rss_kib"], 600)
            self.assertEqual(row["peak_rss_hwm_kib"], 700)

    def test_rejects_missing_or_failed_child_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "missing-child")
            guard_path = run / "thermal-guard.jsonl"
            rows = [json.loads(line) for line in guard_path.read_text().splitlines()]
            rows = [row for row in rows if row.get("event") != "child_exit"]
            guard_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("child_exit", completed.stderr)

            run = self.make_run(root, "failed-child")
            guard_path = run / "thermal-guard.jsonl"
            rows = [json.loads(line) for line in guard_path.read_text().splitlines()]
            next(row for row in rows if row.get("event") == "child_exit")["returncode"] = 9
            guard_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("child_exit", completed.stderr)

    def test_rejects_wrong_bonsai_identity_or_decode_workload(self) -> None:
        mutations = {
            "model path": lambda metadata, stdout: metadata["command"].__setitem__(
                metadata["command"].index(MODEL_PATH), "/tmp/not-bonsai.gguf"
            ),
            "model size": lambda metadata, stdout: stdout.__setitem__("model_size", 123),
            "n_gen": lambda metadata, stdout: stdout.__setitem__("n_gen", 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run = self.make_run(Path(temporary), name.replace(" ", "-"))
                metadata_path = run / "metadata.json"
                stdout_path = run / "stdout.log"
                metadata = json.loads(metadata_path.read_text())
                stdout = json.loads(stdout_path.read_text())
                mutate(metadata, stdout)
                metadata_path.write_text(json.dumps(metadata) + "\n")
                stdout_path.write_text(json.dumps(stdout) + "\n")

                completed = self.invoke(run)

                self.assertNotEqual(completed.returncode, 0)
                self.assertRegex(completed.stderr.lower(), r"model|n_gen|workload")

    def test_rejects_incomplete_reordered_or_unknown_guard_lifecycle(self) -> None:
        mutations = {
            "missing start": lambda rows: rows.__delitem__(0),
            "reordered": lambda rows: rows.__setitem__(
                slice(None), rows[:3] + [rows[-2], rows[3], rows[-1]]
            ),
            "unknown event": lambda rows: rows.insert(
                -2, {"event": "mystery", "monotonic_ns": 5}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run = self.make_run(Path(temporary), name.replace(" ", "-"))
                guard_path = run / "thermal-guard.jsonl"
                rows = [json.loads(line) for line in guard_path.read_text().splitlines()]
                mutate(rows)
                guard_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

                completed = self.invoke(run)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("thermal guard", completed.stderr.lower())

    def test_rejects_negative_frequency_swap_and_rss_telemetry(self) -> None:
        mutations = {
            "cpu frequency": lambda telemetry: telemetry[0]["system"][
                "cpu_frequencies"
            ].__setitem__("cpu0_khz", -1),
            "swap": lambda telemetry: telemetry[0]["system"]["memory"].__setitem__(
                "SwapFree_kib", -1
            ),
            "rss": lambda telemetry: telemetry[0]["process"].__setitem__("rss_kib", -1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run = self.make_run(Path(temporary), name.replace(" ", "-"))
                path = run / "telemetry.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                mutate(rows)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

                completed = self.invoke(run)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("non-negative", completed.stderr.lower())

    def test_requires_complete_telemetry_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "missing-telemetry")
            (run / "telemetry.jsonl").write_text("{}\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("thermal", completed.stderr.lower())

            run = self.make_run(root, "missing-backend")
            stdout_path = run / "stdout.log"
            row = json.loads(stdout_path.read_text())
            row.pop("backends")
            stdout_path.write_text(json.dumps(row) + "\n")
            completed = self.invoke(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("backend", completed.stderr.lower())

    def test_accepts_cpu_and_npu_telemetry_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary), "alias-telemetry")
            path = run / "telemetry.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                system = row["system"]
                system["cpu"] = system.pop("cpu_frequencies")
                system["npu"] = system.pop("npu_frequencies")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            completed = self.invoke(run)

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_chart_ready_mode_feeds_the_chart_without_manual_schema_editing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.make_run(root, "cpu-reference")
            candidate = self.make_run(root, "cpu-candidate")
            reference_golden = root / "golden-reference" / "stdout.log"
            candidate_golden = root / "golden-candidate" / "stdout.log"
            reference_golden.parent.mkdir()
            candidate_golden.parent.mkdir()
            reference_golden.write_bytes(GOLDEN_STDOUT)
            candidate_golden.write_bytes(GOLDEN_STDOUT)
            summary_path = root / "summary.json"
            chart_path = root / "chart.svg"

            summarized = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(reference),
                    str(candidate),
                    "--reference-run-id",
                    reference.name,
                    "--candidate-run-id",
                    candidate.name,
                    "--golden-reference-stdout",
                    str(reference_golden),
                    "--golden-candidate-stdout",
                    str(candidate_golden),
                    "--output",
                    str(summary_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            charted = subprocess.run(
                [
                    sys.executable,
                    str(CHART_SCRIPT),
                    str(summary_path),
                    "--output",
                    str(chart_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            ) if summarized.returncode == 0 else None

            self.assertEqual(summarized.returncode, 0, summarized.stderr)
            assert charted is not None
            self.assertEqual(charted.returncode, 0, charted.stderr)
            self.assertIn('data-run-id="cpu-reference"', chart_path.read_text())
            self.assertIn('data-run-id="cpu-candidate"', chart_path.read_text())

    def test_chart_ready_mode_rejects_cross_workload_comparisons(self) -> None:
        mutations = {
            "n_gen": lambda command, stdout: (
                command.__setitem__(command.index("-n") + 1, "64"),
                stdout.__setitem__("n_gen", 64),
            ),
            "repetitions": lambda command, stdout: (
                command.__setitem__(command.index("-r") + 1, "2"),
                stdout.__setitem__("samples_ts", [1.0, 2.0]),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                reference = self.make_run(root, "cpu-reference")
                candidate = self.make_run(root, "cpu-candidate")
                metadata_path = candidate / "metadata.json"
                stdout_path = candidate / "stdout.log"
                metadata = json.loads(metadata_path.read_text())
                stdout = json.loads(stdout_path.read_text())
                mutate(metadata["command"], stdout)
                metadata_path.write_text(json.dumps(metadata) + "\n")
                stdout_path.write_text(json.dumps(stdout) + "\n")
                reference_golden = root / "golden-reference" / "stdout.log"
                candidate_golden = root / "golden-candidate" / "stdout.log"
                reference_golden.parent.mkdir()
                candidate_golden.parent.mkdir()
                reference_golden.write_bytes(GOLDEN_STDOUT)
                candidate_golden.write_bytes(GOLDEN_STDOUT)

                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        str(reference),
                        str(candidate),
                        "--reference-run-id",
                        reference.name,
                        "--candidate-run-id",
                        candidate.name,
                        "--golden-reference-stdout",
                        str(reference_golden),
                        "--golden-candidate-stdout",
                        str(candidate_golden),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 2)
                self.assertIn("workload", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
