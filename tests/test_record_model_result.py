import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "tooling" / "record_model_result.py"


def load_recorder_module():
    spec = importlib.util.spec_from_file_location("record_model_result", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_result(run_id: str = "tb27b-cpu-001") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "date": "2026-08-07",
        "experiment": "E100",
        "model": {
            "id": "prism-ml/Ternary-Bonsai-27B-gguf",
            "sha256": "a" * 64,
            "quantization": "Q2_0_g128",
            "size_bytes": 7_200_000_000,
        },
        "configuration": {
            "backend": "cpu",
            "partition": "cpu-only",
            "context": 4096,
            "batch": 512,
            "ubatch": 128,
            "threads": 8,
        },
        "software": {
            "repository_commit": "0123456789abcdef0123456789abcdef01234567",
            "runtime": "llama.cpp",
            "runtime_commit": "runtime-2026-08-07",
            "compiler": "gcc 13.3.0",
            "sdk": "none",
            "driver": "linux-cpu",
            "kernel": "6.6.44-sunxi",
            "command": "llama-bench --model model.gguf",
        },
        "workload": {
            "tokenizer": "llama-bpe",
            "prompt_suite": "fixed-512-token-prompt-v1",
            "prompt_tokens": 512,
            "generated_tokens": 128,
            "seed": 7,
            "deterministic": True,
            "warmup_iterations": 2,
        },
        "performance": {
            "prompt_tps": 12.5,
            "decode_tps": 1.75,
            "ttft_ms": 840.0,
            "peak_rss_mib": 8123.0,
            "repetitions": 5,
            "statistics": {
                "prompt_tps": {
                    "median": 12.5,
                    "p10": 12.0,
                    "p90": 13.0,
                    "min": 11.8,
                    "max": 13.2,
                    "cv": 0.04,
                    "samples": [11.8, 12.0, 12.5, 13.0, 13.2],
                },
                "decode_tps": {
                    "median": 1.75,
                    "p10": 1.6,
                    "p90": 1.9,
                    "min": 1.5,
                    "max": 2.0,
                    "cv": 0.08,
                    "samples": [1.5, 1.6, 1.75, 1.9, 2.0],
                },
                "ttft_ms": {
                    "median": 840.0,
                    "p10": 800.0,
                    "p90": 900.0,
                    "min": 780.0,
                    "max": 920.0,
                    "cv": 0.05,
                    "samples": [780.0, 800.0, 840.0, 900.0, 920.0],
                },
            },
        },
        "quality": {
            "method": "deterministic-token-agreement",
            "reference_run_id": run_id,
            "metric": "token_agreement",
            "value": 1.0,
            "threshold": 1.0,
            "passed": True,
        },
        "status": "qualified",
        "raw_result": f"benchmarks/results/{run_id}/summary.json",
        "notes": "reference-shaped smoke run",
    }


def card_text() -> str:
    return """# Model card

| Date | Run | Experiment | Backend / partition | Quantization | ctx / batch / ubatch / threads | Prompt tok/s | Decode tok/s | TTFT ms | Peak RSS MiB | Quality | Status | Raw |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
<!-- MODEL_RESULTS_START -->
<!-- MODEL_RESULTS_END -->
"""


def details_card_text() -> str:
    return """# Model card

<details>
<summary>Полная таблица запусков</summary>

| Date | Run | Experiment | Backend / partition | Quantization | ctx / batch / ubatch / threads | Prompt tok/s | Decode tok/s | TTFT ms | Peak RSS MiB | Quality | Status | Raw |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
<!-- MODEL_RESULTS_START -->
<!-- MODEL_RESULTS_END -->
</details>
"""


class RecordModelResultTests(unittest.TestCase):
    def write_raw_bundle(
        self,
        repo_root: Path,
        result: dict,
        *,
        phase_path: str = "phases.jsonl",
        metadata_updates: dict | None = None,
    ) -> Path:
        run_dir = repo_root / "benchmarks" / "results" / result["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text("{}\n")
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text("")
        (run_dir / "telemetry.jsonl").write_text("")
        phase_file = run_dir / phase_path
        phase_file.parent.mkdir(parents=True, exist_ok=True)
        phase_file.write_text("")
        metadata = {
            "run_id": result["run_id"],
            "exit_code": 0,
            "launch_error": None,
            "profiler_error": None,
            "files": {"phases": phase_path},
        }
        if metadata_updates:
            metadata.update(metadata_updates)
        (run_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
        return run_dir

    def invoke(
        self,
        result_path: Path,
        ledger: Path,
        card: Path,
        *,
        repo_root: Path | None = None,
        stage_raw: bool | None = None,
        phase_path: str = "phases.jsonl",
    ) -> subprocess.CompletedProcess[str]:
        result = json.loads(result_path.read_text())
        repo_root = repo_root or result_path.parent
        if stage_raw is None:
            stage_raw = (
                result.get("status") == "qualified"
                and result.get("raw_result")
                == f"benchmarks/results/{result.get('run_id')}/summary.json"
            )
        if stage_raw:
            self.write_raw_bundle(repo_root, result, phase_path=phase_path)
        return subprocess.run(
            [
                sys.executable,
                str(RECORDER),
                "--input",
                str(result_path),
                "--ledger",
                str(ledger),
                "--card",
                str(card),
                "--repo-root",
                str(repo_root),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_valid_result_updates_ledger_and_model_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(valid_result()))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual([record["run_id"] for record in records], ["tb27b-cpu-001"])
            rendered = card.read_text()
            self.assertIn("tb27b-cpu-001", rendered)
            self.assertIn("1.750", rendered)
            self.assertIn("token_agreement=1", rendered)
            self.assertIn("[raw](../results/tb27b-cpu-001/summary.json)", rendered)

    def test_valid_result_updates_table_inside_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(valid_result()))
            card.write_text(details_card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            rendered = card.read_text()
            self.assertEqual(rendered.count("<!-- MODEL_RESULTS_START -->"), 1)
            self.assertEqual(rendered.count("<!-- MODEL_RESULTS_END -->"), 1)
            row_position = rendered.index("tb27b-cpu-001")
            end_position = rendered.index("<!-- MODEL_RESULTS_END -->")
            self.assertLess(row_position, end_position)
            self.assertLess(end_position, rendered.index("</details>"))

    def test_duplicate_run_is_rejected_without_modifying_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(valid_result()))
            card.write_text(card_text())
            first = self.invoke(result_path, ledger, card)
            self.assertEqual(first.returncode, 0, first.stderr)
            ledger_before = ledger.read_text()
            card_before = card.read_text()

            second = self.invoke(result_path, ledger, card)

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("duplicate run_id", second.stderr)
            self.assertEqual(ledger.read_text(), ledger_before)
            self.assertEqual(card.read_text(), card_before)

    def test_incomplete_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            incomplete = valid_result()
            del incomplete["performance"]["decode_tps"]
            result_path.write_text(json.dumps(incomplete))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("performance.decode_tps", completed.stderr)
            self.assertFalse(ledger.exists())
            self.assertEqual(card.read_text(), card_text())

    def test_qualified_result_requires_passing_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result()
            result["quality"]["passed"] = False
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("qualified result requires quality.passed=true", completed.stderr)
            self.assertFalse(ledger.exists())

    def test_qualified_result_requires_measured_performance(self) -> None:
        for field in ("prompt_tps", "decode_tps", "ttft_ms", "peak_rss_mib"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                result["performance"][field] = None
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("qualified result requires", completed.stderr)

    def test_qualified_result_requires_positive_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result()
            result["performance"]["repetitions"] = 0
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("repetitions", completed.stderr)

    def test_qualified_result_rejects_unresolved_provenance(self) -> None:
        for field in ("runtime", "runtime_commit", "compiler", "sdk", "driver", "kernel", "command"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                result["software"][field] = "unresolved"
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("software", completed.stderr)

    def test_qualified_result_rejects_invalid_repository_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result()
            result["software"]["repository_commit"] = "unknown"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("repository_commit", completed.stderr)

    def test_qualified_result_rejects_missing_or_nondeterministic_workload(self) -> None:
        for mutation in ("missing_tokenizer", "nondeterministic"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                if mutation == "missing_tokenizer":
                    del result["workload"]["tokenizer"]
                else:
                    result["workload"]["deterministic"] = False
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("workload", completed.stderr)

    def test_qualified_result_rejects_missing_or_inconsistent_statistics(self) -> None:
        mutations = {
            "missing": lambda result: result["performance"]["statistics"].pop("decode_tps"),
            "sample_count": lambda result: result["performance"]["statistics"]["prompt_tps"]["samples"].pop(),
            "ordering": lambda result: result["performance"]["statistics"]["prompt_tps"].update(p10=12.6),
            "top_level_mismatch": lambda result: result["performance"].update(prompt_tps=12.6),
        }
        for mutation_name, mutation in mutations.items():
            with self.subTest(mutation=mutation_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                mutation(result)
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("statistics", completed.stderr)

    def test_statistics_samples_reject_null_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result("null-sample-001")
            result["status"] = "unqualified"
            result["quality"]["passed"] = None
            result["quality"]["reference_run_id"] = ""
            result["performance"]["statistics"]["prompt_tps"]["samples"][0] = None
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("samples", completed.stderr)

    def test_nonqualified_result_requires_schema_objects(self) -> None:
        for field in ("software", "workload"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result("unqualified-missing-001")
                result["status"] = "unqualified"
                result["quality"]["passed"] = None
                result["quality"]["reference_run_id"] = ""
                result[field] = None
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(field, completed.stderr)

    def test_nonqualified_statuses_allow_nullable_performance_values(self) -> None:
        for status, passed in (("unqualified", None), ("rejected", False), ("failed", None)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result(f"{status}-001")
                result["status"] = status
                result["quality"]["passed"] = passed
                result["quality"]["reference_run_id"] = ""
                result["performance"].update(
                    prompt_tps=None,
                    decode_tps=None,
                    ttft_ms=None,
                    peak_rss_mib=None,
                    repetitions=0,
                )
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_qualified_result_requires_non_empty_quality_fields(self) -> None:
        for field in ("method", "metric", "reference_run_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                result["quality"][field] = ""
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("qualified result requires", completed.stderr)

    def test_qualified_result_rejects_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result("candidate-001")
            result["quality"]["reference_run_id"] = "missing-reference"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("reference_run_id", completed.stderr)

    def test_qualified_result_rejects_nonexistent_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result = valid_result("candidate-001")
            result["quality"]["reference_run_id"] = "cpu-reference-001"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cpu-reference-001", completed.stderr)

    def test_qualified_result_rejects_missing_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(valid_result()))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card, stage_raw=False)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("raw evidence", completed.stderr)

    def test_qualified_result_rejects_mismatched_raw_metadata_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = valid_result()
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())
            self.write_raw_bundle(root, result, metadata_updates={"run_id": "other-run"})

            completed = self.invoke(result_path, ledger, card, stage_raw=False)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("metadata.run_id", completed.stderr)

    def test_qualified_result_rejects_failed_raw_metadata(self) -> None:
        for field, value in (("exit_code", 7), ("launch_error", "launch failed"), ("profiler_error", "profile failed")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = valid_result()
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())
                self.write_raw_bundle(root, result, metadata_updates={field: value})

                completed = self.invoke(result_path, ledger, card, stage_raw=False)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("metadata", completed.stderr)

    def test_qualified_result_rejects_traversal_in_raw_phase_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = valid_result()
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())
            self.write_raw_bundle(
                root,
                result,
                metadata_updates={"files": {"phases": "../outside-phases.jsonl"}},
            )

            completed = self.invoke(result_path, ledger, card, stage_raw=False)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("phases", completed.stderr)

    def test_qualified_result_accepts_valid_raw_bundle_with_custom_phase_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = valid_result()
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(result))
            card.write_text(card_text())

            completed = self.invoke(
                result_path,
                ledger,
                card,
                phase_path="nested/custom-phases.jsonl",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_qualified_result_rejects_mismatched_reference_model_configuration_workload_or_runtime(self) -> None:
        for mismatch in ("model", "configuration", "workload", "software"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                reference = valid_result("cpu-reference-001")
                candidate = valid_result("candidate-001")
                candidate["quality"]["reference_run_id"] = reference["run_id"]
                if mismatch == "model":
                    candidate["model"]["sha256"] = "b" * 64
                elif mismatch == "configuration":
                    candidate["configuration"]["context"] += 1
                elif mismatch == "workload":
                    candidate["workload"]["prompt_tokens"] += 1
                else:
                    candidate["software"]["runtime"] = "different-runtime"
                ledger.write_text(json.dumps(reference) + "\n")
                result_path.write_text(json.dumps(candidate))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("reference_run_id", completed.stderr)

    def test_qualified_result_accepts_matching_prior_cpu_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            reference = valid_result("cpu-reference-001")
            candidate = valid_result("candidate-001")
            candidate["quality"]["reference_run_id"] = reference["run_id"]
            ledger.write_text(json.dumps(reference) + "\n")
            result_path.write_text(json.dumps(candidate))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                [json.loads(line)["run_id"] for line in ledger.read_text().splitlines()],
                ["cpu-reference-001", "candidate-001"],
            )

    def test_raw_result_rejects_unsafe_components_and_wrong_run_directory(self) -> None:
        for raw_result in (
            "benchmarks/results/tb27b-cpu-001/[summary].json",
            "benchmarks/results/tb27b-cpu-001/summary\n.json",
            "benchmarks/results/tb27b-cpu-001/../other.json",
            "benchmarks/results/other-run/summary.json",
        ):
            with self.subTest(raw_result=raw_result), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = root / "result.json"
                ledger = root / "model-runs.jsonl"
                card = root / "model.md"
                result = valid_result()
                result["raw_result"] = raw_result
                result_path.write_text(json.dumps(result))
                card.write_text(card_text())

                completed = self.invoke(result_path, ledger, card)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("raw_result", completed.stderr)

    def test_raw_result_accepts_valid_qualified_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            result_path.write_text(json.dumps(valid_result()))
            card.write_text(card_text())

            completed = self.invoke(result_path, ledger, card, stage_raw=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_card_replace_failure_restores_both_canonical_files(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "model-runs.jsonl"
            card = root / "model.md"
            ledger.write_text('{"run_id":"existing"}\n')
            card.write_text(card_text())
            ledger_before = ledger.read_bytes()
            card_before = card.read_bytes()
            result = valid_result()
            self.write_raw_bundle(root, result)
            ledger_content, card_content = recorder.update_contents(
                result, ledger, card, repo_root=root
            )
            original_replace = recorder.os.replace

            def fail_card_replace(source, destination):
                if Path(destination) == card:
                    raise OSError("injected card replacement failure")
                return original_replace(source, destination)

            with mock.patch.object(recorder.os, "replace", side_effect=fail_card_replace):
                with self.assertRaises(OSError):
                    recorder.write_pair(ledger, ledger_content, card, card_content)

            self.assertEqual(ledger.read_bytes(), ledger_before)
            self.assertEqual(card.read_bytes(), card_before)

    def test_ledger_and_card_same_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            canonical = root / "same-file"
            result_path.write_text(json.dumps(valid_result()))
            canonical.write_text(card_text())

            completed = self.invoke(result_path, canonical, canonical)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("same file", completed.stderr)


if __name__ == "__main__":
    unittest.main()
