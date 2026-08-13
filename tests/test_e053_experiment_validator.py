import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tooling.branch_preflight import build_manifest
from tooling.experiment_validator import build_file_manifest, validate_experiment
from tooling.scaffold_experiment import scaffold_experiment


COMMON_PROMPT = "Explain the A733 memory bottleneck in one short sentence."
COMMON_PROMPT_SHA256 = "3422031cc96896c8aff5ea0363ef6cddca9ba485437f56dbfa63d7703f01a7de"
ROOT = Path(__file__).resolve().parents[1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_required_planned_files(root: Path) -> None:
    (root / "README.md").write_text("# E047\n\nСтатус: PLANNED.\n", encoding="utf-8")
    (root / "hypothesis-preflight.md").write_text(
        "# Предварительная проверка гипотезы\n\nДубликатов не найдено.\n",
        encoding="utf-8",
    )
    (root / "commands.txt").write_text("Команды ещё не запускались.\n", encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"schema_version": "e053-environment/v1", "captured": False, "status": "planned"}) + "\n",
        encoding="utf-8",
    )
    (root / "device.json").write_text(
        json.dumps({"schema_version": "e053-device/v1", "captured": False, "status": "planned"}) + "\n",
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "results").mkdir()
    (root / "results/summary.json").write_text(
        json.dumps(
            {
                "schema_version": "e053-experiment-summary/v1",
                "experiment_id": "E047",
                "status": "planned",
                "generated_at_utc": "2026-08-14T00:00:00+00:00",
                "reason": "Тестовый план без запуска.",
                "model_used": False,
                "runs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    preflight = build_manifest(
        root,
        ["E047", "LFM2.5"],
        refs=["HEAD"],
        experiment_id="E047",
        hypothesis_query="сравнить Bonsai и LFM2.5 по единому протоколу",
    )
    (root / "data/branch-preflight.json").write_text(
        json.dumps(preflight) + "\n", encoding="utf-8"
    )
    manifest = build_file_manifest(
        root,
        [
            "README.md",
            "hypothesis-preflight.md",
            "commands.txt",
            "environment.json",
            "device.json",
            "data/branch-preflight.json",
            "results/summary.json",
        ],
        status="planned",
    )
    (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def write_valid_executed_files(root: Path, status: str = "failed") -> None:
    write_required_planned_files(root)
    environment = {
        "schema_version": "e053-environment/v1",
        "captured": True,
        "status": "captured",
        "repo_commit": "a" * 40,
        "runtime": {"name": "llama.cpp", "version": "9594", "commit": "b" * 40},
        "compiler": {"name": "gcc", "version": "12.2.0"},
        "sdk": {"used": False, "name": "none", "version": "not_used"},
        "driver": {"name": "vipcore", "version": "2.0.3.2-AW"},
        "kernel": {"release": "6.6.98-sun60iw2", "build": "#1"},
        "captured_at_utc": "2026-08-14T00:00:00+00:00",
    }
    device = {
        "schema_version": "e053-device/v1",
        "captured": True,
        "status": "captured",
        "device_id": "orangepizero3w-lab-1",
        "board": "OrangePi Zero 3W",
        "soc": "Allwinner A733",
        "npu": "Vivante VIP9000",
        "memory_bytes": 12_000_000_000,
        "os_release": "Orange Pi 6.6 BSP",
        "captured_at_utc": "2026-08-14T00:00:00+00:00",
    }
    (root / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
    (root / "device.json").write_text(json.dumps(device) + "\n", encoding="utf-8")
    command = f"llama-cli --prompt '{COMMON_PROMPT}' -n 4 --seed 123 --temp 0"
    (root / "commands.txt").write_text(command + "\n", encoding="utf-8")

    rendered = f"<|user|>{COMMON_PROMPT}<|assistant|>"
    prompt_ids = [101, 733, 9000]
    run_result_status = status if status in {"failed", "rejected"} else "passed"
    memory_source = {
        "measurement_method": "static_tensor_accounting",
        "measurement_source": "ggml graph trace",
        "measurement_confidence": "high",
    }
    workload = {
        "common_prompt_text": COMMON_PROMPT,
        "common_prompt_sha256": COMMON_PROMPT_SHA256,
        "rendered_prompt_text": rendered,
        "rendered_prompt_sha256": sha256_text(rendered),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_encoding": "canonical-json-array-utf8",
        "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "n_predict": 4,
        "seed": 123,
    }
    summary = {
        "schema_version": "e053-experiment-summary/v1",
        "experiment_id": "E047",
        "status": status,
        "generated_at_utc": "2026-08-14T00:01:00+00:00",
        "reason": "Тестовая фиксация результата.",
        "model_used": True,
        "runs": [
            {
                "run_id": "run-001",
                "variant_id": "bonsai-q1-gguf-cpu",
                "model": {"used": True, "sha256": "c" * 64},
                "command": {
                    "shell": command,
                    "sha256": sha256_text(command),
                    "workload_sha256": sha256_text(
                        json.dumps(workload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    ),
                },
                "workload": workload,
                "memory_accounting": {
                    "logical": {"logical_bytes": 1000, **memory_source},
                    "unique_weights": {"unique_weight_bytes": 900, **memory_source},
                    "observed_direct_ddr": {
                        "observed_direct_ddr_read_bytes": 800,
                        "observed_direct_ddr_write_bytes": 80,
                        "measurement_method": "pmu_counter_delta",
                        "measurement_source": "A733 DDR PMU",
                        "measurement_confidence": "medium",
                    },
                    "inferred_ddr": {
                        "inferred_ddr_read_bytes": 820,
                        "inferred_ddr_write_bytes": 90,
                        "measurement_method": "counter_model_fit",
                        "measurement_source": "logical bytes and PMU calibration",
                        "measurement_confidence": "low",
                    },
                },
                "result": {"status": run_result_status},
            }
        ],
    }
    (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    (root / "raw").mkdir()
    (root / "raw/stdout.log").write_text("model output\n", encoding="utf-8")
    (root / "raw/stderr.log").write_text("", encoding="utf-8")
    telemetry = {
        "schema_version": "e047-telemetry-event/v1",
        "run_id": "run-001",
        "timestamp_ns": 1,
        "cpu_temperature_c": 42.0,
        "cpu_frequency_hz": [2208000000, 2208000000],
        "npu_frequency_hz": 1008000000,
        "rss_bytes": 1024,
        "ddr": {
            "observed_direct_ddr_read_bytes": 800,
            "observed_direct_ddr_write_bytes": 80,
            "measurement_method": "pmu_counter_delta",
            "measurement_source": "A733 DDR PMU",
            "measurement_confidence": "medium",
        },
    }
    trace = {
        "schema_version": "e047-trace-event/v1",
        "run_id": "run-001",
        "step": "decode",
        "token_index": 0,
        "layer_index": 0,
        "op_index": 0,
        "op_name": "GEMV",
        "backend": "cpu",
        "start_ns": 10,
        "end_ns": 20,
        "duration_ns": 10,
        "inputs": ["weights", "activation"],
        "outputs": ["result"],
        "memory": {
            "logical": {
                "logical_read_bytes": 1000,
                "logical_write_bytes": 100,
                "measurement_method": "static_tensor_accounting",
                "measurement_source": "ggml graph",
                "measurement_confidence": "high",
            },
            "unique_weights": {
                "unique_weight_bytes": 900,
                "measurement_method": "static_tensor_accounting",
                "measurement_source": "model tensor manifest",
                "measurement_confidence": "high",
            },
            "observed_direct_ddr": {
                "observed_direct_ddr_read_bytes": 800,
                "observed_direct_ddr_write_bytes": 80,
                "measurement_method": "pmu_counter_delta",
                "measurement_source": "A733 DDR PMU",
                "measurement_confidence": "medium",
            },
            "inferred_ddr": {
                "inferred_ddr_read_bytes": 820,
                "inferred_ddr_write_bytes": 90,
                "measurement_method": "counter_model_fit",
                "measurement_source": "PMU calibration model",
                "measurement_confidence": "low",
            },
        },
    }
    (root / "raw/telemetry.jsonl").write_text(json.dumps(telemetry) + "\n", encoding="utf-8")
    (root / "raw/trace.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")

    paths = [
        "README.md",
        "hypothesis-preflight.md",
        "commands.txt",
        "environment.json",
        "device.json",
        "data/branch-preflight.json",
        "results/summary.json",
        "raw/stdout.log",
        "raw/stderr.log",
        "raw/telemetry.jsonl",
        "raw/trace.jsonl",
    ]
    manifest = build_file_manifest(root, paths, status=status)
    for entry in manifest["files"]:
        if entry["path"].startswith("raw/"):
            entry["capture"] = {"captured": True, "empty_reason": None}
            if entry["path"] == "raw/stderr.log":
                entry["capture"]["empty_reason"] = "Команда не записала stderr."
    (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


class E053ExperimentValidatorTest(unittest.TestCase):
    def test_scaffold_creates_a_valid_planned_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "seed.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/prior", "HEAD"], check=True
            )
            root = scaffold_experiment(
                repo / "experiments",
                "E047-common-benchmark",
                repo_root=repo,
                hypothesis_query="сравнить Bonsai и LFM2.5",
            )
            result = validate_experiment(root)
            self.assertTrue(result["valid"], result)
            self.assertEqual("planned", result["status"])
            self.assertTrue(any("\u0400" <= char <= "\u04ff" for char in (root / "README.md").read_text(encoding="utf-8")))
            preflight = json.loads((root / "data/branch-preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(["refs/remotes/origin/prior"], preflight["remote_refs_at_scan"])
            self.assertEqual("сравнить Bonsai и LFM2.5", preflight["hypothesis_query"])
            manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
            self.assertIn("data/branch-preflight.json", {entry["path"] for entry in manifest["files"]})

    def test_planned_scaffold_requires_russian_readme_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            result = validate_experiment(root)
            self.assertTrue(result["valid"], result)

    def test_validator_cli_works_as_documented_direct_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tooling/experiment_validator.py"),
                    "--experiment",
                    str(root),
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_rejects_non_russian_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            (root / "README.md").write_text("# E047\n\nStatus: PLANNED.\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertIn("README.md", " ".join(result["errors"]))

    def test_provenance_files_need_machine_readable_capture_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            (root / "environment.json").write_text("{}\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("environment.json" in error for error in result["errors"]))

    def test_valid_passed_rejected_and_failed_keep_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            for status in ("passed", "rejected", "failed"):
                with self.subTest(status=status):
                    root = Path(tmp) / status
                    root.mkdir()
                    write_valid_executed_files(root, status=status)
                    result = validate_experiment(root)
                    self.assertTrue(result["valid"], result)

    def test_summary_status_must_match_at_least_one_run_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root, status="failed")
            summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
            summary["status"] = "passed"
            (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("result=passed" in error for error in result["errors"]))

    def test_executed_rejects_empty_invalid_or_arbitrary_jsonl(self):
        cases = ("", "not-json\n", "{}\n")
        with tempfile.TemporaryDirectory() as tmp:
            for index, payload in enumerate(cases):
                with self.subTest(payload=repr(payload)):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    (root / "raw/trace.jsonl").write_text(payload, encoding="utf-8")
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any("trace.jsonl" in error for error in result["errors"]))

    def test_executed_rejects_captured_false_and_empty_log_without_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            environment["captured"] = False
            (root / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("captured=true" in error for error in result["errors"]))

            root = Path(tmp) / "second"
            root.mkdir()
            write_valid_executed_files(root)
            manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
            stderr = next(entry for entry in manifest["files"] if entry["path"] == "raw/stderr.log")
            stderr["capture"]["empty_reason"] = None
            (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("empty_reason" in error for error in result["errors"]))

    def test_executed_rejects_bad_summary_schema_model_hash_and_prompt_binding(self):
        mutations = (
            ("schema_version", lambda summary: summary.update(schema_version="wrong")),
            ("model sha256", lambda summary: summary["runs"][0]["model"].update(sha256=None)),
            ("common prompt", lambda summary: summary["runs"][0]["workload"].update(common_prompt_text="other")),
            ("rendered prompt hash", lambda summary: summary["runs"][0]["workload"].update(rendered_prompt_sha256="0" * 64)),
            ("token IDs hash", lambda summary: summary["runs"][0]["workload"].update(prompt_token_ids_sha256="0" * 64)),
            ("command workload hash", lambda summary: summary["runs"][0]["command"].update(workload_sha256="0" * 64)),
            ("observed/inferred conflation", lambda summary: summary["runs"][0]["memory_accounting"]["observed_direct_ddr"].update(measurement_method="counter_model_fit")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (name, mutate) in enumerate(mutations):
                with self.subTest(case=name):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
                    mutate(summary)
                    (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])

    def test_executed_rejects_incomplete_runtime_compiler_driver_kernel_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            del environment["runtime"]["commit"]
            (root / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            (root / "commands.txt").write_text("different command\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            joined = " ".join(result["errors"])
            self.assertIn("runtime.commit", joined)
            self.assertIn("commands.txt", joined)

    def test_memory_classes_reject_cross_class_byte_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
            summary["runs"][0]["memory_accounting"]["observed_direct_ddr"][
                "inferred_ddr_read_bytes"
            ] = 999
            (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("смешивает" in error for error in result["errors"]))

    def test_executed_experiment_requires_raw_trace_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for status in ("failed", "rejected"):
                with self.subTest(status=status):
                    root = Path(tmp) / status
                    root.mkdir()
                    write_required_planned_files(root)
                    (root / "results/summary.json").write_text(
                        json.dumps({"status": status}) + "\n", encoding="utf-8"
                    )
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any("raw/" in error for error in result["errors"]))

    def test_manifest_hash_must_match_referenced_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            (root / "raw").mkdir()
            (root / "raw/stdout.log").write_text("провал\n", encoding="utf-8")
            (root / "raw/stderr.log").write_text("\n", encoding="utf-8")
            (root / "raw/telemetry.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "raw/trace.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "results/summary.json").write_text(
                json.dumps({"status": "failed"}) + "\n", encoding="utf-8"
            )
            manifest = {
                "schema_version": "e053-experiment-manifest/v1",
                "status": "failed",
                "files": [{"path": "raw/stdout.log", "sha256": "0" * 64}],
            }
            (root / "data/manifest.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("sha256" in error for error in result["errors"]))

    def test_manifest_must_hash_required_provenance_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
            manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "device.json"]
            (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("device.json" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
