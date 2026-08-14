import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tooling.branch_preflight import (
    build_manifest,
    duplicate_decision_from_refs,
    tree_binding_sha256,
)
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def refresh_manifest_hash(root: Path, relative: str) -> None:
    path = root / relative
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["size_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


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
                "trace_ab_results": [],
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
    workload_identity = {
        "variant_id": "bonsai-q1-gguf-cpu",
        "contract_sha256": "d" * 64,
        "common_prompt_sha256": COMMON_PROMPT_SHA256,
        "rendered_prompt_sha256": sha256_text(rendered),
        "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "n_predict": 4,
        "seed": 123,
    }
    workload_identity_sha = sha256_text(
        json.dumps(workload_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    runs = []
    sides = {"trace_off": [], "trace_on": []}
    raw_ab_paths = []
    for mode in ("off", "on"):
        for sample_index in range(1, 6):
            pair_id = f"pair-{sample_index}"
            run_id = f"{mode}-{sample_index}"
            raw_path = f"raw/ab/{run_id}.{'trace.jsonl' if mode == 'on' else 'stdout.log'}"
            raw_ab_paths.append(raw_path)
            path = root / raw_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"artifact {run_id}\n", encoding="utf-8")
            runs.append(
                {
                    "run_id": run_id,
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
                    "trace_ab": {
                        "result_path": "results/trace-ab.json",
                        "pair_id": pair_id,
                        "mode": mode,
                    },
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
            )
            sides[f"trace_{mode}"].append(
                {
                    "pair_id": pair_id,
                    "run_id": run_id,
                    "latency_ms": 1000.0 if mode == "off" else 1005.0,
                    "token_ids_sha256": "e" * 64,
                    "raw_artifacts": [raw_path],
                }
            )
    trace_ab = {
        "schema_version": "e047-trace-ab-result/v1",
        "ab_result_id": "test-trace-ab",
        "workload_identity": workload_identity,
        "workload_identity_sha256": workload_identity_sha,
        "trace_off": {"workload_identity_sha256": workload_identity_sha, "samples": sides["trace_off"]},
        "trace_on": {"workload_identity_sha256": workload_identity_sha, "samples": sides["trace_on"]},
        "statistics": {
            "off_median_ms": 1000.0,
            "on_median_ms": 1005.0,
            "off_p95_ms": 1000.0,
            "on_p95_ms": 1005.0,
            "median_overhead_fraction": 0.005,
            "p95_overhead_fraction": 0.005,
        },
        "token_stream_equal": True,
        "classification": "PASS",
    }
    (root / "results/trace-ab.json").write_text(json.dumps(trace_ab) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "e053-experiment-summary/v1",
        "experiment_id": "E047",
        "status": status,
        "generated_at_utc": "2026-08-14T00:01:00+00:00",
        "reason": "Тестовая фиксация результата.",
        "model_used": True,
        "trace_ab_results": [
            {"path": "results/trace-ab.json", "sha256": sha256_text(json.dumps(trace_ab) + "\n")}
        ],
        "runs": runs,
    }
    (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    (root / "raw").mkdir(exist_ok=True)
    (root / "raw/stdout.log").write_text("model output\n", encoding="utf-8")
    (root / "raw/stderr.log").write_text("", encoding="utf-8")
    telemetry_events = []
    trace_events = []
    trace_template = {
        "schema_version": "e047-trace-event/v1",
        "step": 0,
        "phase": "decode",
        "token_index": 0,
        "layer_index": 0,
        "layer_scope": "layer",
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
    for index, run in enumerate(runs, start=1):
        telemetry_events.append(
            {
                "schema_version": "e047-telemetry-event/v1",
                "run_id": run["run_id"],
                "timestamp_ns": index,
                "cpu_temperature_c": 42.0,
                "cpu_frequency_hz": [2208000000, 2208000000],
                "npu_frequency_hz": 1008000000,
                "rss_bytes": 1024,
                "ddr": dict(run["memory_accounting"]["observed_direct_ddr"]),
            }
        )
        trace = json.loads(json.dumps(trace_template))
        trace["run_id"] = run["run_id"]
        trace_events.append(trace)
    write_jsonl(root / "raw/telemetry.jsonl", telemetry_events)
    write_jsonl(root / "raw/trace.jsonl", trace_events)

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
        "results/trace-ab.json",
        *raw_ab_paths,
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

    def test_executed_summary_requires_linked_trace_ab_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
            summary.pop("trace_ab_results", None)
            (root / "results/summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("trace_ab_results" in error for error in result["errors"]))

    def test_trace_ab_rejects_missing_unmanifested_or_hash_mismatched_raw_artifact(self):
        mutations = ("missing", "unmanifested", "hash_mismatch")
        with tempfile.TemporaryDirectory() as tmp:
            for index, mutation in enumerate(mutations):
                with self.subTest(mutation=mutation):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
                    trace_link = summary["trace_ab_results"][0]
                    trace_ab = json.loads((root / trace_link["path"]).read_text(encoding="utf-8"))
                    raw_path = trace_ab["trace_off"]["samples"][0]["raw_artifacts"][0]
                    if mutation == "missing":
                        (root / raw_path).unlink()
                    elif mutation == "unmanifested":
                        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
                        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != raw_path]
                        (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
                    else:
                        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
                        entry = next(entry for entry in manifest["files"] if entry["path"] == raw_path)
                        entry["sha256"] = "0" * 64
                        (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any("trace A/B raw" in error or raw_path in error for error in result["errors"]))

    def test_trace_ab_cross_validates_run_pair_mode_and_workload_links(self):
        mutations = (
            ("run", lambda value: value["trace_on"]["samples"][0].update(run_id="unknown")),
            ("pair", lambda value: value["trace_on"]["samples"][0].update(pair_id="wrong")),
            ("workload", lambda value: value["trace_on"].update(workload_identity_sha256="0" * 64)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (name, mutate) in enumerate(mutations):
                with self.subTest(case=name):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
                    link = summary["trace_ab_results"][0]
                    path = root / link["path"]
                    value = json.loads(path.read_text(encoding="utf-8"))
                    mutate(value)
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any("trace A/B" in error or "INVALID" in error for error in result["errors"]))

    def test_trace_event_strict_types_ranges_backend_and_nullable_layer_gate(self):
        mutations = (
            ("bool token", lambda event: event.update(token_index=True)),
            ("negative op", lambda event: event.update(op_index=-1)),
            ("bad backend", lambda event: event.update(backend="magic")),
            ("empty op", lambda event: event.update(op_name="")),
            ("bad timestamps", lambda event: event.update(start_ns=20, end_ns=10, duration_ns=-10)),
            ("null layer without global", lambda event: event.update(layer_index=None)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (name, mutate) in enumerate(mutations):
                with self.subTest(case=name):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    path = root / "raw/trace.jsonl"
                    events = read_jsonl(path)
                    mutate(events[0])
                    write_jsonl(path, events)
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any("trace.jsonl" in error for error in result["errors"]))

    def test_memory_and_telemetry_reject_bool_nan_ranges_and_invalid_null_methods(self):
        mutations = (
            ("bool logical", "trace", lambda event: event["memory"]["logical"].update(logical_read_bytes=True)),
            ("negative unique", "trace", lambda event: event["memory"]["unique_weights"].update(unique_weight_bytes=-1)),
            ("null direct method", "trace", lambda event: event["memory"]["observed_direct_ddr"].update(observed_direct_ddr_read_bytes=None)),
            ("nan temperature", "telemetry", lambda event: event.update(cpu_temperature_c=float("nan"))),
            ("temperature range", "telemetry", lambda event: event.update(cpu_temperature_c=999.0)),
            ("negative frequency", "telemetry", lambda event: event.update(npu_frequency_hz=-1)),
            ("direct missing counter", "telemetry", lambda event: event["ddr"].update(observed_direct_ddr_read_bytes=None)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (name, stream, mutate) in enumerate(mutations):
                with self.subTest(case=name):
                    root = Path(tmp) / str(index)
                    root.mkdir()
                    write_valid_executed_files(root)
                    path = root / f"raw/{stream}.jsonl"
                    events = read_jsonl(path)
                    mutate(events[0])
                    write_jsonl(path, events)
                    result = validate_experiment(root)
                    self.assertFalse(result["valid"])
                    self.assertTrue(any(f"{stream}.jsonl" in error for error in result["errors"]))

    def test_preflight_invalidates_changed_ref_set_commit_and_forged_duplicate_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
            subprocess.run(["git", "-C", str(repo), "branch", "other"], check=True)
            root = scaffold_experiment(
                repo / "experiments",
                "E047-review",
                repo_root=repo,
                hypothesis_query="проверить ref snapshot",
            )
            baseline_result = validate_experiment(root)
            self.assertTrue(baseline_result["valid"], baseline_result)

            subprocess.run(["git", "-C", str(repo), "branch", "added-later"], check=True)
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("ref snapshot" in error for error in result["errors"]))

            subprocess.run(["git", "-C", str(repo), "branch", "-D", "added-later"], check=True, capture_output=True)
            preflight_path = root / "data/branch-preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["ref_snapshot"][0]["commit"] = "0" * 40
            preflight["duplicate_decision"] = {
                "status": "duplicate_found",
                "experiment_id": "E047-REVIEW",
                "candidate_count": 1,
                "candidates": [{"experiment_id": "E047-REVIEW", "path": "fake", "ref": "fake", "commit": "0" * 40}],
            }
            preflight_path.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
            result = validate_experiment(root)
            self.assertFalse(result["valid"])
            joined = " ".join(result["errors"])
            self.assertIn("commit", joined)
            self.assertIn("duplicate_decision", joined)

    def test_rejects_cross_stream_direct_ddr_contradiction_for_same_run(self):
        """Один run_id не может заявить два разных прямых PMU-результата."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            telemetry_path = root / "raw/telemetry.jsonl"
            events = read_jsonl(telemetry_path)
            events[0]["ddr"]["observed_direct_ddr_read_bytes"] = 801
            write_jsonl(telemetry_path, events)
            refresh_manifest_hash(root, "raw/telemetry.jsonl")

            result = validate_experiment(root)
            self.assertFalse(result["valid"], result)
            self.assertTrue(any("direct DDR" in error and "off-1" in error for error in result["errors"]))

    def test_accepts_documented_trace_partitions_that_sum_to_run_ddr_total(self):
        """Несколько op-событий допустимы, только если их PMU-части явно суммируются."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_executed_files(root)
            trace_path = root / "raw/trace.jsonl"
            events = read_jsonl(trace_path)
            first = events[0]
            relation = {
                "kind": "partition_of_run_total",
                "run_total_source": "results/summary.json:runs[].memory_accounting.observed_direct_ddr",
            }
            first["memory"]["observed_direct_ddr"].update(
                observed_direct_ddr_read_bytes=400,
                observed_direct_ddr_write_bytes=40,
                aggregation_relation=relation,
            )
            second = json.loads(json.dumps(first))
            second.update(op_index=1, start_ns=20, end_ns=30, duration_ns=10)
            events.insert(1, second)
            write_jsonl(trace_path, events)
            refresh_manifest_hash(root, "raw/trace.jsonl")

            result = validate_experiment(root)
            self.assertTrue(result["valid"], result)

    def test_preflight_independently_rescans_hidden_committed_experiment(self):
        """Подмена сохранённых refs/decision не скрывает новый committed root."""

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "seed.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
            root = scaffold_experiment(
                repo / "experiments",
                "E047-review",
                repo_root=repo,
                hypothesis_query="независимо пересканировать refs",
            )
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            experiment_relative = root.relative_to(repo)
            excludes = [
                (experiment_relative / "data/branch-preflight.json").as_posix(),
                (experiment_relative / "data/manifest.json").as_posix(),
            ]
            preflight = build_manifest(
                repo,
                ["E047", "LFM2.5"],
                experiment_id="E047-review",
                hypothesis_query="независимо пересканировать refs",
                binding_excludes=excludes,
            )
            (root / "data/branch-preflight.json").write_text(json.dumps(preflight) + "\n", encoding="utf-8")
            refresh_manifest_hash(root, "data/branch-preflight.json")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "scaffold"], check=True)
            baseline_result = validate_experiment(root)
            self.assertTrue(baseline_result["valid"], baseline_result)

            hidden = repo / "experiments/E047-review-hidden/README.md"
            hidden.parent.mkdir(parents=True)
            hidden.write_text("# Скрытый E047-review\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "hidden"], check=True)

            preflight_path = root / "data/branch-preflight.json"
            forged = json.loads(preflight_path.read_text(encoding="utf-8"))
            forged["active_worktree"]["tree_binding_sha256"] = tree_binding_sha256(
                repo, excludes
            )
            forged["duplicate_decision"] = duplicate_decision_from_refs(
                forged["refs"], "E047-review"
            )
            preflight_path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
            refresh_manifest_hash(root, "data/branch-preflight.json")

            result = validate_experiment(root)
            self.assertFalse(result["valid"], result)
            self.assertTrue(any("independent ref rescan" in error for error in result["errors"]))

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
