"""Adversarial Git-sealing and raw-ingestion gates for E055."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tooling.e055_q1_hotcold import load_publication_contract
from tooling.e055_raw_bundle import load_sealed_bundle, seal_bundle_files


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_BINARY = (
    ROOT / "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64"
)
PHASE_RELATIVE = Path("experiments/E055-q1-hot-cold/raw/phase-a")
ROLE_FILES = {
    "harness_stdout": "harness.stdout.capture.json",
    "harness_stderr": "harness.stderr.capture.json",
    "e049c_json": "e049c.json",
    "e049c_stderr": "e049c.stderr.capture.json",
    "runner_metadata": "runner.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BundleFixture:
    def __init__(self, run_specs: list[dict] | None = None) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e055-sealed-")
        self.root = Path(self.temp.name)
        self.phase = self.root / PHASE_RELATIVE
        self.run_dir = self.phase / "runs/run-a"
        self.manifest = self.phase / "bundle.json"
        self.contract = load_publication_contract()
        self.publication_manifest_sha256 = sha256(
            ROOT / "experiments/E055-q1-hot-cold/data/manifest.json"
        )
        self.git("init", "-q")
        self.git("config", "user.email", "e055-test@example.invalid")
        self.git("config", "user.name", "E055 Test")
        self.run_dir.parent.mkdir(parents=True)
        artifact_dir = self.phase / "artifacts"
        artifact_dir.mkdir()
        self.run_specs = run_specs or [{
            "run_id": "run-a", "pair_id": "pair-a", "pair_index": 1,
            "pair_order": "hot_then_cold", "order_index": 1,
            "build_name": "O3", "mode": "full_dotprod",
            "cache_state": "hot_repeat", "cpu": 6,
            "target_working_set_bytes": 65536, "pmu_group": "core",
        }]
        for build_name in sorted({spec["build_name"] for spec in self.run_specs}):
            source = ROOT / "experiments/E055-q1-hot-cold/artifacts" / (
                "e055-O3-aarch64" if build_name == "O3"
                else "e055-O3-flto-aarch64"
            )
            shutil.copyfile(source, artifact_dir / f"harness-{build_name}.bin")
        run_entries = [self._write_run(spec) for spec in self.run_specs]
        self.run_dir = self.phase / "runs" / self.run_specs[0]["run_id"]
        self.payload = self._payload(run_entries)
        self.write_manifest()
        self.commit("valid sealed bundle")

    def close(self) -> None:
        self.temp.cleanup()

    def __enter__(self) -> "BundleFixture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    def artifact(self, role: str, path: Path) -> dict:
        relative = path.relative_to(self.root).as_posix()
        return {
            "role": role,
            "path": relative,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "git_blob_oid": self.git("hash-object", relative),
        }

    def _stream_capture(
        self, run_id: str, producer: str, stream: str, payload: bytes,
    ) -> bytes:
        document = {
            "schema": "e055-stream-capture/v1",
            "run_id": run_id,
            "producer": producer,
            "stream": stream,
            "encoding": "base64",
            "payload_size_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    def _write_run(self, spec: dict) -> dict:
        run_id = spec["run_id"]
        run_dir = self.phase / "runs" / run_id
        run_dir.mkdir()
        target = spec["target_working_set_bytes"]
        blocks = (target + 207) // 208
        actual = blocks * 208
        calls = 1 if spec["cache_state"] == "cold_conditioned" else 250
        elapsed_ns = calls * 1_000_000
        if spec["cache_state"] == "cold_conditioned":
            conditioning = {
                "strategy": "verified_write_read_each_64B_line",
                "requested_bytes": 64 * 1024 * 1024,
                "actual_bytes": 64 * 1024 * 1024,
                "line_bytes": 64,
                "lines_touched": 1024 * 1024,
                "checksum": "0xe055c001",
                "verified_touched": True,
                "warmup_calls": 0,
            }
        else:
            conditioning = {
                "strategy": "verified_kernel_warmup",
                "requested_bytes": actual,
                "actual_bytes": actual,
                "line_bytes": 64,
                "lines_touched": (actual + 63) // 64,
                "checksum": "0xe055c002",
                "verified_touched": True,
                "warmup_calls": 16,
            }
        stdout = {
            "schema": "e055-q1-hot-cold-harness/v1",
            "mode": spec["mode"],
            "cache_state": spec["cache_state"],
            "cpu": spec["cpu"],
            "q1_layout": "E039 stock native block_q1_0x4 4x4 DOTPROD",
            "golden_pass": True,
            "golden_cases": 18,
            "target_working_set_bytes": target,
            "actual_working_set_bytes": actual,
            "blocks": blocks,
            "iterations": calls,
            "calls": calls,
            "elapsed_ns": elapsed_ns,
            "first_call_ns": 1_000_000,
            "calls_per_second": 1000.0,
            "logical_bytes_per_call": {
                "q1_packed_bytes": blocks * 72,
                "q8_bytes": blocks * 4 * 34,
                "total_input_bytes": blocks * 208,
                "output_bytes": 16,
                "dot_products": blocks * 512,
            },
            "checksum": "0xe055d001",
            "cold_conditioning": conditioning,
            "sync": {
                "requested": True, "started": True, "acknowledged": True,
                "ended": True, "sequence": "S/A/E",
            },
            "qualification": "unqualified_harness_output_requires_E049c_join",
        }
        binary_relative = (
            PHASE_RELATIVE / "artifacts" / f"harness-{spec['build_name']}.bin"
        ).as_posix()
        argv = [
            "taskset", "-c", str(spec["cpu"]), binary_relative,
            "--mode", spec["mode"], "--cache-state", spec["cache_state"],
            "--working-set-bytes", str(target), "--cpu", str(spec["cpu"]),
            "--iterations", str(calls),
        ]
        if spec["cache_state"] == "cold_conditioned":
            argv.extend(["--budget-ms", "0", "--warmup", "0"])
        else:
            argv.extend(["--warmup", "16"])
        argv.extend(["--thrash-bytes", str(64 * 1024 * 1024), "--sync"])
        configs = self.contract["pmu_group_configs"][spec["pmu_group"]]
        e049c = {
            "schema_version": "e049c-arm-pmu/v2",
            "status": "ok",
            "sample_valid": True,
            "event_source": "armv8_pmuv3_raw_config",
            "counter_semantics": "event counts only; no DDR-byte conversion",
            "event_group": spec["pmu_group"],
            "software_group_size_limit": 4,
            "event_group_size": len(configs),
            "software_group_size_limit_semantics": (
                "conservative launcher policy; not measured hardware PMU capacity"
            ),
            "min_running_ratio": 0.95,
            "pid": 1000 + spec["pair_index"],
            "process_group": 1000 + spec["pair_index"],
            "command": argv,
            "sync": {
                "mode": "start_ack_end", "started": True,
                "acknowledged": True, "ended": True,
            },
            "measured_elapsed_ns": elapsed_ns + 1000,
            "thermal": {
                "guard_enabled": True, "checked": True, "readable": True,
                "limit_c": 80.0, "max_observed_c": 55.0, "tripped": False,
            },
            "exit": {"code": 0, "raw_wait_status": 0},
            "failure_reason": None,
            "events": [
                {
                    "name": name, "config": config,
                    "meaning": f"ARMv8 PMUv3 {name} event count",
                    "support": "supported", "count_semantics": "event_count_not_bytes",
                    "sample_valid": True, "value": 100 + index,
                    "time_enabled_ns": elapsed_ns, "time_running_ns": elapsed_ns,
                    "running_ratio": 1.0, "errno": 0, "error": None,
                }
                for index, (name, config) in enumerate(configs.items())
            ],
        }
        harness_stdout = run_dir / ROLE_FILES["harness_stdout"]
        harness_stderr = run_dir / ROLE_FILES["harness_stderr"]
        e049c_path = run_dir / ROLE_FILES["e049c_json"]
        e049c_stderr = run_dir / ROLE_FILES["e049c_stderr"]
        runner_path = run_dir / ROLE_FILES["runner_metadata"]
        stdout_payload = json.dumps(stdout, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        harness_stdout.write_bytes(
            self._stream_capture(run_id, "e055_harness", "stdout", stdout_payload)
        )
        harness_stderr.write_bytes(
            self._stream_capture(run_id, "e055_harness", "stderr", b"")
        )
        e049c_path.write_text(
            json.dumps(e049c, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        e049c_stderr.write_bytes(
            self._stream_capture(run_id, "e049c_launcher", "stderr", b"")
        )
        provenance = {
            "source_sha256": self.contract["source_sha256"],
            "binary_sha256": self.contract["allowed_builds"][spec["build_name"]],
            "compiler_sha256": self.contract["compiler_sha256"],
            "compiler_id": self.contract["compiler_id"],
            "pmu_source_sha256": self.contract["pmu_source_sha256"],
            "upstream_commit": self.contract["upstream_commit"],
            "upstream_ref": self.contract["upstream_ref"],
            "upstream_repack_sha256": self.contract["upstream_repack_sha256"],
            "publication_manifest_sha256": self.publication_manifest_sha256,
        }
        runner = {
            "schema": "e055-runner-capture/v1",
            "run_id": run_id,
            "pair_id": spec["pair_id"],
            "pair_index": spec["pair_index"],
            "pair_order": spec["pair_order"],
            "order_index": spec["order_index"],
            "build_name": spec["build_name"],
            "argv": argv,
            "environment": {
                "LC_ALL": "C", "LANG": "C", "E055_BUILD_NAME": spec["build_name"],
            },
            "affinity": {
                "requested_cpus": [spec["cpu"]], "effective_cpus": [spec["cpu"]],
                "cpu_start": spec["cpu"], "cpu_end": spec["cpu"],
                "migration_count": 0,
            },
            "exit": {"code": 0, "signal": None},
            "provenance": provenance,
            "artifact_sha256": {
                "harness_stdout": sha256(harness_stdout),
                "harness_stderr": sha256(harness_stderr),
                "e049c_json": sha256(e049c_path),
                "e049c_stderr": sha256(e049c_stderr),
            },
            "target_workload_executed": True,
        }
        runner_path.write_text(
            json.dumps(runner, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "pair_id": spec["pair_id"],
            "pair_index": spec["pair_index"],
            "pair_order": spec["pair_order"],
            "order_index": spec["order_index"],
            "build_name": spec["build_name"],
            "cell": {
                "mode": spec["mode"], "cache_state": spec["cache_state"],
                "cpu": spec["cpu"], "target_working_set_bytes": target,
                "actual_working_set_bytes": actual, "blocks": blocks,
                "pmu_group": spec["pmu_group"],
            },
            "artifacts": {
                role: self.artifact(role, run_dir / name)
                for role, name in ROLE_FILES.items()
            },
        }

    def _payload(self, run_entries: list[dict]) -> dict:
        build_artifacts = []
        for build_name in sorted({spec["build_name"] for spec in self.run_specs}):
            binary = self.phase / "artifacts" / f"harness-{build_name}.bin"
            build_artifacts.append({
                "build_name": build_name,
                "artifact": self.artifact("harness_executable", binary),
            })
        qualification = {
            "source_sha256": self.contract["source_sha256"],
            "compiler_sha256": self.contract["compiler_sha256"],
            "compiler_id": self.contract["compiler_id"],
            "pmu_source_sha256": self.contract["pmu_source_sha256"],
            "upstream_commit": self.contract["upstream_commit"],
            "upstream_ref": self.contract["upstream_ref"],
            "upstream_repack_sha256": self.contract["upstream_repack_sha256"],
            "publication_manifest_sha256": self.publication_manifest_sha256,
        }
        return {
            "schema": "e055-raw-bundle/v1",
            "experiment": "E055-Q1-HOT-COLD",
            "phase_id": "phase-a",
            "qualification": qualification,
            "build_artifacts": build_artifacts,
            "runs": run_entries,
            "target_workload_executed": True,
        }

    def write_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(self.payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def role_path(self, role: str, run_index: int = 0) -> Path:
        declaration = self.payload["runs"][run_index]["artifacts"][role]
        return self.root / declaration["path"]

    def reseal_roles(self, roles: set[str], run_index: int = 0) -> None:
        run = self.payload["runs"][run_index]
        if "runner_metadata" not in roles:
            runner_path = self.role_path("runner_metadata", run_index)
            runner = json.loads(runner_path.read_text(encoding="utf-8"))
            for role in roles:
                runner["artifact_sha256"][role] = sha256(self.role_path(role, run_index))
            runner_path.write_text(
                json.dumps(runner, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            roles = set(roles) | {"runner_metadata"}
        for role in roles:
            run["artifacts"][role] = self.artifact(
                role, self.role_path(role, run_index)
            )
        self.write_manifest()
        self.commit("reseal adversarial raw mutation")

    def mutate_e049c(self, mutate: object, run_index: int = 0) -> None:
        path = self.role_path("e049c_json", run_index)
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.reseal_roles({"e049c_json"}, run_index)

    def mutate_runner(self, mutate: object, run_index: int = 0) -> None:
        path = self.role_path("runner_metadata", run_index)
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.reseal_roles({"runner_metadata"}, run_index)

    def mutate_harness_stdout(self, mutate: object, run_index: int = 0) -> None:
        path = self.role_path("harness_stdout", run_index)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = base64.b64decode(envelope["payload_base64"], validate=True)
        document = json.loads(payload.decode("utf-8"))
        mutate(document)
        new_payload = (
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        path.write_bytes(
            self._stream_capture(document.get("run_id", self.run_specs[run_index]["run_id"]),
                                 "e055_harness", "stdout", new_payload)
        )
        self.reseal_roles({"harness_stdout"}, run_index)

    def set_stream_payload(self, role: str, payload: bytes, run_index: int = 0) -> None:
        producer = "e055_harness" if role == "harness_stderr" else "e049c_launcher"
        path = self.role_path(role, run_index)
        path.write_bytes(
            self._stream_capture(self.run_specs[run_index]["run_id"], producer,
                                 "stderr", payload)
        )
        self.reseal_roles({role}, run_index)


class E055SealedArtifactTest(unittest.TestCase):
    def test_committed_regular_bundle_is_sealed_to_head_and_tree(self) -> None:
        with BundleFixture() as fixture:
            sealed = seal_bundle_files(fixture.manifest)
            self.assertEqual(sealed.commit, fixture.git("rev-parse", "HEAD"))
            self.assertEqual(sealed.tree, fixture.git("rev-parse", "HEAD^{tree}"))
            self.assertEqual(len(sealed.artifacts), 6)
            self.assertEqual(sealed.manifest.relative_path,
                             (PHASE_RELATIVE / "bundle.json").as_posix())

    def test_dirty_staged_untracked_and_uncommitted_manifest_fail_closed(self) -> None:
        mutations = (
            "dirty_artifact", "staged_artifact", "untracked_extra", "dirty_manifest",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), BundleFixture() as fixture:
                artifact = fixture.run_dir / ROLE_FILES["e049c_json"]
                if mutation in ("dirty_artifact", "staged_artifact"):
                    artifact.write_bytes(artifact.read_bytes() + b"appended")
                    if mutation == "staged_artifact":
                        fixture.git("add", artifact.relative_to(fixture.root).as_posix())
                elif mutation == "untracked_extra":
                    (fixture.phase / "untracked.bin").write_bytes(b"not declared")
                else:
                    fixture.manifest.write_bytes(fixture.manifest.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    seal_bundle_files(fixture.manifest)


class E055RawParserTest(unittest.TestCase):
    def test_canonical_argv_uses_real_harness_options_and_cold_contract(self) -> None:
        from tooling.e055_raw_bundle import canonical_harness_argv

        base = {
            "mode": "full_dotprod", "cache_state": "hot_repeat", "cpu": 6,
            "target_working_set_bytes": 65536,
        }
        blocks = (base["target_working_set_bytes"] + 207) // 208
        base.update({
            "actual_working_set_bytes": blocks * 208,
            "blocks": blocks, "pmu_group": "core",
        })
        hot = dict(base)
        cold = {**base, "cache_state": "cold_conditioned"}
        hot_argv = canonical_harness_argv(hot, "bin/e055", 250)
        cold_argv = canonical_harness_argv(cold, "bin/e055", 1)
        self.assertIn("--cpu", hot_argv)
        self.assertNotIn("--cpu-label", hot_argv)
        self.assertNotIn("--budget-ms", hot_argv)
        self.assertEqual(hot_argv[hot_argv.index("--warmup") + 1], "16")
        self.assertEqual(cold_argv[cold_argv.index("--iterations") + 1], "1")
        self.assertEqual(cold_argv[cold_argv.index("--budget-ms") + 1], "0")
        self.assertEqual(cold_argv[cold_argv.index("--warmup") + 1], "0")

    def test_valid_sealed_raw_capture_derives_one_immutable_sample(self) -> None:
        with BundleFixture() as fixture:
            bundle = load_sealed_bundle(fixture.manifest)
            self.assertEqual(len(bundle.samples), 1)
            sample = bundle.samples[0]
            self.assertEqual(sample.run_id, "run-a")
            self.assertEqual(sample.mode, "full_dotprod")
            self.assertEqual(sample.commit, fixture.git("rev-parse", "HEAD"))
            self.assertEqual(sample.tree, fixture.git("rev-parse", "HEAD^{tree}"))
            self.assertEqual(len(sample.raw_artifacts), 5)

    def test_launcher_floor_does_not_relax_qualification_ratio(self) -> None:
        with BundleFixture() as fixture:
            fixture.mutate_e049c(
                lambda raw: raw["events"][0].update(running_ratio=0.95)
            )
            with self.assertRaisesRegex(ValueError, "exactly 1.0"):
                load_sealed_bundle(fixture.manifest)

    def test_thermal_timing_golden_sync_and_stderr_forgeries_fail_closed(self) -> None:
        mutations = {
            "thermal": lambda f: f.mutate_e049c(
                lambda raw: raw["thermal"].update(max_observed_c=81.0)
            ),
            "timing": lambda f: f.mutate_e049c(
                lambda raw: raw.update(measured_elapsed_ns=999)
            ),
            "golden": lambda f: f.mutate_harness_stdout(
                lambda raw: raw.update(golden_cases=17)
            ),
            "sync": lambda f: f.mutate_e049c(
                lambda raw: raw["sync"].update(ended=False)
            ),
            "harness stderr": lambda f: f.set_stream_payload(
                "harness_stderr", b"unexpected stderr\n"
            ),
            "E049c stderr": lambda f: f.set_stream_payload(
                "e049c_stderr", b"unexpected stderr\n"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), BundleFixture() as fixture:
                mutate(fixture)
                with self.assertRaises(ValueError):
                    load_sealed_bundle(fixture.manifest)

    def test_runner_unknown_or_secret_environment_and_cpu_forgery_fail_closed(self) -> None:
        mutations = {
            "unknown env": lambda raw: raw["environment"].update(EXTRA="value"),
            "secret env": lambda raw: raw["environment"].update(API_TOKEN="secret"),
            "Boolean CPU": lambda raw: raw["affinity"].update(cpu_start=True),
            "migration": lambda raw: raw["affinity"].update(migration_count=1),
            "forged build hash": lambda raw: raw["provenance"].update(
                binary_sha256="f" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), BundleFixture() as fixture:
                fixture.mutate_runner(mutate)
                with self.assertRaises(ValueError):
                    load_sealed_bundle(fixture.manifest)

    def test_raw_files_cannot_be_swapped_between_pairs_or_builds(self) -> None:
        specs = [
            {
                "run_id": "run-a", "pair_id": "pair-a", "pair_index": 1,
                "pair_order": "hot_then_cold", "order_index": 1,
                "build_name": "O3", "mode": "full_dotprod",
                "cache_state": "hot_repeat", "cpu": 6,
                "target_working_set_bytes": 65536, "pmu_group": "core",
            },
            {
                "run_id": "run-b", "pair_id": "pair-b", "pair_index": 2,
                "pair_order": "cold_then_hot", "order_index": 2,
                "build_name": "O3-flto", "mode": "full_dotprod",
                "cache_state": "hot_repeat", "cpu": 6,
                "target_working_set_bytes": 65536, "pmu_group": "core",
            },
        ]
        with BundleFixture(specs) as fixture:
            first = fixture.role_path("e049c_json", 0)
            second = fixture.role_path("e049c_json", 1)
            first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
            first.write_bytes(second_bytes)
            second.write_bytes(first_bytes)
            for index in (0, 1):
                run = fixture.payload["runs"][index]
                run["artifacts"]["e049c_json"] = fixture.artifact(
                    "e049c_json", fixture.role_path("e049c_json", index)
                )
            fixture.write_manifest()
            fixture.commit("swap E049c files across pairs and builds")
            with self.assertRaises(ValueError):
                load_sealed_bundle(fixture.manifest)

    def test_truncate_append_role_swap_and_duplicate_blob_fail_closed(self) -> None:
        for mutation in ("truncate", "append", "role_swap", "duplicate"):
            with self.subTest(mutation=mutation), BundleFixture() as fixture:
                artifacts = fixture.payload["runs"][0]["artifacts"]
                if mutation in ("truncate", "append"):
                    path = fixture.root / artifacts["e049c_json"]["path"]
                    path.write_bytes(
                        path.read_bytes()[:4] if mutation == "truncate"
                        else path.read_bytes() + b"appended"
                    )
                    fixture.commit(mutation)
                elif mutation == "role_swap":
                    artifacts["e049c_json"], artifacts["runner_metadata"] = (
                        artifacts["runner_metadata"], artifacts["e049c_json"]
                    )
                    fixture.write_manifest()
                    fixture.commit(mutation)
                else:
                    artifacts["runner_metadata"] = copy.deepcopy(artifacts["e049c_json"])
                    artifacts["runner_metadata"]["role"] = "runner_metadata"
                    fixture.write_manifest()
                    fixture.commit(mutation)
                with self.assertRaises(ValueError):
                    seal_bundle_files(fixture.manifest)

    def test_path_traversal_symlink_hardlink_and_oversize_claim_fail_closed(self) -> None:
        for mutation in ("traversal", "symlink", "hardlink", "oversize"):
            with self.subTest(mutation=mutation), BundleFixture() as fixture:
                declared = fixture.payload["runs"][0]["artifacts"]["e049c_json"]
                path = fixture.root / declared["path"]
                if mutation == "traversal":
                    declared["path"] = (PHASE_RELATIVE / "runs/../outside.json").as_posix()
                    fixture.write_manifest()
                    fixture.commit(mutation)
                elif mutation == "oversize":
                    declared["size_bytes"] = 256 * 1024 + 1
                    fixture.write_manifest()
                    fixture.commit(mutation)
                else:
                    replacement = fixture.phase / f"{mutation}-replacement"
                    if mutation == "symlink":
                        path.unlink()
                        os.symlink("../runner.json", path)
                    else:
                        replacement.write_bytes(path.read_bytes())
                        path.unlink()
                        os.link(replacement, path)
                with self.assertRaises(ValueError):
                    seal_bundle_files(fixture.manifest)


if __name__ == "__main__":
    unittest.main()
