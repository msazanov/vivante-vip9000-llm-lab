"""Adversarial Git-sealing and raw-ingestion gates for E055."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tooling.e055_raw_bundle import seal_bundle_files


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
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="e055-sealed-")
        self.root = Path(self.temp.name)
        self.phase = self.root / PHASE_RELATIVE
        self.run_dir = self.phase / "runs/run-a"
        self.manifest = self.phase / "bundle.json"
        self.git("init", "-q")
        self.git("config", "user.email", "e055-test@example.invalid")
        self.git("config", "user.name", "E055 Test")
        self.run_dir.mkdir(parents=True)
        artifact_dir = self.phase / "artifacts"
        artifact_dir.mkdir()
        shutil.copyfile(PUBLISHED_BINARY, artifact_dir / "harness-O3.bin")
        for role, name in ROLE_FILES.items():
            (self.run_dir / name).write_bytes(f"{role}|run-a\n".encode())
        self.payload = self._payload()
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

    def _payload(self) -> dict:
        binary = self.phase / "artifacts/harness-O3.bin"
        run_artifacts = {
            role: self.artifact(role, self.run_dir / name)
            for role, name in ROLE_FILES.items()
        }
        return {
            "schema": "e055-raw-bundle/v1",
            "experiment": "E055-Q1-HOT-COLD",
            "phase_id": "phase-a",
            "qualification": {
                "source_sha256": "1" * 64,
                "compiler_sha256": "2" * 64,
                "compiler_id": "fixture compiler",
                "pmu_source_sha256": "3" * 64,
                "upstream_commit": "4" * 40,
                "upstream_ref": "https://example.invalid/commit/" + "4" * 40,
                "upstream_repack_sha256": "5" * 64,
                "publication_manifest_sha256": "6" * 64,
            },
            "build_artifacts": [{
                "build_name": "O3",
                "artifact": self.artifact("harness_executable", binary),
            }],
            "runs": [{
                "run_id": "run-a",
                "pair_id": "pair-a",
                "pair_index": 1,
                "pair_order": "hot_then_cold",
                "order_index": 1,
                "build_name": "O3",
                "cell": {
                    "mode": "full_dotprod",
                    "cache_state": "hot_repeat",
                    "cpu": 6,
                    "target_working_set_bytes": 65536,
                    "actual_working_set_bytes": 65728,
                    "blocks": 316,
                    "pmu_group": "core",
                },
                "artifacts": run_artifacts,
            }],
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
