"""Behavioral tests for E055 staged-tree and post-commit publication binding."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import tooling.generate_e055_manifest as generator
import tooling.verify_e055_publication as verifier
from tooling.branch_preflight import tree_binding_sha256


def git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


class E055PublicationBindingTest(unittest.TestCase):
    def test_raw_phase_stays_in_tree_binding_through_commit_and_push(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e055-raw-publication-") as raw:
            workspace = pathlib.Path(raw)
            root = workspace / "work"
            remote = workspace / "remote.git"
            remote.mkdir()
            git(remote, "init", "--bare", "-q")
            root.mkdir()
            git(root, "init", "-q", "-b", "codex/e055-q1-hot-cold")
            git(root, "config", "user.name", "E055 Test")
            git(root, "config", "user.email", "e055@example.invalid")
            git(root, "remote", "add", "origin", str(remote))
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git(root, "add", "base.txt")
            git(root, "commit", "-q", "-m", "base")
            git(root, "push", "-q", "-u", "origin", "codex/e055-q1-hot-cold")
            base = git(root, "rev-parse", "HEAD")

            data = root / "experiments/E055-q1-hot-cold/data"
            phase = root / "experiments/E055-q1-hot-cold/raw/phase-a"
            data.mkdir(parents=True)
            phase.mkdir(parents=True)
            raw_artifact = phase / "capture.bin"
            raw_bundle = phase / "bundle.json"
            raw_artifact.write_bytes(b"immutable raw capture\n")
            raw_bundle.write_text('{"schema":"fixture"}\n', encoding="utf-8")
            git(root, "add", str(phase.relative_to(root)))

            excludes = [
                "experiments/E055-q1-hot-cold/data/branch-preflight.json",
                "experiments/E055-q1-hot-cold/data/manifest.json",
            ]
            self.assertNotIn(raw_artifact.relative_to(root).as_posix(), excludes)
            binding = tree_binding_sha256(root, excludes, source="index")
            preflight = {
                "active_worktree": {
                    "base_commit": base,
                    "ref": "refs/heads/codex/e055-q1-hot-cold",
                    "upstream_ref": (
                        "refs/remotes/origin/codex/e055-q1-hot-cold"
                    ),
                    "tree_binding_source": "git-index-stage0",
                    "tree_binding_sha256": binding,
                    "binding_excludes": excludes,
                }
            }
            artifact_relative = raw_artifact.relative_to(root).as_posix()
            bundle_relative = raw_bundle.relative_to(root).as_posix()
            manifest = {
                "schema": "e055-q1-hot-cold-manifest/v2",
                "publication_binding": {
                    "mode": "e047-staged-index-plus-post-commit-ref-verification",
                    "preflight_base_commit": base,
                    "staged_tree_binding_sha256": binding,
                    "active_ref": "refs/heads/codex/e055-q1-hot-cold",
                    "upstream_ref": (
                        "refs/remotes/origin/codex/e055-q1-hot-cold"
                    ),
                    "binding_excludes": excludes,
                },
                "files": [
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
                    for relative, path in (
                        (artifact_relative, raw_artifact),
                        (bundle_relative, raw_bundle),
                    )
                ],
            }
            preflight_path = data / "branch-preflight.json"
            manifest_path = data / "manifest.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            git(root, "add", str(data.relative_to(root)))
            git(root, "commit", "-q", "-m", "publish raw phase")
            git(root, "push", "-q")

            self.assertEqual(
                verifier.verify_global_publication(
                    root, preflight_path, manifest_path
                ),
                [],
            )
            self.assertEqual(
                binding,
                tree_binding_sha256(root, excludes, source="HEAD"),
            )

            for label, extra_exclude in (
                ("raw", artifact_relative),
                ("code", "base.txt"),
            ):
                with self.subTest(extra_exclusion=label):
                    forged_excludes = [*excludes, extra_exclude]
                    forged_binding = tree_binding_sha256(
                        root, forged_excludes, source="HEAD"
                    )
                    preflight["active_worktree"].update(
                        tree_binding_sha256=forged_binding,
                        binding_excludes=forged_excludes,
                    )
                    manifest["publication_binding"].update(
                        staged_tree_binding_sha256=forged_binding,
                        binding_excludes=forged_excludes,
                    )
                    preflight_path.write_text(
                        json.dumps(preflight), encoding="utf-8"
                    )
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    git(root, "add", str(data.relative_to(root)))
                    git(root, "commit", "-q", "-m", f"forge {label} exclusion")
                    git(root, "push", "-q")
                    errors = verifier.verify_global_publication(
                        root, preflight_path, manifest_path
                    )
                    self.assertTrue(
                        any("binding_excludes" in item for item in errors), errors
                    )

            raw_artifact.write_bytes(b"post-publication mutation\n")
            errors = verifier.verify_global_publication(
                root, preflight_path, manifest_path
            )
            self.assertTrue(any("mismatch" in item or "clean" in item for item in errors))

    def test_only_exact_e055_raw_harness_binaries_are_unignored(self) -> None:
        allowed = (
            "experiments/E055-q1-hot-cold/raw/phase-a/artifacts/harness-O3.bin",
            "experiments/E055-q1-hot-cold/raw/phase-a/artifacts/harness-O3-flto.bin",
        )
        denied = (
            "experiments/E055-q1-hot-cold/raw/phase-a/artifacts/harness-debug.bin",
            "experiments/E055-q1-hot-cold/raw/phase-a/runs/run-a/capture.bin",
            "experiments/E054-other/raw/phase-a/artifacts/harness-O3.bin",
            "artifacts/unrelated.bin",
        )
        for path in allowed:
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", path],
                cwd=generator.ROOT, check=False,
            )
            self.assertEqual(result.returncode, 1, path)
        for path in denied:
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", path],
                cwd=generator.ROOT, check=False,
            )
            self.assertEqual(result.returncode, 0, path)

    def test_publication_includes_every_sealed_bundle_component_and_binary(self) -> None:
        relative = {
            path.relative_to(generator.ROOT).as_posix() for path in generator.PUBLISHED
        }
        required = {
            "experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64",
            "experiments/E055-q1-hot-cold/artifacts/e055-O3-flto-aarch64",
            "experiments/E055-q1-hot-cold/data/raw-bundle.schema.json",
            "experiments/E055-q1-hot-cold/data/runner-capture.schema.json",
            "experiments/E055-q1-hot-cold/data/stream-capture.schema.json",
            "experiments/E055-q1-hot-cold/data/review-rejection-stage4-7679828.md",
            "experiments/E055-q1-hot-cold/data/review-rejection-stage5-f8acab9.md",
            "tooling/e055_raw_bundle.py",
            "tooling/e055_capture_scaffold.py",
            "tests/test_e055_raw_bundle.py",
            "tests/test_e055_sealed_promotion.py",
            "tests/test_e055_capture_scaffold.py",
        }
        self.assertTrue(required.issubset(relative), sorted(required - relative))

    def test_sample_schema_is_derived_output_not_an_analyzer_input(self) -> None:
        schema = json.loads((
            generator.ROOT
            / "experiments/E055-q1-hot-cold/data/sample.schema.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"],
                         "e055-derived-sample/v1")
        self.assertFalse(schema["x-e055-analyzer-input"])
        evidence = schema["properties"]["evidence"]
        self.assertTrue({"commit", "tree", "manifest_path", "manifest_blob_oid",
                         "raw_artifacts"}.issubset(evidence["required"]))

    def test_documented_scripts_bootstrap_when_executed_from_tooling(self) -> None:
        tooling = pathlib.Path(__file__).resolve().parents[1] / "tooling"
        for name in ("generate_e055_manifest.py", "verify_e055_publication.py"):
            with self.subTest(script=name):
                script = tooling / name
                probe = (
                    "import runpy,sys;"
                    f"sys.path=[{str(tooling)!r}]+[p for p in sys.path "
                    f"if p and {str(tooling.parent)!r} not in p];"
                    f"runpy.run_path({str(script)!r},run_name='e055_import_probe')"
                )
                result = subprocess.run(
                    [sys.executable, "-c", probe], text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_uses_staged_tree_binding_not_a_claimed_future_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e055-manifest-") as raw:
            root = pathlib.Path(raw)
            experiment = root / "experiments/E055-q1-hot-cold"
            data = experiment / "data"
            data.mkdir(parents=True)
            source = root / "source.cpp"
            source.write_text("source\n", encoding="utf-8")
            (root / "tooling").mkdir()
            (root / "tooling/a733_pmu_exec.c").write_text(
                "/* PMU fixture */\n", encoding="utf-8"
            )
            artifact_dir = experiment / "artifacts"
            artifact_dir.mkdir()
            pmu_artifact = artifact_dir / "a733-pmu-exec-aarch64"
            pmu_artifact.write_bytes(b"PMU fixture executable\n")
            pmu_source_sha256 = hashlib.sha256(
                (root / "tooling/a733_pmu_exec.c").read_bytes()
            ).hexdigest()
            pmu_binary_sha256 = hashlib.sha256(pmu_artifact.read_bytes()).hexdigest()
            preflight = {
                "active_worktree": {
                    "base_commit": "a" * 40,
                    "ref": "refs/heads/codex/e055-q1-hot-cold",
                    "upstream_ref": "refs/remotes/origin/codex/e055-q1-hot-cold",
                    "tree_binding_source": "git-index-stage0",
                    "tree_binding_sha256": "b" * 64,
                    "binding_excludes": [
                        "experiments/E055-q1-hot-cold/data/branch-preflight.json",
                        "experiments/E055-q1-hot-cold/data/manifest.json",
                    ],
                }
            }
            preflight_path = data / "branch-preflight.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            (data / "disassembly-review.json").write_text(json.dumps({
                "status": "PASS",
                "provenance": {
                    "source_sha256": "1" * 64,
                    "compiler_sha256": "2" * 64,
                    "compiler_id": "aarch64-linux-gnu-g++ test",
                },
                "builds": [{"name": "O3", "binary_sha256": "3" * 64,
                            "golden_pass": True, "golden_cases": 18}],
            }), encoding="utf-8")
            (data / "upstream-source-binding.json").write_text(json.dumps({
                "repository": "https://github.com/ggml-org/llama.cpp",
                "commit": "38c66ad0241da4f9fcce541cda8edc219086cec5",
                "immutable_ref": (
                    "https://github.com/ggml-org/llama.cpp/commit/"
                    "38c66ad0241da4f9fcce541cda8edc219086cec5"
                ),
                "sha256": (
                    "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d"
                ),
            }), encoding="utf-8")
            (data / "pmu-build-review.json").write_text(json.dumps({
                "schema": "e055-pmu-build-review/v1",
                "status": "PASS",
                "source_sha256": pmu_source_sha256,
                "binary_sha256": pmu_binary_sha256,
                "compiler_sha256": "4" * 64,
                "compiler_id": "aarch64-linux-gnu-gcc test",
                "independent_build_sha256": [pmu_binary_sha256, pmu_binary_sha256],
            }), encoding="utf-8")
            payload = generator.build_payload(
                root=root,
                experiment=experiment,
                published=(source,),
                preflight_path=preflight_path,
            )
            binding = payload["publication_binding"]
            self.assertNotIn("base_commit_at_generation", payload)
            self.assertEqual(binding["mode"],
                             "e047-staged-index-plus-post-commit-ref-verification")
            self.assertEqual(binding["preflight_base_commit"], "a" * 40)
            self.assertEqual(binding["staged_tree_binding_sha256"], "b" * 64)
            self.assertEqual(payload["upstream_source_binding"]["repack_cpp_sha256"],
                             "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d")
            runtime = payload["runtime_qualification"]
            self.assertEqual(runtime["source_sha256"], "1" * 64)
            self.assertEqual(runtime["allowed_builds"], {"O3": "3" * 64})
            self.assertEqual(runtime["upstream_commit"],
                             "38c66ad0241da4f9fcce541cda8edc219086cec5")
            self.assertEqual(runtime["pmu_group_configs"]["core"]["instructions"],
                             "0x8")
            self.assertEqual(runtime["pmu_source_sha256"], hashlib.sha256(
                (root / "tooling/a733_pmu_exec.c").read_bytes()
            ).hexdigest())
            self.assertEqual(runtime["pmu_binary_sha256"], pmu_binary_sha256)
            self.assertEqual(runtime["pmu_compiler_sha256"], "4" * 64)
            self.assertEqual(runtime["pmu_compiler_id"],
                             "aarch64-linux-gnu-gcc test")

    def test_post_commit_verifier_requires_head_local_and_upstream_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e055-binding-") as raw:
            root = pathlib.Path(raw)
            git(root, "init", "-q", "-b", "codex/e055-q1-hot-cold")
            git(root, "config", "user.name", "E055 Test")
            git(root, "config", "user.email", "e055@example.invalid")
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            git(root, "add", "base.txt")
            git(root, "commit", "-q", "-m", "base")
            base = git(root, "rev-parse", "HEAD")
            data = root / "experiments/E055-q1-hot-cold/data"
            data.mkdir(parents=True)
            (root / "payload.txt").write_text("reviewed payload\n", encoding="utf-8")
            git(root, "add", "payload.txt")
            excludes = [
                "experiments/E055-q1-hot-cold/data/branch-preflight.json",
                "experiments/E055-q1-hot-cold/data/manifest.json",
            ]
            binding = tree_binding_sha256(root, excludes, source="index")
            preflight = {
                "active_worktree": {
                    "base_commit": base,
                    "ref": "refs/heads/codex/e055-q1-hot-cold",
                    "upstream_ref": "refs/remotes/origin/codex/e055-q1-hot-cold",
                    "tree_binding_source": "git-index-stage0",
                    "tree_binding_sha256": binding,
                    "binding_excludes": excludes,
                }
            }
            manifest = {
                "schema": "e055-q1-hot-cold-manifest/v2",
                "publication_binding": {
                    "mode": "e047-staged-index-plus-post-commit-ref-verification",
                    "preflight_base_commit": base,
                    "staged_tree_binding_sha256": binding,
                    "active_ref": "refs/heads/codex/e055-q1-hot-cold",
                    "upstream_ref": "refs/remotes/origin/codex/e055-q1-hot-cold",
                    "binding_excludes": excludes,
                },
                "files": [],
            }
            preflight_path = data / "branch-preflight.json"
            manifest_path = data / "manifest.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            git(root, "add", str(preflight_path.relative_to(root)),
                str(manifest_path.relative_to(root)))
            git(root, "commit", "-q", "-m", "publication")
            head = git(root, "rev-parse", "HEAD")
            git(root, "update-ref", "refs/remotes/origin/codex/e055-q1-hot-cold", head)
            self.assertEqual(verifier.verify_publication(root, preflight_path, manifest_path), [])
            self.assertEqual(
                verifier.verify_global_publication(root, preflight_path, manifest_path), []
            )
            untracked = root / "untracked-replacement.bin"
            untracked.write_bytes(b"forged replacement")
            global_errors = verifier.verify_global_publication(
                root, preflight_path, manifest_path
            )
            self.assertTrue(any("clean" in error for error in global_errors), global_errors)
            untracked.unlink()
            manifest_path.write_text(json.dumps({**manifest, "dirty": True}), encoding="utf-8")
            errors = verifier.verify_publication(root, preflight_path, manifest_path)
            self.assertTrue(any("manifest worktree" in error for error in errors), errors)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            git(root, "update-ref", "refs/remotes/origin/codex/e055-q1-hot-cold", base)
            errors = verifier.verify_publication(root, preflight_path, manifest_path)
            self.assertTrue(any("upstream" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
