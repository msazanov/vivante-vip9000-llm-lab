"""Behavioral tests for E055 staged-tree and post-commit publication binding."""

from __future__ import annotations

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
            git(root, "update-ref", "refs/remotes/origin/codex/e055-q1-hot-cold", base)
            errors = verifier.verify_publication(root, preflight_path, manifest_path)
            self.assertTrue(any("upstream" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
