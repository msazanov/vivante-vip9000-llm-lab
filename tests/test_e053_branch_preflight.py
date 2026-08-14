import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tooling.branch_preflight import build_manifest, scan_ref, tree_binding_sha256


class E053BranchPreflightTest(unittest.TestCase):
    def test_scan_ref_finds_term_and_experiment_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "LFM2.5 is covered by E047 and E053.\n", encoding="utf-8"
            )
            result = scan_ref(root, "HEAD", ["LFM2.5", "E047"])
            self.assertEqual(2, result["match_count"])
            self.assertIn("README.md", {match["path"] for match in result["matches"]})
            self.assertEqual({"E047", "E053"}, set(result["experiment_ids"]))

    def test_sha_substrings_and_partial_terms_are_not_matches_or_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "sha256=88e7f19f0b664ca610637e0532af9438432fa3b385ef615ff\n"
                "LFMapper is a different identifier.\n",
                encoding="utf-8",
            )
            manifest = build_manifest(
                root,
                ["E053", "LFM"],
                refs=["HEAD"],
                experiment_id="E053",
                hypothesis_query="строгая проверка E053",
            )
            self.assertFalse(manifest["match_found"])
            self.assertFalse(manifest["duplicate_found"])
            self.assertEqual("no_duplicate", manifest["duplicate_decision"]["status"])

    def test_build_manifest_is_machine_readable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "README.md").write_text("E047 LFM2.5\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            manifest = build_manifest(
                root,
                ["LFM2.5"],
                experiment_id="E047",
                hypothesis_query="сравнить LFM2.5 и Bonsai",
            )
            self.assertEqual("e053-branch-preflight/v3", manifest["schema_version"])
            self.assertEqual(["refs/heads/master"], [item["ref"] for item in manifest["refs"]])
            self.assertTrue(manifest["match_found"])
            self.assertFalse(manifest["duplicate_found"])
            self.assertEqual("сравнить LFM2.5 и Bonsai", manifest["hypothesis_query"])
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    def test_build_manifest_includes_remote_tracking_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "README.md").write_text("E047 only on remote branch\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "update-ref", "refs/remotes/origin/feature-e047", "HEAD"],
                check=True,
            )
            manifest = build_manifest(
                root,
                ["E047"],
                experiment_id="E047",
                hypothesis_query="проверить E047",
            )
            refs = [item["ref"] for item in manifest["refs"]]
            self.assertIn("refs/remotes/origin/feature-e047", refs)
            self.assertEqual(
                ["refs/remotes/origin/feature-e047"], manifest["remote_refs_at_scan"]
            )

    def test_duplicate_decision_contains_exact_path_ref_and_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            path = root / "experiments/E047-existing/README.md"
            path.parent.mkdir(parents=True)
            path.write_text("# Существующий опыт\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "existing"], check=True)
            manifest = build_manifest(
                root,
                ["E047"],
                experiment_id="E047",
                hypothesis_query="новая гипотеза E047",
            )
            decision = manifest["duplicate_decision"]
            self.assertFalse(manifest["match_found"])
            self.assertEqual("duplicate_found", decision["status"])
            self.assertEqual(1, decision["candidate_count"])
            self.assertEqual("experiments/E047-existing/README.md", decision["candidates"][0]["path"])
            self.assertEqual(40, len(decision["candidates"][0]["commit"]))

    def test_candidate_is_grouped_once_per_experiment_root_and_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            experiment = root / "experiments/E047-existing"
            experiment.mkdir(parents=True)
            (experiment / "README.md").write_text("опыт\n", encoding="utf-8")
            (experiment / "commands.txt").write_text("команда\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "existing"], check=True)
            manifest = build_manifest(
                root,
                ["E047"],
                experiment_id="E047",
                hypothesis_query="проверить группировку",
            )
            candidates = manifest["duplicate_decision"]["candidates"]
            self.assertEqual(1, len(candidates))
            self.assertEqual("experiments/E047-existing", candidates[0]["canonical_path"])

    def test_manifest_snapshots_every_ref_commit_and_active_tree_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "README.md").write_text("основа\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
            subprocess.run(["git", "-C", str(root), "branch", "other"], check=True)
            subprocess.run(["git", "-C", str(root), "update-ref", "refs/remotes/origin/main", "HEAD"], check=True)
            manifest = build_manifest(
                root,
                ["E047"],
                experiment_id="E047",
                hypothesis_query="снимок refs",
                binding_excludes=("evidence.json", "manifest.json"),
            )
            self.assertEqual("e053-branch-preflight/v3", manifest["schema_version"])
            self.assertEqual(
                {"refs/heads/other", "refs/remotes/origin/main"},
                {item["ref"] for item in manifest["ref_snapshot"]},
            )
            self.assertTrue(all(len(item["commit"]) == 40 for item in manifest["ref_snapshot"]))
            active = manifest["active_worktree"]
            self.assertEqual("refs/heads/master", active["ref"])
            self.assertEqual(40, len(active["base_commit"]))
            self.assertEqual(64, len(active["tree_binding_sha256"]))
            self.assertEqual("git-index-stage0", active.get("tree_binding_source"))
            self.assertEqual(["evidence.json", "manifest.json"], active["binding_excludes"])

    def test_tree_binding_uses_index_entries_and_cannot_skip_deleted_tracked_file(self):
        """Удаление tracked-файла меняет binding только после фиксации в Git index."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "tracked.txt").write_text("данные\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)

            base = tree_binding_sha256(root, ())
            (root / "tracked.txt").unlink()
            self.assertEqual(base, tree_binding_sha256(root, ()))
            subprocess.run(["git", "-C", str(root), "add", "-u"], check=True)
            self.assertNotEqual(base, tree_binding_sha256(root, ()))


if __name__ == "__main__":
    unittest.main()
