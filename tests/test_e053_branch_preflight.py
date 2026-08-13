import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tooling.branch_preflight import build_manifest, scan_ref


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
            self.assertEqual("e053-branch-preflight/v2", manifest["schema_version"])
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
            self.assertEqual("experiments/E047-existing/README.md", decision["candidates"][0]["path"])
            self.assertEqual(40, len(decision["candidates"][0]["commit"]))


if __name__ == "__main__":
    unittest.main()
