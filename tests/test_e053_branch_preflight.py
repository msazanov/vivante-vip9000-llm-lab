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

    def test_build_manifest_is_machine_readable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "README.md").write_text("E047 LFM2.5\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            manifest = build_manifest(root, ["LFM2.5"])
            self.assertEqual("e053-branch-preflight/v1", manifest["schema_version"])
            self.assertEqual(["refs/heads/master"], [item["ref"] for item in manifest["refs"]])
            self.assertTrue(manifest["duplicate_found"])
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
            manifest = build_manifest(root, ["E047"])
            refs = [item["ref"] for item in manifest["refs"]]
            self.assertIn("refs/remotes/origin/feature-e047", refs)


if __name__ == "__main__":
    unittest.main()
