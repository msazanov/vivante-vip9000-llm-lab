import json
import tempfile
import unittest
from pathlib import Path

from tooling.experiment_validator import build_file_manifest, validate_experiment
from tooling.scaffold_experiment import scaffold_experiment


def write_required_planned_files(root: Path) -> None:
    (root / "README.md").write_text("# E047\n\nСтатус: PLANNED.\n", encoding="utf-8")
    (root / "hypothesis-preflight.md").write_text(
        "# Предварительная проверка гипотезы\n\nДубликатов не найдено.\n",
        encoding="utf-8",
    )
    (root / "commands.txt").write_text("Команды ещё не запускались.\n", encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"captured": False, "status": "planned"}) + "\n", encoding="utf-8"
    )
    (root / "device.json").write_text(
        json.dumps({"captured": False, "status": "planned"}) + "\n", encoding="utf-8"
    )
    (root / "data").mkdir()
    (root / "results").mkdir()
    (root / "results/summary.json").write_text(
        json.dumps({"status": "planned"}) + "\n", encoding="utf-8"
    )
    manifest = build_file_manifest(
        root,
        [
            "README.md",
            "hypothesis-preflight.md",
            "commands.txt",
            "environment.json",
            "device.json",
            "results/summary.json",
        ],
        status="planned",
    )
    (root / "data/manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


class E053ExperimentValidatorTest(unittest.TestCase):
    def test_scaffold_creates_a_valid_planned_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = scaffold_experiment(Path(tmp), "E047-common-benchmark")
            result = validate_experiment(root)
            self.assertTrue(result["valid"], result)
            self.assertEqual("planned", result["status"])
            self.assertTrue(any("\u0400" <= char <= "\u04ff" for char in (root / "README.md").read_text(encoding="utf-8")))

    def test_planned_scaffold_requires_russian_readme_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_required_planned_files(root)
            result = validate_experiment(root)
            self.assertTrue(result["valid"], result)

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
