from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "E049d-steady-pmu-v2" / "scripts" / "analyze.py"
RAW = ROOT / "experiments" / "E049d-steady-pmu-v2" / "raw"
STOCK = ROOT / "experiments" / "E049d-steady-pmu" / "raw" / "stock-reference-n4" / "token-ids.jsonl"


def load_module():
    spec = importlib.util.spec_from_file_location("e049d_pmu_analyze", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E049dPmuAnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_published_raw_is_valid_and_normalized_by_three(self) -> None:
        summary, rows = self.module.build_summary(RAW, STOCK)
        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(summary["sample_count"], 15)
        self.assertTrue(summary["all_samples_valid"])
        self.assertFalse(summary["optimization_claim"])
        self.assertEqual(summary["measured_tokens_per_sample"], 3)
        self.assertEqual(len(rows), 40)
        core = summary["groups"]["core"]
        self.assertEqual(core["events"]["instructions"]["median_raw_count"], 53307699279)
        self.assertEqual(
            core["events"]["instructions"]["median_event_count_per_token"],
            53307699279 / 3,
        )

    def test_token_mismatch_rejects_sample(self) -> None:
        stock = STOCK.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            run = pathlib.Path(temporary)
            source = RAW / "series" / "core-r1"
            for name in ("exit.txt", "pmu.json"):
                (run / name).write_bytes((source / name).read_bytes())
            records = [json.loads(line) for line in stock.splitlines()]
            records[-1]["token_id"] += 1
            (run / "token-ids.jsonl").write_text(
                "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "token capture differs"):
                self.module.validate_sample(run, "core", stock)

    def test_manifest_covers_every_artifact_except_itself(self) -> None:
        manifest = ROOT / "experiments" / "E049d-steady-pmu-v2" / "data" / "manifest.tsv"
        listed = {line.split("\t", 1)[0] for line in manifest.read_text(encoding="utf-8").splitlines()[1:]}
        expected = {
            path.relative_to(RAW.parent).as_posix()
            for path in RAW.parent.rglob("*")
            if path.is_file() and path != manifest
        }
        self.assertEqual(listed, expected)


if __name__ == "__main__":
    unittest.main()
