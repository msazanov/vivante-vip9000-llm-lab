import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "benchmarks" / "results" / "q1-vip-cpu-golden-tests-001" / "summary.json"


class Q1GoldenSummaryTests(unittest.TestCase):
    def test_summary_matches_current_sources_and_exact_test_contract(self) -> None:
        value = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(value["schema_version"], "q1-vip-cpu-golden-tests/v1")
        self.assertEqual(value["tests_run"], 13)
        self.assertEqual(value["failures"], 0)
        self.assertEqual(value["errors"], 0)
        self.assertEqual(len(value["tests"]), 13)
        self.assertTrue(all(test["passed"] is True for test in value["tests"]))
        self.assertFalse(value["npu_executed"])
        self.assertFalse(value["performance_measured"])
        for relative, expected in value["source_sha256"].items():
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)


if __name__ == "__main__":
    unittest.main()
