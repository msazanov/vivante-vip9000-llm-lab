import hashlib
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from tooling.experiment_validator import validate_trace_ab_result


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks/schema/e047-trace-ab-result.schema.json"
EXAMPLE_PATH = ROOT / "benchmarks/contracts/e047-trace-ab-result.example.json"


def workload_hash(workload: dict) -> str:
    encoded = json.dumps(workload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class E047TraceABTest(unittest.TestCase):
    def test_example_validates_and_pass_classification_is_computed(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual([], list(Draft202012Validator(schema).iter_errors(example)))
        self.assertEqual([], validate_trace_ab_result(example))

    def test_fake_pass_above_one_percent_is_rejected(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        for sample in example["trace_on"]["samples"]:
            sample["latency_ms"] = 1020.0
        example["statistics"].update(
            {
                "on_median_ms": 1020.0,
                "on_p95_ms": 1020.0,
                "median_overhead_fraction": 0.02,
                "p95_overhead_fraction": 1020.0 / 1002.0 - 1.0,
            }
        )
        self.assertTrue(any("TRACE_ONLY" in error for error in validate_trace_ab_result(example)))

    def test_mismatched_workload_or_token_stream_is_invalid(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        example["trace_on"]["workload_identity_sha256"] = "f" * 64
        self.assertTrue(any("INVALID" in error for error in validate_trace_ab_result(example)))
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        example["token_stream_equal"] = False
        self.assertTrue(any("INVALID" in error for error in validate_trace_ab_result(example)))

    def test_every_sample_is_paired_and_has_raw_artifact_links(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        example["trace_on"]["samples"][0]["pair_id"] = "wrong"
        self.assertTrue(any("pair_id" in error for error in validate_trace_ab_result(example)))
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        example["trace_off"]["samples"][0]["raw_artifacts"] = []
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertNotEqual([], list(Draft202012Validator(schema).iter_errors(example)))

    def test_schema_requires_exactly_five_off_and_five_on_samples(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        for side in ("trace_off", "trace_on"):
            for size in (4, 6):
                with self.subTest(side=side, size=size):
                    example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
                    samples = example[side]["samples"]
                    if size == 4:
                        example[side]["samples"] = samples[:4]
                    else:
                        extra = dict(samples[-1])
                        extra["pair_id"] = "pair-6"
                        extra["run_id"] = f"{side}-6"
                        example[side]["samples"] = samples + [extra]
                    self.assertNotEqual(
                        [], list(Draft202012Validator(schema).iter_errors(example))
                    )


if __name__ == "__main__":
    unittest.main()
