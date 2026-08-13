import hashlib
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks/schema/e047-common-benchmark.schema.json"
EXAMPLE_PATH = ROOT / "benchmarks/contracts/e047-common-benchmark.example.json"


class E047ContractTest(unittest.TestCase):
    def test_example_is_valid_against_versioned_schema(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(example), key=str)
        self.assertEqual([], errors)

    def test_prompt_and_profiles_are_pinned(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        prompt = example["common_prompt"]
        self.assertEqual(
            "Explain the A733 memory bottleneck in one short sentence.",
            prompt["text"],
        )
        self.assertEqual(
            "3422031cc96896c8aff5ea0363ef6cddca9ba485437f56dbfa63d7703f01a7de",
            prompt["text_sha256"],
        )
        self.assertEqual(prompt["text_sha256"], hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest())
        self.assertFalse(prompt["text"].endswith("\n"))
        self.assertEqual({4, 32}, {item["n_predict"] for item in example["profiles"]})
        self.assertEqual({1}, {item["warmup_runs"] for item in example["profiles"]})
        self.assertEqual({5}, {item["measured_repeats"] for item in example["profiles"]})

    def test_contract_declares_metrics_quality_and_trace_overhead(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertIn("metrics", example)
        self.assertIn("token_char_byte", example["metrics"])
        self.assertIn("quality_metrics", example["metrics"])
        memory = example["metrics"]["memory_accounting"]
        self.assertEqual(
            {
                "logical_bytes",
                "unique_weight_bytes",
                "observed_direct_ddr_read_bytes",
                "observed_direct_ddr_write_bytes",
                "inferred_ddr_read_bytes",
                "inferred_ddr_write_bytes",
            },
            set(memory["byte_fields"]),
        )
        self.assertEqual(
            ["measurement_method", "measurement_source", "measurement_confidence"],
            memory["provenance_fields"],
        )
        self.assertIn("observed_direct_is_hardware_counter_only", memory["separation_rules"])
        self.assertIn("inferred_is_never_reported_as_observed", memory["separation_rules"])
        self.assertIn("trace_overhead", example)
        self.assertEqual(0.01, example["trace_overhead"]["max_allowed_fraction"])
        self.assertEqual("same_model_variant_only", example["quality_rubric"]["exact_token_match_scope"])


if __name__ == "__main__":
    unittest.main()
