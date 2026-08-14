import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / 'experiments/E048-per-op-trace/data/trace-v2-allocation-identity-requirement.json'
LEDGER_PATH = ROOT / 'experiments/E048-per-op-trace/data/compat-ab-ledger.json'


class E048TraceSchemaV2RequirementTest(unittest.TestCase):
    def test_future_q1_regions_require_allocation_identity_and_range(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        self.assertEqual(schema['schema_version'], 'e048-trace/v2-requirement')
        self.assertEqual(schema['event_types']['q1_kernel']['region_names'],
                         ['weight', 'activation', 'output'])
        required = schema['event_types']['q1_kernel']['region_descriptor']['required']
        self.assertEqual(required, [
            'allocation_id', 'base_address', 'byte_offset', 'byte_length',
        ])

    def test_current_v1_unique_q8_is_explicitly_unproven(self):
        ledger = json.loads(LEDGER_PATH.read_text())
        claim = ledger['semantics']['q8_activation_unique_read_bytes']
        self.assertIsNone(claim['value'])
        self.assertIsNone(claim['lower_bound_bytes'])
        self.assertIsNone(claim['upper_bound_bytes'])
        self.assertEqual(claim['status'], 'unknown_unproven')


if __name__ == '__main__':
    unittest.main()
