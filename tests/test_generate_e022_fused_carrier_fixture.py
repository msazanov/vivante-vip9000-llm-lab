import unittest

from tooling.generate_e022_fused_carrier_fixture import build_carrier_fixture


class E022FusedCarrierFixtureTest(unittest.TestCase):
    def test_carrier_sizes_and_offsets(self):
        fixture = build_carrier_fixture(4, 128, "mixed")
        self.assertEqual(len(fixture["weights_q1_carrier"]), 4 * 18)
        self.assertEqual(len(fixture["activation_q8_carrier"]), 4 * 34)
        self.assertEqual(fixture["metadata"]["carrier"]["q1_sign_offset"], 0)
        self.assertEqual(fixture["metadata"]["carrier"]["q8_value_offset"], 2)

    def test_rejects_non_128_initial_gate(self):
        with self.assertRaises(ValueError):
            build_carrier_fixture(4, 256, "mixed")


if __name__ == "__main__":
    unittest.main()
