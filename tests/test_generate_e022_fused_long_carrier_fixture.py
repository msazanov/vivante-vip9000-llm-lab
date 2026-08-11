import unittest

from tooling.generate_e022_fused_long_carrier_fixture import build_long_carrier_fixture


class LongCarrierFixtureTest(unittest.TestCase):
    def test_5120_carrier_layout_and_golden(self):
        fixture = build_long_carrier_fixture(m=4, k=5120, pattern="mixed")
        self.assertEqual(len(fixture["weights_q1_carrier"]), 4 * 40 * 18)
        self.assertEqual(len(fixture["activation_q8_carrier"]), 160 * 34)
        self.assertEqual(len(fixture["golden_i32"]), 4)
        self.assertEqual(fixture["metadata"]["carrier"]["q8_blocks"], 160)
        self.assertEqual(fixture["metadata"]["carrier"]["q1_superblocks_per_row"], 40)

    def test_requires_128_value_superblocks(self):
        with self.assertRaises(ValueError):
            build_long_carrier_fixture(m=4, k=256 + 64)


if __name__ == "__main__":
    unittest.main()
