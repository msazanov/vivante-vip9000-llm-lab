import unittest

from tooling.generate_e022_fused_scaled_fixture import build_scaled_fixture


class ScaledFixtureTest(unittest.TestCase):
    def test_layout_and_float_golden(self):
        fixture = build_scaled_fixture(m=4, k=5120, pattern="mixed")
        self.assertEqual(len(fixture["weights_q1_carrier"]), 4 * 40 * 18)
        self.assertEqual(len(fixture["activation_q8_carrier"]), 160 * 34)
        self.assertEqual(len(fixture["golden_f32"]), 4)
        self.assertEqual(fixture["golden_f32"], [-33.9375, 35.5, 127.75, 65.6875])
        self.assertEqual(fixture["metadata"]["scale_semantics"],
                         "fp16_le_q1_per_128_and_q8_per_32")

    def test_requires_128_value_superblocks(self):
        with self.assertRaises(ValueError):
            build_scaled_fixture(m=4, k=256 + 64)


if __name__ == "__main__":
    unittest.main()
