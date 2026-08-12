import unittest

from tooling.e030_q1_paired_golden import (
    make_q8_blocks,
    make_sign_rows,
    scalar_outputs,
)
from tooling.e031_q1_noinline_golden import noinline_pair_outputs


class E031Q1NoinlineGoldenTest(unittest.TestCase):
    def test_pair_of_128_blocks_applies_independent_row_scales(self):
        signs = make_sign_rows(rows=8, k=256, pattern="alternating")
        q8 = make_q8_blocks(k=256, pattern="extremes")
        scales = (0.5, 1.0, -0.25, 2.0, 1.25, -0.75, 0.125, -2.0)

        expected = tuple(
            dot * scales[row]
            for row, dot in enumerate(scalar_outputs(signs, q8))
        )
        self.assertEqual(noinline_pair_outputs(signs, q8, scales), expected)

    def test_long_k_and_random_signs_are_exact(self):
        signs = make_sign_rows(rows=8, k=5120, pattern="random")
        q8 = make_q8_blocks(k=5120, pattern="random")
        scales = tuple((row + 1) / 7.0 for row in range(8))

        expected = tuple(
            dot * scales[row]
            for row, dot in enumerate(scalar_outputs(signs, q8))
        )
        self.assertEqual(noinline_pair_outputs(signs, q8, scales), expected)

    def test_inputs_are_not_mutated(self):
        signs = make_sign_rows(rows=8, k=128, pattern="ones")
        q8 = make_q8_blocks(k=128, pattern="alternating")
        signs_before = tuple(tuple(row) for row in signs)
        q8_before = tuple(tuple(block) for block in q8)
        noinline_pair_outputs(signs, q8, (1.0,) * 8)
        self.assertEqual(tuple(tuple(row) for row in signs), signs_before)
        self.assertEqual(tuple(tuple(block) for block in q8), q8_before)


if __name__ == "__main__":
    unittest.main()
