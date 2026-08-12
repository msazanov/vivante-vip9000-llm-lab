import unittest

from tooling.e030_q1_paired_golden import (
    make_q8_blocks,
    make_sign_rows,
    scalar_outputs,
)
from tooling.e032_q1_block4_golden import block4_outputs


class E032Q1Block4GoldenTest(unittest.TestCase):
    def test_all_adversarial_patterns_are_exact(self):
        for pattern in ("zeros", "ones", "alternating", "extremes", "random"):
            with self.subTest(pattern=pattern):
                signs = make_sign_rows(rows=4, k=512, pattern=pattern)
                q8 = make_q8_blocks(k=512, pattern=pattern)
                scales = (0.5, -1.0, 1.25, -2.0)
                expected = tuple(
                    dot * scales[row]
                    for row, dot in enumerate(scalar_outputs(signs, q8))
                )
                self.assertEqual(block4_outputs(signs, q8, scales), expected)

    def test_long_k_and_independent_scales_are_exact(self):
        signs = make_sign_rows(rows=4, k=5120, pattern="random")
        q8 = make_q8_blocks(k=5120, pattern="random")
        scales = tuple((row + 1) / 9.0 for row in range(4))
        expected = tuple(
            dot * scales[row]
            for row, dot in enumerate(scalar_outputs(signs, q8))
        )
        self.assertEqual(block4_outputs(signs, q8, scales), expected)

    def test_inputs_are_not_mutated(self):
        signs = make_sign_rows(rows=4, k=128, pattern="ones")
        q8 = make_q8_blocks(k=128, pattern="alternating")
        signs_before = tuple(tuple(row) for row in signs)
        q8_before = tuple(tuple(block) for block in q8)
        block4_outputs(signs, q8, (1.0,) * 4)
        self.assertEqual(tuple(tuple(row) for row in signs), signs_before)
        self.assertEqual(tuple(tuple(block) for block in q8), q8_before)


if __name__ == "__main__":
    unittest.main()
