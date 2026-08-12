import unittest


from tooling.e030_q1_paired_golden import (
    QK1,
    QK8,
    paired_outputs,
    scalar_outputs,
    make_q8_blocks,
    make_sign_rows,
)


class E030Q1PairedGoldenTest(unittest.TestCase):
    def _assert_case(self, *, rows: int, k: int, pattern: str) -> None:
        signs = make_sign_rows(rows=rows, k=k, pattern=pattern)
        q8 = make_q8_blocks(k=k, pattern=pattern)
        scalar = scalar_outputs(signs, q8)
        paired = paired_outputs(signs, q8)
        self.assertEqual(paired, scalar)

    def test_dimensions_match_kernel_contract(self):
        self.assertEqual(QK1, 128)
        self.assertEqual(QK8, 32)

    def test_all_adversarial_patterns_are_exact(self):
        for pattern in ("zeros", "ones", "alternating", "random", "extremes"):
            with self.subTest(pattern=pattern):
                self._assert_case(rows=8, k=128, pattern=pattern)

    def test_long_k_exact(self):
        for pattern in ("alternating", "random", "extremes"):
            with self.subTest(pattern=pattern):
                self._assert_case(rows=8, k=5120, pattern=pattern)

    def test_large_row_count_exact(self):
        self._assert_case(rows=1024, k=128, pattern="random")

    def test_pairing_does_not_mutate_inputs(self):
        signs = make_sign_rows(rows=8, k=128, pattern="random")
        q8 = make_q8_blocks(k=128, pattern="extremes")
        signs_before = tuple(tuple(row) for row in signs)
        q8_before = tuple(tuple(block) for block in q8)
        paired_outputs(signs, q8)
        self.assertEqual(tuple(tuple(row) for row in signs), signs_before)
        self.assertEqual(tuple(tuple(block) for block in q8), q8_before)


if __name__ == "__main__":
    unittest.main()
