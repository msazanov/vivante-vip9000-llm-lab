import unittest

from tooling.generate_e022_fused_fixture import build_fixture


class E022FusedFixtureTest(unittest.TestCase):
    def test_identity_for_all_patterns_and_sizes(self):
        for pattern in ("zeros", "ones", "alternating", "mixed", "random"):
            for m, k in ((1, 32), (4, 128), (16, 512)):
                fixture = build_fixture(m, k, pattern)
                activation = [x if x < 128 else x - 256
                              for x in fixture["activation_i8"]]
                packed = fixture["packed_q1"]
                expected = []
                for row in range(m):
                    total = 0
                    for col, value in enumerate(activation):
                        bit = (packed[row * (k // 8) + col // 8] >> (col % 8)) & 1
                        total += value if bit else -value
                    expected.append(total)
                self.assertEqual(fixture["golden_i32"], expected,
                                 (pattern, m, k))

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            build_fixture(1, 31, "mixed")


if __name__ == "__main__":
    unittest.main()
