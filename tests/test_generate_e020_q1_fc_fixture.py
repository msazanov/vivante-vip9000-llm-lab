import tempfile
import unittest
from pathlib import Path

from tooling.generate_e020_q1_fc_fixture import K, M, build_fixture, main


class E020Q1FCFixtureTest(unittest.TestCase):
    def test_fixture_matches_direct_integer_dot(self):
        packed, activation, golden, metadata = build_fixture()
        self.assertEqual(len(packed), M * K // 8)
        self.assertEqual(len(activation), K)
        self.assertEqual(len(golden), M)
        activations = [value if value < 128 else value - 256 for value in activation]
        expected = []
        for row in range(M):
            total = 0
            for column, value in enumerate(activations):
                byte = packed[row * (K // 8) + column // 8]
                sign = 1 if (byte >> (column % 8)) & 1 else -1
                total += sign * value
            expected.append(total)
        actual = [value if value < 128 else value - 256 for value in golden]
        self.assertEqual(actual, expected)
        self.assertEqual(metadata["golden_i8"], expected)

    def test_adversarial_patterns_have_expected_packed_bytes(self):
        zeros, _, _, _ = build_fixture(pattern="zeros")
        ones, _, _, _ = build_fixture(pattern="ones")
        alternating, _, _, _ = build_fixture(pattern="alternating")
        self.assertEqual(zeros, bytes(len(zeros)))
        self.assertEqual(ones, bytes([0xFF]) * len(ones))
        self.assertEqual(alternating[:8], bytes.fromhex("55555555aaaaaaaa"))


if __name__ == "__main__":
    unittest.main()
