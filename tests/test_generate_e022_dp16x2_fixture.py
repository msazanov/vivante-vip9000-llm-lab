import unittest

from tooling.generate_e022_dp16x2_fixture import build_fixture


class E022DP16x2FixtureTest(unittest.TestCase):
    def test_all_cases_are_exact_int8_dot_fixtures(self):
        for case in ("base", "selector_all_zero", "selector_all_five", "reverse_split"):
            fixture = build_fixture(case)
            self.assertEqual(len(fixture["a_hi"]), 16)
            self.assertEqual(len(fixture["a_lo"]), 16)
            self.assertEqual(len(fixture["b"]), 16)
            self.assertEqual(fixture["golden"], fixture["metadata"]["golden"])


if __name__ == "__main__":
    unittest.main()
