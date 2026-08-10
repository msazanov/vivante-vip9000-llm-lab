import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E008-a733-q1-4x8"
PATCH = EXPERIMENT / "a733-q1-dotprod-4x8.patch"
README = EXPERIMENT / "README.md"


class A733Q14x8PatchTests(unittest.TestCase):
    def test_patch_changes_only_dotprod_q1_selector_to_existing_4x8_trait(self):
        payload = PATCH.read_text(encoding="utf-8")
        self.assertIn("GGML_TYPE_Q1_0", payload)
        self.assertIn("ggml_cpu_has_neon() && ggml_cpu_has_dotprod()", payload)
        self.assertIn("-                return &q1_0_4x4_q8_0;", payload)
        self.assertIn("+                return &q1_0_4x8_q8_0;", payload)
        self.assertEqual(len(re.findall(r"^[-+]\s+return &q1_0_4x[48]_q8_0;", payload, re.MULTILINE)), 2)
        changed = [line for line in payload.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        self.assertEqual(changed, [
            "-                return &q1_0_4x4_q8_0;",
            "+                return &q1_0_4x8_q8_0;",
        ])

    def test_readme_gates_full_model_on_exact_operator_golden_and_speed(self):
        payload = README.read_text(encoding="utf-8")
        self.assertIn("Q1_0×Q8_0", payload)
        self.assertIn("exact", payload.lower())
        self.assertIn(">=2%", payload)
        self.assertIn("tg32", payload)
        self.assertIn("expanded", payload.lower())


if __name__ == "__main__":
    unittest.main()
