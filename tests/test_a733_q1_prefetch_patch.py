import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E009-a733-q1-prefetch"
PATCH = EXPERIMENT / "a733-q1-prefetch4.patch"
README = EXPERIMENT / "README.md"


class A733Q1PrefetchPatchTests(unittest.TestCase):
    def test_patch_adds_one_read_prefetch_to_q1_4x4_inner_block_loop(self):
        payload = PATCH.read_text(encoding="utf-8")
        self.assertIn("ggml_gemv_q1_0_4x4_q8_0", payload)
        self.assertIn("for (int l = 0; l < nb; l++)", payload)
        self.assertIn("if (l + 4 < nb)", payload)
        self.assertIn("__builtin_prefetch(b_ptr + l + 4, 0, 3);", payload)
        changed = [line for line in payload.splitlines() if line.startswith("+") and not line.startswith("+++")]
        self.assertEqual(changed, [
            "+            if (l + 4 < nb) {",
            "+                __builtin_prefetch(b_ptr + l + 4, 0, 3);",
            "+            }",
        ])
        self.assertEqual(len(re.findall(r"__builtin_prefetch", payload)), 1)

    def test_readme_keeps_exact_and_full_model_gates(self):
        payload = README.read_text(encoding="utf-8")
        self.assertIn("288", payload)
        self.assertIn("exact", payload.lower())
        self.assertIn(">=2%", payload)
        self.assertIn("tg32", payload)
        self.assertIn("expanded", payload.lower())


if __name__ == "__main__":
    unittest.main()
