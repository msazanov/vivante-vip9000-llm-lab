from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tooling" / "profiled_viplite_runner.c"


class ProfiledVIPLiteRunnerSourceTests(unittest.TestCase):
    def test_source_exposes_resident_phase_and_quantization_contract(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        for required in (
            "CLOCK_MONOTONIC_RAW",
            "VIP_BUFFER_PROP_TF_SCALE",
            "VIP_BUFFER_PROP_TF_ZERO_POINT",
            "VIP_BUFFER_OPER_TYPE_FLUSH",
            "VIP_BUFFER_OPER_TYPE_INVALIDATE",
            "VIP_NETWORK_PROP_PROFILING",
            "kind=%s,h2d_us=%.3f,run_us=%.3f,d2h_us=%.3f",
            "repeat_equal=%d",
        ):
            self.assertIn(required, text)
        self.assertEqual(text.count("vip_prepare_network("), 1)
        self.assertEqual(text.count("vip_create_network("), 1)
        self.assertNotIn("fallback", text.lower())

    def test_input_size_is_exact_and_output_is_written_outside_iteration_loop(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("extra != EOF", text)
        loop = text.index("for (unsigned iteration = 0; iteration < parsed; ++iteration)")
        output_write = text.index("write_exact(", loop)
        execute_end = text.index("\n    }", loop)
        self.assertGreater(output_write, execute_end)


if __name__ == "__main__":
    unittest.main()
