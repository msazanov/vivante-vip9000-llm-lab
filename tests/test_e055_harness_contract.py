"""Static safety gates for the AArch64 E055 harness."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tooling/e055_q1_hotcold.cpp"


class E055HarnessContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_reuses_e039_reference_without_model_payload(self) -> None:
        self.assertIn("E039-q1-pair-wholek/e039_wholek_harness.cpp", self.source)
        self.assertIn("native_simd_group", self.source)
        self.assertIn("scalar_native_group", self.source)
        self.assertNotIn("Bonsai-27B", self.source)
        self.assertNotIn(".gguf", self.source)

    def test_first_red_lut_gate_is_retained_and_fix_is_explicit(self) -> None:
        failure = (ROOT / "experiments/E055-q1-hot-cold/data/failure-qemu-uninitialized-lut.txt")
        text = failure.read_text(encoding="utf-8")
        self.assertIn("status=REJECTED", text)
        self.assertIn("g_table_q1_signs", text)
        self.assertIn("scalar=0x1.48p+7", text)
        self.assertIn("make_table()", self.source)
        self.assertIn("memcpy(g_table_q1_signs", self.source)

    def test_machine_schema_is_present_and_forbids_direct_ddr_numbers(self) -> None:
        import json

        schema = json.loads(
            (ROOT / "experiments/E055-q1-hot-cold/data/sample.schema.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(schema["properties"]["schema"]["const"], "e055-q1-hot-cold/v1")
        self.assertEqual(schema["properties"]["cpu"]["enum"], [0, 6])
        self.assertEqual(schema["properties"]["observed_ddr_read_bytes"]["type"], "null")

    def test_all_three_controls_and_checksum_are_present(self) -> None:
        for mode in ("packed_stream", "unpack_scale", "full_dotprod"):
            self.assertIn(mode, self.source)
        self.assertIn("volatile uint64_t g_checksum", self.source)
        self.assertIn("noinline, noclone, used", self.source)
        self.assertIn("golden mismatch", self.source)

    def test_marker_protocol_uses_fixed_e049c_descriptors(self) -> None:
        self.assertIn("constexpr int kMarkerFd = 9", self.source)
        self.assertIn("constexpr int kAckFd = 8", self.source)
        self.assertIn("const char start = 'S'", self.source)
        self.assertIn("const char end = 'E'", self.source)
        self.assertIn("ack != 'A'", self.source)

    def test_cold_conditioning_is_verified_but_not_called_ddr_measurement(self) -> None:
        self.assertIn("verified_write_read_each_64B_line", self.source)
        self.assertIn("verified_touched", self.source)
        self.assertNotIn("observed_ddr_read_bytes", self.source)
        self.assertNotIn("bandwidth_bytes", self.source)
        self.assertIn("PMU values are event counts", self.source)

    def test_native_layout_sizes_and_exact_operation_count_are_explicit(self) -> None:
        self.assertIn("kNativeCarrierBytes = 208", self.source)
        self.assertIn("count * 512", self.source)
        self.assertIn("count * 72", self.source)
        self.assertIn("count * 4 * 34", self.source)


if __name__ == "__main__":
    unittest.main()
