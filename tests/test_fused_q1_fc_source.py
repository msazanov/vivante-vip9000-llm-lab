import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E009-fused-q1-native-fc"
BUILDER = EXPERIMENT / "fused_q1_fc_nbg_builder.c"


class FusedQ1FcSourceTests(unittest.TestCase):
    def test_phase_a_graph_has_virtual_weights_and_native_fc(self):
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn("vxCreateVirtualTensor2", source)
        self.assertIn("vxFullyConnectedLayer", source)
        self.assertIn("vxIdentifyGraphInputsAndOutputs(graph, 2", source)
        self.assertNotIn("expanded_weights", source)

    def test_phase_a_contract_documents_shapes_quantization_and_io(self):
        source = BUILDER.read_text(encoding="utf-8")
        readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
        compact = re.sub(r"\s+", " ", source)

        self.assertIn("repeat_cap=100", readme)
        self.assertIn("packed_dims[2] = {18, (vx_uint32)rows}", compact)
        self.assertIn("activation_dims[2] = {128, 1}", compact)
        self.assertIn("weight_dims[2] = {128, (vx_uint32)rows}", compact)
        self.assertIn("output_dims[2] = {(vx_uint32)rows, 1}", compact)

        for tensor_name in ("packed_params", "activation_params", "wp", "output_params"):
            self.assertIn(f"{tensor_name}.data_format =", source)
            self.assertIn(f"{tensor_name}.quant_format =", source)
        self.assertIn("packed_params.data_format = VX_TYPE_UINT8", source)
        self.assertIn("packed_params.quant_format = VX_QUANT_NONE", source)
        self.assertIn("activation_params.data_format = VX_TYPE_UINT8", source)
        self.assertIn("activation_params.quant_format = VX_QUANT_AFFINE_SCALE", source)
        self.assertIn("wp.data_format = VX_TYPE_UINT8", source)
        self.assertIn("wp.quant_format = VX_QUANT_AFFINE_SCALE", source)
        self.assertIn("output_params.data_format = VX_TYPE_INT16", source)
        self.assertIn("output_params.quant_format = VX_QUANT_DYNAMIC_FIXED_POINT", source)
        self.assertIn("aq.affine.scale = 1.0f", source)
        self.assertIn("aq.affine.zeroPoint = 128", source)
        self.assertIn("wq.affine.scale = 1.0f", source)
        self.assertIn("wq.affine.zeroPoint = 1", source)
        self.assertIn("oq.dfp.fixed_point_pos = 0", source)

        graph_inputs = re.search(
            r"vx_reference graph_inputs\[2\].*?vx_reference graph_outputs\[1\]",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(graph_inputs)
        io_source = graph_inputs.group(0)
        self.assertIn("packed_q1", io_source)
        self.assertIn("activation_u8", io_source)
        self.assertNotIn("scratch", io_source)
        self.assertNotIn("scratch", source.split("vx_reference graph_inputs[2]", 1)[1].split("vxGenerateNBG", 1)[0])

    def test_write_binary_always_closes_output_stream(self):
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn("size_t written = fwrite(data, 1, size, stream);", source)
        self.assertIn("int close_status = fclose(stream);", source)
        self.assertNotIn(
            "fwrite(data, 1, size, stream) != size || fclose(stream) != 0",
            source,
        )

    def test_evis_source_does_not_use_vccompiler_reserved_packed_identifier(self):
        source = (EXPERIMENT / "q1_unpack_u8_evis.vx").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\bpacked\b")

    def test_evis_unpack_is_vectorized_into_eight_non_overlapping_chunks(self):
        source = (EXPERIMENT / "q1_unpack_u8_evis.vx").read_text(encoding="utf-8")
        self.assertIn('#include "cl_viv_vx_ext.h"', source)
        self.assertIn("vxc_uchar16 sign_blob", source)
        self.assertIn("VXC_ReadImage(sign_blob, packed_q1", source)
        self.assertIn("VXC_BitExtract(sign_codes, sign_blob, sign_blob", source)
        self.assertIn("VXC_WriteImage(scratch, (int2)(chunk * 16, row),", source)
        self.assertIn("VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)", source)
        self.assertNotIn("write_imageui", source)
        self.assertNotIn("get_global_id(0) / 128u", source)

    def test_evis_bit_extract_uses_static_raw_int4_masks(self):
        source = (EXPERIMENT / "q1_unpack_u8_evis.vx").read_text(encoding="utf-8")
        self.assertIn("int4 mask_lo", source)
        self.assertIn("int4 mask_hi", source)
        self.assertIn("switch (chunk)", source)
        self.assertIn("0x03020100", source)
        self.assertIn("0x0f0e0d0c", source)
        self.assertNotRegex(source, r"vxc_uchar16\s+mask_[a-z]+\s*=.*first_bit")
        self.assertNotIn("first_bit +", source)

    def test_unpack_launches_one_work_item_per_16_output_codes(self):
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn("execution.globalWorkSize[0] = rows * 8", source)
        self.assertNotIn("execution.globalWorkSize[0] = rows * 128", source)

    def test_e010_swapped_fc_mode_keeps_q1_as_batched_input(self):
        source = BUILDER.read_text(encoding="utf-8")
        compact = re.sub(r"\s+", " ", source)

        self.assertIn('"q1-as-batched-input"', source)
        self.assertIn("vxFullyConnectedLayer(graph, scratch, activation_u8", compact)
        self.assertIn("output_dims[0] = 1", source)
        self.assertIn("output_dims[1] = (vx_uint32)rows", source)
        self.assertIn("mode=q1-as-batched-input", source)
        self.assertNotIn("vxTensorTransposeNode", source)


if __name__ == "__main__":
    unittest.main()
