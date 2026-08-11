import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E003-q1-packed-carrier"
BUILDER = EXPERIMENT / "vxc_nbg_builder.c"
KERNEL = EXPERIMENT / "q1_vip_1x128.cl"
KERNEL_M = EXPERIMENT / "q1_vip_mx128.cl"
KERNEL_5120 = EXPERIMENT / "q1_vip_mx5120.cl"
KERNEL_Q1Q8 = EXPERIMENT / "q1_vip_q8_evis_1x128.vx"
KERNEL_Q1Q8_REUSE = EXPERIMENT / "q1_vip_q8_evis_4row_reuse.vx"
KERNEL_Q1Q8_DIRECT_SIGNS = EXPERIMENT / "q1_vip_q8_evis_4row_direct_signs.vx"
KERNEL_Q1Q8_SIGNED_READ = EXPERIMENT / "q1_vip_q8_evis_4row_signed_read.vx"
KERNEL_Q1Q8_REUSE8 = EXPERIMENT / "q1_vip_q8_evis_8row_reuse.vx"
KERNEL_Q1Q8_VECTOR_ACCUM = EXPERIMENT / "q1_vip_q8_evis_4row_vector_accum.vx"


class Q1VipNbgBuilderSourceTests(unittest.TestCase):
    def test_kernel_compiles_as_opencl_12(self):
        clang = shutil.which("clang")
        if clang is None:
            self.skipTest("clang is unavailable")
        completed = subprocess.run(
            [clang, "-x", "cl", "-cl-std=CL1.2", "-fsyntax-only", str(KERNEL)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        completed_m = subprocess.run(
            [clang, "-x", "cl", "-cl-std=CL1.2", "-fsyntax-only", str(KERNEL_M)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed_m.returncode, 0, completed_m.stderr)

        completed_5120 = subprocess.run(
            [clang, "-x", "cl", "-cl-std=CL1.2", "-fsyntax-only", str(KERNEL_5120)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed_5120.returncode, 0, completed_5120.stderr)

    def test_kernel_has_packed_three_tensor_abi(self):
        source = KERNEL.read_text(encoding="utf-8")
        match = re.search(
            r"__kernel\s+void\s+q1_vip_1x128\s*\((.*?)\)\s*\{",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "kernel symbol or signature is missing")
        arguments = [re.sub(r"\s+", " ", item.strip()) for item in match.group(1).split(",")]
        self.assertEqual(
            arguments,
            [
                "__read_only image2d_t packed_weights",
                "__read_only image2d_t activation",
                "__write_only image2d_t output",
            ],
        )
        self.assertIn("read_imageui(packed_weights", source)
        self.assertIn("read_imagef(activation", source)
        self.assertIn("write_imagef(output", source)
        self.assertNotIn("expanded_weights", source)

    def test_multirow_kernel_keeps_packed_image_abi(self):
        source = KERNEL_M.read_text(encoding="utf-8")
        self.assertIn("__kernel void q1_vip_mx128", source)
        self.assertIn("__read_only image2d_t packed_weights", source)
        self.assertIn("__read_only image2d_t activation", source)
        self.assertIn("__write_only image2d_t output", source)
        self.assertIn("const size_t row = get_global_id(0)", source)
        self.assertIn("(int2)(2u + lane / 8u, row)", source)
        self.assertNotIn("expanded_weights", source)

    def test_k5120_kernel_reads_forty_packed_blocks_per_row(self):
        source = KERNEL_5120.read_text(encoding="utf-8")
        self.assertIn("__kernel void q1_vip_mx5120", source)
        self.assertIn("block < 40u", source)
        self.assertIn("const size_t byte_base = block * 18u", source)
        self.assertIn("block * 128u + lane", source)
        self.assertNotIn("expanded_weights", source)

    def test_builder_registers_exact_graph_io_and_exports_nbg(self):
        source = BUILDER.read_text(encoding="utf-8")
        required = [
            "vxCreateProgramWithBinary",
            "vxAllocateUserKernelId",
            "vxAddKernelInProgram",
            "vxAddParameterToKernel",
            "vxFinalizeKernel",
            "vxCreateGenericNode",
            "VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS",
            "vxIdentifyGraphInputsAndOutputs",
            "vxGenerateNBG",
        ]
        for symbol in required:
            self.assertIn(symbol, source)
        self.assertIn("vx_reference graph_inputs[2]", source)
        self.assertIn("vx_reference graph_outputs[1]", source)
        self.assertIn("const vx_size packed_row_bytes = columns / 128 * 18", source)
        self.assertIn("const vx_size weight_dims[2] = {packed_row_bytes, rows}", source)
        self.assertIn("vx_size activation_dims[2] = {columns, 1}", source)
        self.assertIn("const vx_size output_dims[2] = {rows, 1}", source)
        self.assertIn("execution.globalWorkSize[0]", source)
        self.assertIn("q1_validator, q1_initializer, q1_deinitializer", source)
        self.assertNotIn("NULL, NULL, NULL);", source)

    def test_evis_kernel_keeps_q1_and_q8_canonical_carriers(self):
        source = KERNEL_Q1Q8.read_text(encoding="utf-8")
        self.assertIn("__kernel void q1_vip_q8_evis_1x128", source)
        self.assertIn("__read_only image2d_t packed_weights", source)
        self.assertIn("__read_only image2d_t packed_q8", source)
        self.assertIn("__write_only image2d_t output", source)
        self.assertIn("VXC_BitExtract", source)
        self.assertIn("VXC_DP16x1", source)
        self.assertIn("VXC_ReadImage", source)
        self.assertIn("get_global_id(0) * 4", source)
        self.assertIn("_viv_uniform int block_count", source)
        self.assertIn("block < block_count", source)
        self.assertEqual(source.count("COMPUTE_ROW(output_values."), 4)
        self.assertIn("write_imagef(output", source)
        self.assertNotIn("expanded_weights", source)

    def test_builder_supports_exact_q8_carrier_shape(self):
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn('strcmp(argv[6], "q8") == 0', source)
        self.assertIn("const vx_size q8_row_bytes = columns / 128 * 136", source)
        self.assertIn("activation_dims[0] = q8_row_bytes", source)
        self.assertIn("activation_type = VX_TYPE_UINT8", source)
        self.assertIn("vxSetNodeUniform", source)
        self.assertIn('"block_count"', source)
        self.assertIn("q8_carrier && rows % row_group != 0", source)
        self.assertIn("execution.globalWorkSize[0] = q8_carrier ? rows / row_group : rows", source)
        self.assertIn('"q1_vip_q8_evis_"', source)
        self.assertIn("q8_kernel && !q8_carrier", source)
        self.assertIn("Q8 EVIS kernel requires q8 activation carrier", source)

    def test_e011_kernel_reuses_each_q8_chunk_across_four_rows(self):
        source = KERNEL_Q1Q8_REUSE.read_text(encoding="utf-8")

        self.assertIn("__kernel void q1_vip_q8_evis_4row_reuse", source)
        self.assertEqual(source.count("VXC_ReadImage(q8_raw, packed_q8"), 1)
        self.assertIn("uniSumInt8_16x1", source)
        self.assertIn("VXC_DP16x1(total_sum", source)
        self.assertIn("2 * selected_sum - total_sum", source)
        self.assertNotIn("sign_bits = sign_bits *", source)
        self.assertEqual(source.count("ACCUMULATE_SHARED_Q8("), 2)
        self.assertIn("subblock < 4", source)
        self.assertIn("half_index < 2", source)
        self.assertIn("subblock * 34", source)
        self.assertIn("subblock * 2 + half_index", source)
        for row in range(4):
            self.assertIn(f"sign_blob{row}", source)
            self.assertIn(f"integer_dot{row}", source)
        self.assertIn("int4 mask_lo", source)
        self.assertIn("0x03020100", source)
        self.assertIn("0x7f7e7d7c", source)
        self.assertIn("get_global_id(0) * 4", source)
        self.assertIn("write_imagef(output", source)
        self.assertNotIn("COMPUTE_ROW", source)
        self.assertNotIn("expanded_weights", source)

    def test_e011_rejected_direct_signs_candidate_is_reproducible(self):
        source = KERNEL_Q1Q8_DIRECT_SIGNS.read_text(encoding="utf-8")

        self.assertIn("__kernel void q1_vip_q8_evis_4row_direct_signs", source)
        self.assertNotIn("uniSumInt8_16x1", source)
        self.assertNotIn("total_sum", source)
        self.assertIn("sign_bits = sign_bits * (vxc_uchar16)(2)", source)
        self.assertIn("sign_bits = sign_bits + (vxc_uchar16)(255)", source)
        self.assertIn("VXC_DP16x1(exact_dot", source)

        builder = BUILDER.read_text(encoding="utf-8")
        self.assertIn('"q1_vip_q8_evis_4row_direct_signs"', builder)
        self.assertIn("if (!direct_signs)", builder)

    def test_e012_signed_read_removes_only_q8_copy(self):
        source = KERNEL_Q1Q8_SIGNED_READ.read_text(encoding="utf-8")

        self.assertIn("__kernel void q1_vip_q8_evis_4row_signed_read", source)
        self.assertIn("VXC_ReadImage(signed_q8, packed_q8", source)
        self.assertNotIn("vxc_uchar16 q8_raw", source)
        self.assertNotIn("_viv_asm(COPY, signed_q8, q8_raw, 16)", source)
        self.assertIn("VXC_DP16x1(total_sum", source)
        self.assertIn("2 * selected_sum - total_sum", source)
        self.assertIn("get_global_id(0) * 4", source)

    def test_e013_kernel_reuses_q8_across_eight_rows(self):
        source = KERNEL_Q1Q8_REUSE8.read_text(encoding="utf-8")

        self.assertIn("__kernel void q1_vip_q8_evis_8row_reuse", source)
        self.assertIn("get_global_id(0) * 8", source)
        self.assertEqual(source.count("VXC_ReadImage(q8_raw, packed_q8"), 1)
        self.assertIn("VXC_DP16x1(total_sum", source)
        for row in range(8):
            self.assertIn(f"sign_blob{row}", source)
            self.assertIn(f"integer_dot{row}", source)
        self.assertEqual(source.count("write_imagef(output"), 2)
        self.assertIn("base_row + 4", source)

        builder = BUILDER.read_text(encoding="utf-8")
        self.assertIn('"q1_vip_q8_evis_8row_reuse"', builder)
        self.assertIn("rows % row_group", builder)
        self.assertIn("rows / row_group", builder)

    def test_e014_vectorizes_only_four_row_accumulation(self):
        source = KERNEL_Q1Q8_VECTOR_ACCUM.read_text(encoding="utf-8")

        self.assertIn("__kernel void q1_vip_q8_evis_4row_vector_accum", source)
        self.assertIn("float4 accumulator", source)
        self.assertIn("float4 block_accumulator", source)
        self.assertIn("const float4 q1_scales", source)
        self.assertIn("const int4 integer_dots", source)
        self.assertIn("convert_float4(integer_dots)", source)
        self.assertIn("accumulator += q1_scales * block_accumulator", source)
        self.assertIn("VXC_DP16x1(total_sum", source)
        self.assertIn("2 * selected_sum - total_sum", source)
        self.assertIn("get_global_id(0) * 4", source)
        self.assertEqual(source.count("write_imagef(output"), 1)


if __name__ == "__main__":
    unittest.main()
