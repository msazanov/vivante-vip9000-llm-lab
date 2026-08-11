#include "q1_i16_evis_gemm_contract.h"

#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_ext_program.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_binary(const char *path, uint8_t **data, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    long length;
    uint8_t *buffer;
    if (stream == NULL) {
        perror(path);
        return 1;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(stream);
        return 1;
    }
    length = ftell(stream);
    if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        fprintf(stderr, "invalid shader binary size: %ld\n", length);
        fclose(stream);
        return 1;
    }
    buffer = (uint8_t *)malloc((size_t)length);
    if (buffer == NULL || fread(buffer, 1, (size_t)length, stream) != (size_t)length) {
        fprintf(stderr, "failed to read shader binary %s\n", path);
        free(buffer);
        fclose(stream);
        return 1;
    }
    fclose(stream);
    *data = buffer;
    *size = (size_t)length;
    return 0;
}

static int write_binary(const char *path, const void *data, size_t size)
{
    FILE *stream = fopen(path, "wb");
    int failed;
    if (stream == NULL) {
        perror(path);
        return 1;
    }
    failed = fwrite(data, 1, size, stream) != size || fclose(stream) != 0;
    if (failed) {
        fprintf(stderr, "failed to write NBG %s\n", path);
        return 1;
    }
    return 0;
}

static int check_status(vx_status status, const char *operation)
{
    if (status == VX_SUCCESS) return 0;
    fprintf(stderr, "%s failed: status=%d\n", operation, status);
    return 1;
}

static int check_reference(vx_reference reference, const char *operation)
{
    if (reference == NULL) {
        fprintf(stderr, "%s returned NULL\n", operation);
        return 1;
    }
    return check_status(vxGetStatus(reference), operation);
}

static void report_phase(const char *phase)
{
    fprintf(stderr, "phase=%s\n", phase);
    fflush(stderr);
}

static vx_status VX_CALLBACK q1_i16_validator(
    vx_node node, const vx_reference parameters[], vx_uint32 num,
    vx_meta_format metas[])
{
    (void)node;
    (void)parameters;
    (void)num;
    (void)metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_i16_initializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_i16_deinitializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

static int parse_u32(const char *text, uint32_t *value)
{
    char *end = NULL;
    unsigned long parsed;
    if (text == NULL || *text == '\0') return 1;
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed > UINT32_MAX) return 1;
    *value = (uint32_t)parsed;
    return 0;
}

int main(int argc, char **argv)
{
    vx_char kernel_name[VX_MAX_KERNEL_NAME] =
        "com.vivantecorp.extension.evis.gemm_I16I16toI16";
    const char *program_path;
    const char *nbg_path;
    uint32_t m, k, n;
    q1_i16_evis_gemm_contract contract;
    char contract_error[128] = {0};
    uint8_t *program_binary = NULL;
    size_t program_size = 0;
    void *nbg = NULL;
    vx_context context = NULL;
    vx_graph graph = NULL;
    vx_program program = NULL;
    vx_kernel kernel = NULL;
    vx_node node = NULL;
    vx_tensor input_a = NULL;
    vx_tensor input_b = NULL;
    vx_tensor output_c = NULL;
    vx_scalar scalars[7] = {NULL};
    int result = 1;
    uint32_t scalar_values[7];

    if (argc != 6 || parse_u32(argv[3], &m) || parse_u32(argv[4], &k) ||
        parse_u32(argv[5], &n) ||
        q1_i16_evis_gemm_contract_build(m, k, n, &contract,
                                        contract_error, sizeof(contract_error))) {
        fprintf(stderr, "usage: %s shader.clgcSL output.nb M K N\n", argv[0]);
        if (contract_error[0] != '\0') fprintf(stderr, "%s\n", contract_error);
        return 2;
    }
    program_path = argv[1];
    nbg_path = argv[2];

    if (read_binary(program_path, &program_binary, &program_size)) goto cleanup;
    context = vxCreateContext();
    if (check_reference((vx_reference)context, "vxCreateContext")) goto cleanup;
    report_phase("context");
    graph = vxCreateGraph(context);
    if (check_reference((vx_reference)graph, "vxCreateGraph")) goto cleanup;
    program = vxCreateProgramWithBinary(context, program_binary, program_size);
    if (check_reference((vx_reference)program, "vxCreateProgramWithBinary")) goto cleanup;
    if (check_status(vxBuildProgram(program, "-cl-viv-vx-extension -D VX_VERSION=2"),
                     "vxBuildProgram")) goto cleanup;
    report_phase("program_built");

    vx_enum kernel_id = 0;
    if (check_status(vxAllocateUserKernelId(context, &kernel_id),
                     "vxAllocateUserKernelId")) goto cleanup;
    kernel = vxAddKernelInProgram(program, kernel_name, kernel_id, 10,
                                  q1_i16_validator, q1_i16_initializer,
                                  q1_i16_deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram(full_namespace)"))
        goto cleanup;
    report_phase("kernel_full_namespace");
    if (check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(A)")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(B)")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(C)")) goto cleanup;
    for (uint32_t i = 3; i < 10; ++i) {
        if (check_status(vxAddParameterToKernel(kernel, i, VX_INPUT, VX_TYPE_SCALAR,
                                                VX_PARAMETER_STATE_REQUIRED),
                         "vxAddParameterToKernel(scalar)")) goto cleanup;
    }
    if (check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;

    {
        vx_size a_dims[2] = {k, m};
        vx_size b_dims[2] = {n, k};
        vx_size c_dims[2] = {n, m};
        input_a = vxCreateTensor(context, 2, a_dims, VX_TYPE_INT16, 8);
        input_b = vxCreateTensor(context, 2, b_dims, VX_TYPE_INT16, 8);
        output_c = vxCreateTensor(context, 2, c_dims, VX_TYPE_INT16, 8);
    }
    if (check_reference((vx_reference)input_a, "vxCreateTensor(A)")) goto cleanup;
    if (check_reference((vx_reference)input_b, "vxCreateTensor(B)")) goto cleanup;
    if (check_reference((vx_reference)output_c, "vxCreateTensor(C)")) goto cleanup;
    report_phase("tensors_dfp8");

    scalar_values[0] = 0U; /* transposeA */
    scalar_values[1] = 0U; /* transposeB */
    scalar_values[2] = 0U; /* adjointA */
    scalar_values[3] = 0U; /* adjointB */
    scalar_values[4] = m;
    scalar_values[5] = k;
    scalar_values[6] = n;
    for (uint32_t i = 0; i < 7; ++i) {
        scalars[i] = vxCreateScalar(context, VX_TYPE_INT32, &scalar_values[i]);
        if (check_reference((vx_reference)scalars[i], "vxCreateScalar(I32)")) goto cleanup;
    }
    node = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference)node, "vxCreateGenericNode")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node, 0, (vx_reference)input_a),
                     "vxSetParameterByIndex(A)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node, 1, (vx_reference)input_b),
                     "vxSetParameterByIndex(B)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node, 2, (vx_reference)output_c),
                     "vxSetParameterByIndex(C)")) goto cleanup;
    for (uint32_t i = 0; i < 7; ++i) {
        if (check_status(vxSetParameterByIndex(node, i + 3, (vx_reference)scalars[i]),
                         "vxSetParameterByIndex(scalar)")) goto cleanup;
    }

    {
        uint32_t convert_int32_to_u8[16] = {
            0x33333333U, 0x11110000U, 0x03020100U, 0x03020100U,
            0x00000000U, 0x00000000U, 0x00002400U, 0x00000000U,
            0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
            0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
        };
        uint32_t convert_a[16] = {
            0x09090909U, 0x04040404U, 0x00010000U, 0x00030002U,
            0x0a0a0a0aU, 0x00000000U, 0x00000000U, 0x00000617U,
            0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
            0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U,
        };
        uint32_t convert_b[16];
        const uint32_t multiplier = 0x80008000U;
        uint32_t input0_zp = 0U;
        uint32_t input1_zp = 0U;
        float output_zp = 0.0f;
        float output_scale = 256.0f;
        int32_t ac2zero = 0;
        int32_t bc2zero = 0;
        memcpy(convert_b, convert_a, sizeof(convert_b));
        convert_a[8] = convert_a[10] = convert_a[12] = convert_a[14] = multiplier;
        convert_b[8] = convert_b[10] = convert_b[12] = convert_b[14] = multiplier;
        if (check_status(vxSetNodeUniform(node, "input0_ZP", 1, &input0_zp),
                         "vxSetNodeUniform(input0_ZP)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "input1_ZP", 1, &input1_zp),
                         "vxSetNodeUniform(input1_ZP)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "output_ZP", 1, &output_zp),
                         "vxSetNodeUniform(output_ZP)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "outputScale", 1, &output_scale),
                         "vxSetNodeUniform(outputScale)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "uniConvertUint8SubZpToFp32_4x4",
                                          1, convert_a),
                         "vxSetNodeUniform(convertA)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "uniConvertUint8SubZpToFp32B_4x4",
                                          1, convert_b),
                         "vxSetNodeUniform(convertB)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "uniConvertInt32toUint8_2x8",
                                          1, convert_int32_to_u8),
                         "vxSetNodeUniform(convertOutput)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "ac2zero", 1, &ac2zero),
                         "vxSetNodeUniform(ac2zero)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, "bc2zero", 1, &bc2zero),
                         "vxSetNodeUniform(bc2zero)")) goto cleanup;
    }
    {
        vx_kernel_execution_parameters_t execution = {0};
        execution.workDim = 3;
        execution.globalWorkScale[0] = contract.global_scale[0];
        execution.globalWorkScale[1] = contract.global_scale[1];
        execution.globalWorkScale[2] = contract.global_scale[2];
        execution.localWorkSize[0] = 1;
        execution.localWorkSize[1] = 1;
        execution.localWorkSize[2] = 1;
        execution.globalWorkSize[0] = contract.global_size[0];
        execution.globalWorkSize[1] = contract.global_size[1];
        execution.globalWorkSize[2] = contract.global_size[2];
        if (check_status(vxSetNodeAttribute(
                node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                &execution, sizeof(execution)),
                "vxSetNodeAttribute(execution)")) goto cleanup;
    }
    {
        vx_border_t border = {0};
        vx_reference inputs[2] = {(vx_reference)input_a, (vx_reference)input_b};
        vx_reference outputs[1] = {(vx_reference)output_c};
        border.mode = VX_BORDER_CONSTANT;
        border.constant_value.S16 = 0;
        if (check_status(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)),
                         "vxSetNodeAttribute(border)")) goto cleanup;
        if (check_status(vxIdentifyGraphInputsAndOutputs(graph, 2, inputs, 1, outputs),
                         "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    }
    report_phase("graph_configured");

    {
        vx_size nbg_size = 0;
        if (check_status(vxGenerateNBG(graph, NULL, &nbg_size),
                         "vxGenerateNBG(size)")) goto cleanup;
        if (nbg_size == 0) {
            fprintf(stderr, "vxGenerateNBG returned zero bytes\n");
            goto cleanup;
        }
        nbg = malloc(nbg_size);
        if (nbg == NULL) {
            fprintf(stderr, "failed to allocate %zu NBG bytes\n", (size_t)nbg_size);
            goto cleanup;
        }
        if (check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                         "vxGenerateNBG(data)")) goto cleanup;
        if (write_binary(nbg_path, nbg, nbg_size)) goto cleanup;
        printf("nbg,path=%s,bytes=%zu,program_bytes=%zu,kernel=%s,m=%u,k=%u,n=%u\n",
               nbg_path, (size_t)nbg_size, program_size, kernel_name, m, k, n);
    }
    result = 0;

cleanup:
    free(nbg);
    if (node != NULL) vxReleaseNode(&node);
    for (uint32_t i = 0; i < 7; ++i) {
        if (scalars[i] != NULL) vxReleaseScalar(&scalars[i]);
    }
    if (input_a != NULL) vxReleaseTensor(&input_a);
    if (input_b != NULL) vxReleaseTensor(&input_b);
    if (output_c != NULL) vxReleaseTensor(&output_c);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(program_binary);
    return result;
}
