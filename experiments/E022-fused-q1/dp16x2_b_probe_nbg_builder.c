#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_ext_program.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int read_binary(const char *path, uint8_t **data, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) return 1;
    long length = ftell(stream);
    if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return 1;
    }
    *data = (uint8_t *)malloc((size_t)length);
    if (*data == NULL || fread(*data, 1, (size_t)length, stream) != (size_t)length) {
        free(*data);
        *data = NULL;
        fclose(stream);
        return 1;
    }
    fclose(stream);
    *size = (size_t)length;
    return 0;
}

static int write_binary(const char *path, const void *data, size_t size)
{
    FILE *stream = fopen(path, "wb");
    if (stream == NULL) return 1;
    return fwrite(data, 1, size, stream) != size || fclose(stream) != 0;
}

static vx_status VX_CALLBACK validator(vx_node node,
    const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    (void)node; (void)parameters; (void)num; (void)metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializer(vx_node node,
    const vx_reference parameters[], vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK deinitializer(vx_node node,
    const vx_reference parameters[], vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s shader.vxgcSL output.nb mode\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[3], "base") != 0 && strcmp(argv[3], "v1") != 0 &&
        strcmp(argv[3], "v2") != 0 && strcmp(argv[3], "ctrl0") != 0 &&
        strcmp(argv[3], "tcfg_hi") != 0 && strcmp(argv[3], "tcfg_dp8") != 0 &&
        strcmp(argv[3], "b_bs0") != 0 && strcmp(argv[3], "b_as4444") != 0 &&
        strcmp(argv[3], "b_as5555") != 0 && strcmp(argv[3], "b_tcfg7777") != 0) {
        fprintf(stderr, "mode must be base, v1, v2, ctrl0, tcfg_hi, tcfg_dp8, b_bs0, b_as4444, b_as5555, or b_tcfg7777\n");
        return 2;
    }
    uint8_t *binary = NULL;
    size_t binary_size = 0;
    if (read_binary(argv[1], &binary, &binary_size)) return 1;

    int result = 1;
    vx_context context = vxCreateContext();
    vx_graph graph = NULL;
    vx_program program = NULL;
    vx_kernel kernel = NULL;
    vx_node node = NULL;
    vx_tensor a_hi = NULL;
    vx_tensor a_lo = NULL;
    vx_tensor b = NULL;
    vx_tensor output = NULL;
    void *nbg = NULL;

    if (check_reference((vx_reference)context, "vxCreateContext")) goto cleanup;
    graph = vxCreateGraph(context);
    if (check_reference((vx_reference)graph, "vxCreateGraph")) goto cleanup;
    program = vxCreateProgramWithBinary(context, binary, binary_size);
    if (check_reference((vx_reference)program, "vxCreateProgramWithBinary") ||
        check_status(vxBuildProgram(program, "-cl-viv-vx-extension -D VX_VERSION=2"),
                     "vxBuildProgram")) goto cleanup;
    vx_enum kernel_id = 0;
    if (check_status(vxAllocateUserKernelId(context, &kernel_id),
                     "vxAllocateUserKernelId")) goto cleanup;
    kernel = vxAddKernelInProgram(program, (vx_char *)"e022_dp16x2_b_probe",
        kernel_id, 4, validator, initializer, deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram") ||
        check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(a_hi)") ||
        check_status(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(a_lo)") ||
        check_status(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(b)") ||
        check_status(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(output)") ||
        check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;

    const vx_size input_dims[2] = {16, 1};
    const vx_size output_dims[2] = {8, 1};
    a_hi = vxCreateTensor(context, 2, input_dims, VX_TYPE_INT8, 0);
    a_lo = vxCreateTensor(context, 2, input_dims, VX_TYPE_INT8, 0);
    b = vxCreateTensor(context, 2, input_dims, VX_TYPE_INT8, 0);
    output = vxCreateTensor(context, 2, output_dims, VX_TYPE_INT32, 0);
    if (check_reference((vx_reference)a_hi, "vxCreateTensor(a_hi)") ||
        check_reference((vx_reference)a_lo, "vxCreateTensor(a_lo)") ||
        check_reference((vx_reference)b, "vxCreateTensor(b)") ||
        check_reference((vx_reference)output, "vxCreateTensor(output)")) goto cleanup;

    node = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference)node, "vxCreateGenericNode") ||
        check_status(vxSetParameterByIndex(node, 0, (vx_reference)a_hi),
                     "vxSetParameterByIndex(a_hi)") ||
        check_status(vxSetParameterByIndex(node, 1, (vx_reference)a_lo),
                     "vxSetParameterByIndex(a_lo)") ||
        check_status(vxSetParameterByIndex(node, 2, (vx_reference)b),
                     "vxSetParameterByIndex(b)") ||
        check_status(vxSetParameterByIndex(node, 3, (vx_reference)output),
                     "vxSetParameterByIndex(output)")) goto cleanup;

    uint32_t uni_dot[16] = {
        0x55555555, 0x00000000,
        0x76543210, 0xfedcba98,
        0x55555555,
        0x76543210, 0xfedcba98,
        0x00000400,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    if (strcmp(argv[3], "v1") == 0) {
        uni_dot[0] = 0x00000000;
        uni_dot[4] = 0x55555555;
    } else if (strcmp(argv[3], "v2") == 0) {
        uni_dot[1] = 0x55550000;
        uni_dot[4] = 0x00000000;
    } else if (strcmp(argv[3], "ctrl0") == 0) {
        uni_dot[1] = 0x00000000;
        uni_dot[4] = 0x00000000;
        uni_dot[7] = 0x00000000;
    } else if (strcmp(argv[3], "tcfg_hi") == 0) {
        uni_dot[0] = 0x55555555;
    } else if (strcmp(argv[3], "tcfg_dp8") == 0) {
        uni_dot[0] = 0x55555555;
        uni_dot[1] = 0x00000000;
        uni_dot[2] = 0x76543210;
        uni_dot[3] = 0x98765432;
        uni_dot[4] = 0x00000000;
        uni_dot[5] = 0x76543210;
        uni_dot[6] = 0x76543210;
        uni_dot[7] = 0x00000000;
    } else if (strcmp(argv[3], "b_bs0") == 0) {
        /* Keep the signed-I8 bins, but let the _b instruction select B
         * through its zero selector. */
        uni_dot[4] = 0x00000000;
    } else if (strcmp(argv[3], "b_as4444") == 0) {
        /* Official DP2x8_b-style A selectors and a contiguous signed-I8 B. */
        uni_dot[1] = 0x44444444;
        uni_dot[2] = 0x33221100;
        uni_dot[3] = 0x77665544;
        uni_dot[4] = 0x00000000;
        uni_dot[5] = 0x76543210;
        uni_dot[6] = 0xfedcba98;
    } else if (strcmp(argv[3], "b_as5555") == 0) {
        uni_dot[1] = 0x55555555;
        uni_dot[4] = 0x00000000;
    } else if (strcmp(argv[3], "b_tcfg7777") == 0) {
        uni_dot[0] = 0x77777777;
        uni_dot[1] = 0x44444444;
        uni_dot[2] = 0x33221100;
        uni_dot[3] = 0x77665544;
        uni_dot[4] = 0x00000000;
        uni_dot[5] = 0x76543210;
        uni_dot[6] = 0xfedcba98;
        uni_dot[7] = 0x00004000;
    }
    if (check_status(vxSetNodeUniform(node, (const vx_char *)"uniDotInt8_16x1",
                                      1, uni_dot),
                    "vxSetNodeUniform(uniDotInt8_16x1)")) goto cleanup;
    vx_reference graph_inputs[3] = {
        (vx_reference)a_hi, (vx_reference)a_lo, (vx_reference)b};
    vx_reference graph_outputs[1] = {(vx_reference)output};
    if (check_status(vxIdentifyGraphInputsAndOutputs(
            graph, 3, graph_inputs, 1, graph_outputs),
            "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    vx_size nbg_size = 0;
    if (check_status(vxGenerateNBG(graph, NULL, &nbg_size),
                     "vxGenerateNBG(size)")) goto cleanup;
    nbg = malloc(nbg_size);
    if (nbg == NULL || check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                                    "vxGenerateNBG(data)")) goto cleanup;
    if (write_binary(argv[2], nbg, nbg_size)) goto cleanup;
    printf("nbg,path=%s,bytes=%zu,custom=1,inputs=3,outputs=1\n",
           argv[2], (size_t)nbg_size);
    result = 0;

cleanup:
    free(nbg);
    if (node != NULL) vxReleaseNode(&node);
    if (output != NULL) vxReleaseTensor(&output);
    if (b != NULL) vxReleaseTensor(&b);
    if (a_lo != NULL) vxReleaseTensor(&a_lo);
    if (a_hi != NULL) vxReleaseTensor(&a_hi);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(binary);
    return result;
}
