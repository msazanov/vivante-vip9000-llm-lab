#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_ext_program.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int read_binary(const char *path, uint8_t **data, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    long length;
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) return 1;
    length = ftell(stream);
    if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        if (stream != NULL) fclose(stream);
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
    if (fwrite(data, 1, size, stream) != size) {
        fclose(stream);
        return 1;
    }
    return fclose(stream) != 0;
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

static vx_status VX_CALLBACK validator(vx_node node, const vx_reference parameters[],
                                       vx_uint32 num, vx_meta_format metas[])
{
    (void)node; (void)parameters; (void)num; (void)metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializer(vx_node node, const vx_reference parameters[],
                                          vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK deinitializer(vx_node node, const vx_reference parameters[],
                                            vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s shader.vxgcSL output.nb rows\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long rows_value = strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || rows_value == 0 || rows_value > 65535) {
        fprintf(stderr, "invalid rows: %s\n", argv[3]);
        return 2;
    }
    const vx_size rows = (vx_size)rows_value;
    uint8_t *program_binary = NULL;
    size_t program_size = 0;
    void *nbg = NULL;
    vx_context context = NULL;
    vx_graph graph = NULL;
    vx_program program = NULL;
    vx_kernel kernel = NULL;
    vx_node node = NULL;
    vx_tensor weights = NULL;
    vx_tensor activation = NULL;
    vx_tensor output = NULL;
    int result = 1;

    if (read_binary(argv[1], &program_binary, &program_size)) goto cleanup;
    context = vxCreateContext();
    if (check_reference((vx_reference)context, "vxCreateContext")) goto cleanup;
    graph = vxCreateGraph(context);
    if (check_reference((vx_reference)graph, "vxCreateGraph")) goto cleanup;
    program = vxCreateProgramWithBinary(context, program_binary, program_size);
    if (check_reference((vx_reference)program, "vxCreateProgramWithBinary")) goto cleanup;
    if (check_status(vxBuildProgram(program, "-cl-viv-vx-extension -D VX_VERSION=2"),
                     "vxBuildProgram")) goto cleanup;
    vx_enum kernel_id = 0;
    if (check_status(vxAllocateUserKernelId(context, &kernel_id),
                     "vxAllocateUserKernelId")) goto cleanup;
    kernel = vxAddKernelInProgram(program, (vx_char *)"e022_q1_fused_dp16x1",
                                   kernel_id, 3, validator, initializer, deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram") ||
        check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(weights)") ||
        check_status(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(activation)") ||
        check_status(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(output)") ||
        check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;

    const vx_size weight_dims[2] = {18, rows};
    const vx_size activation_dims[2] = {136, 1};
    const vx_size output_dims[2] = {rows * 4, 1};
    weights = vxCreateTensor(context, 2, weight_dims, VX_TYPE_UINT8, 0);
    activation = vxCreateTensor(context, 2, activation_dims, VX_TYPE_UINT8, 0);
    output = vxCreateTensor(context, 2, output_dims, VX_TYPE_INT32, 0);
    if (check_reference((vx_reference)weights, "vxCreateTensor(weights)") ||
        check_reference((vx_reference)activation, "vxCreateTensor(activation)") ||
        check_reference((vx_reference)output, "vxCreateTensor(output)")) goto cleanup;

    node = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference)node, "vxCreateGenericNode") ||
        check_status(vxSetParameterByIndex(node, 0, (vx_reference)weights),
                     "vxSetParameterByIndex(weights)") ||
        check_status(vxSetParameterByIndex(node, 1, (vx_reference)activation),
                     "vxSetParameterByIndex(activation)") ||
        check_status(vxSetParameterByIndex(node, 2, (vx_reference)output),
                     "vxSetParameterByIndex(output)")) goto cleanup;

    uint32_t uni_dot[16] = {
        0x55555555, 0x00000000, 0x76543210, 0xfedcba98,
        0x55555555, 0x76543210, 0xfedcba98, 0x00000400,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    uint32_t uni_sum[16] = {
        0x55555555, 0x00000000, 0x76543210, 0xfedcba98,
        0xaaaaaaaa, 0x00000000, 0x00000000, 0x00002400,
        0x00010001, 0x00010001, 0x00010001, 0x00010001,
        0x00010001, 0x00010001, 0x00010001, 0x00010001,
    };
    if (check_status(vxSetNodeUniform(node, (const vx_char *)"uniDotInt8_16x1", 1,
                                      uni_dot),
                     "vxSetNodeUniform(dot)") ||
        check_status(vxSetNodeUniform(node, (const vx_char *)"uniSumInt8_16x1", 1,
                                      uni_sum),
                     "vxSetNodeUniform(sum)")) goto cleanup;

    vx_kernel_execution_parameters_t execution = {0};
    execution.workDim = 1;
    execution.globalWorkScale[0] = 1;
    execution.localWorkSize[0] = 1;
    execution.globalWorkSize[0] = rows;
    if (check_status(vxSetNodeAttribute(node,
                                        VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &execution, sizeof(execution)),
                     "vxSetNodeAttribute(execution)")) goto cleanup;

    vx_reference graph_inputs[2] = {(vx_reference)weights, (vx_reference)activation};
    vx_reference graph_outputs[1] = {(vx_reference)output};
    if (check_status(vxIdentifyGraphInputsAndOutputs(graph, 2, graph_inputs, 1,
                                                     graph_outputs),
                     "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    vx_size nbg_size = 0;
    if (check_status(vxGenerateNBG(graph, NULL, &nbg_size), "vxGenerateNBG(size)"))
        goto cleanup;
    nbg = malloc(nbg_size);
    if (nbg == NULL || check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                                    "vxGenerateNBG(data)")) goto cleanup;
    if (write_binary(argv[2], nbg, nbg_size)) goto cleanup;
    printf("nbg,path=%s,bytes=%zu,custom=1,inputs=2,outputs=1,rows=%zu\n",
           argv[2], (size_t)nbg_size, (size_t)rows);
    result = 0;

cleanup:
    free(nbg);
    if (node != NULL) vxReleaseNode(&node);
    if (output != NULL) vxReleaseTensor(&output);
    if (activation != NULL) vxReleaseTensor(&activation);
    if (weights != NULL) vxReleaseTensor(&weights);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(program_binary);
    return result;
}
