#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_ext_program.h>
#include <VX/vx_khr_nn.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    *data = malloc((size_t)length);
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
    if (argc != 5) {
        fprintf(stderr, "usage: %s q1_unpack.vxgcSL output.nb M K\n", argv[0]);
        return 2;
    }
    char *end_m = NULL;
    char *end_k = NULL;
    unsigned long parsed_m = strtoul(argv[3], &end_m, 10);
    unsigned long parsed_k = strtoul(argv[4], &end_k, 10);
    if (end_m == argv[3] || *end_m != '\0' || parsed_m == 0 || parsed_m > 8192 ||
        end_k == argv[4] || *end_k != '\0' || parsed_k < 32 || parsed_k > 8192 ||
        parsed_k % 32 != 0) {
        fprintf(stderr, "M must be 1..8192 and K must be 32..8192 divisible by 32\n");
        return 2;
    }
    const vx_size m = (vx_size)parsed_m;
    const vx_size k = (vx_size)parsed_k;
    uint8_t *binary = NULL;
    size_t binary_size = 0;
    if (read_binary(argv[1], &binary, &binary_size)) return 1;

    int result = 1;
    vx_context context = vxCreateContext();
    vx_graph graph = NULL;
    vx_program program = NULL;
    vx_kernel kernel = NULL;
    vx_node unpack = NULL;
    vx_node fc = NULL;
    vx_tensor packed = NULL;
    vx_tensor activation = NULL;
    vx_tensor weights = NULL;
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
    kernel = vxAddKernelInProgram(program, (vx_char *)"e020_q1_unpack_i8",
        kernel_id, 2, validator, initializer, deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram") ||
        check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                             VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(input)") ||
        check_status(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR,
                                             VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(output)") ||
        check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;

    const vx_size packed_dims[2] = {m * k / 8, 1};
    const vx_size activation_dims[2] = {k, 1};
    const vx_size weight_dims[2] = {k, m};
    const vx_size output_dims[2] = {m, 1};
    packed = vxCreateTensor(context, 2, packed_dims, VX_TYPE_UINT8, 0);
    activation = vxCreateTensor(context, 2, activation_dims, VX_TYPE_INT8, 0);
    weights = vxCreateVirtualTensor(graph, 2, weight_dims, VX_TYPE_INT8, 0);
    output = vxCreateTensor(context, 2, output_dims, VX_TYPE_INT8, 0);
    if (check_reference((vx_reference)packed, "vxCreateTensor(packed)") ||
        check_reference((vx_reference)activation, "vxCreateTensor(activation)") ||
        check_reference((vx_reference)weights, "vxCreateVirtualTensor(weights)") ||
        check_reference((vx_reference)output, "vxCreateTensor(output)")) goto cleanup;

    unpack = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference)unpack, "vxCreateGenericNode") ||
        check_status(vxSetParameterByIndex(unpack, 0, (vx_reference)packed),
                     "vxSetParameterByIndex(packed)") ||
        check_status(vxSetParameterByIndex(unpack, 1, (vx_reference)weights),
                     "vxSetParameterByIndex(weights)")) goto cleanup;
    vx_kernel_execution_parameters_t execution = {0};
    execution.workDim = 1;
    execution.globalWorkScale[0] = 1;
    execution.localWorkSize[0] = 1;
    execution.globalWorkSize[0] = m * (k / 32);
    if (check_status(vxSetNodeAttribute(unpack,
            VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
            &execution, sizeof(execution)), "vxSetNodeAttribute(execution)")) goto cleanup;

    fc = vxFullyConnectedLayer(graph, activation, weights, NULL,
        VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, output);
    if (check_reference((vx_reference)fc, "vxFullyConnectedLayer")) goto cleanup;

    vx_reference graph_inputs[2] = {
        (vx_reference)packed, (vx_reference)activation};
    vx_reference graph_outputs[1] = {(vx_reference)output};
    if (check_status(vxIdentifyGraphInputsAndOutputs(
            graph, 2, graph_inputs, 1, graph_outputs),
            "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    vx_size nbg_size = 0;
    if (check_status(vxGenerateNBG(graph, NULL, &nbg_size),
                     "vxGenerateNBG(size)") || nbg_size == 0) goto cleanup;
    nbg = malloc(nbg_size);
    if (nbg == NULL || check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                                    "vxGenerateNBG(data)")) goto cleanup;
    if (write_binary(argv[2], nbg, nbg_size)) goto cleanup;
    printf("nbg,path=%s,bytes=%zu,custom=1,native_fc=1,m=%zu,k=%zu,n=1\n",
           argv[2], (size_t)nbg_size, (size_t)m, (size_t)k);
    result = 0;

cleanup:
    free(nbg);
    if (fc != NULL) vxReleaseNode(&fc);
    if (unpack != NULL) vxReleaseNode(&unpack);
    if (output != NULL) vxReleaseTensor(&output);
    if (weights != NULL) vxReleaseTensor(&weights);
    if (activation != NULL) vxReleaseTensor(&activation);
    if (packed != NULL) vxReleaseTensor(&packed);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(binary);
    return result;
}
