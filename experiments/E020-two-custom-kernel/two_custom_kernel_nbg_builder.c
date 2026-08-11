#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_ext_program.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    int failed = fwrite(data, 1, size, stream) != size || fclose(stream) != 0;
    return failed;
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

static vx_status VX_CALLBACK validator(
    vx_node node, const vx_reference parameters[], vx_uint32 num,
    vx_meta_format metas[])
{
    (void)node;
    (void)parameters;
    (void)num;
    (void)metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK deinitializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

static vx_kernel register_binary_kernel(
    vx_context context,
    const uint8_t *binary,
    size_t binary_size,
    const char *name,
    vx_program *program_out)
{
    vx_program program = vxCreateProgramWithBinary(context, binary, binary_size);
    if (check_reference((vx_reference)program, "vxCreateProgramWithBinary")) return NULL;
    if (check_status(vxBuildProgram(program, "-cl-viv-vx-extension -D VX_VERSION=2"),
                     "vxBuildProgram")) {
        vxReleaseProgram(&program);
        return NULL;
    }
    vx_enum kernel_id = 0;
    if (check_status(vxAllocateUserKernelId(context, &kernel_id),
                     "vxAllocateUserKernelId")) {
        vxReleaseProgram(&program);
        return NULL;
    }
    vx_kernel kernel = vxAddKernelInProgram(
        program, (vx_char *)name, kernel_id, 2,
        validator, initializer, deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram")) {
        vxReleaseProgram(&program);
        return NULL;
    }
    if (check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(input)") ||
        check_status(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(output)") ||
        check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) {
        vxReleaseKernel(&kernel);
        vxReleaseProgram(&program);
        return NULL;
    }
    *program_out = program;
    return kernel;
}

static int configure_node(vx_node node, vx_size elements)
{
    vx_kernel_execution_parameters_t execution = {0};
    execution.workDim = 1;
    execution.globalWorkScale[0] = 1;
    execution.localWorkSize[0] = 1;
    execution.globalWorkSize[0] = elements / 16;
    return check_status(vxSetNodeAttribute(
        node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &execution, sizeof(execution)), "vxSetNodeAttribute(execution)");
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s stage_a.vxgcSL stage_b.vxgcSL output.nb elements\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0' || parsed == 0 || parsed > 65536 ||
        parsed % 16 != 0) {
        fprintf(stderr, "elements must be a positive multiple of 16\n");
        return 2;
    }
    const vx_size elements = (vx_size)parsed;
    uint8_t *binary_a = NULL;
    uint8_t *binary_b = NULL;
    size_t size_a = 0;
    size_t size_b = 0;
    if (read_binary(argv[1], &binary_a, &size_a) ||
        read_binary(argv[2], &binary_b, &size_b)) {
        fprintf(stderr, "failed to read shader binary\n");
        free(binary_a);
        free(binary_b);
        return 1;
    }

    int result = 1;
    vx_context context = vxCreateContext();
    vx_graph graph = NULL;
    vx_program program_a = NULL;
    vx_program program_b = NULL;
    vx_kernel kernel_a = NULL;
    vx_kernel kernel_b = NULL;
    vx_node node_a = NULL;
    vx_node node_b = NULL;
    vx_tensor input = NULL;
    vx_tensor intermediate = NULL;
    vx_tensor output = NULL;
    void *nbg = NULL;
    if (check_reference((vx_reference)context, "vxCreateContext")) goto cleanup;
    graph = vxCreateGraph(context);
    if (check_reference((vx_reference)graph, "vxCreateGraph")) goto cleanup;
    kernel_a = register_binary_kernel(context, binary_a, size_a,
                                      "e020_invert_stage_a", &program_a);
    kernel_b = register_binary_kernel(context, binary_b, size_b,
                                      "e020_invert_stage_b", &program_b);
    if (kernel_a == NULL || kernel_b == NULL) goto cleanup;

    const vx_size dims[2] = {elements, 1};
    input = vxCreateTensor(context, 2, dims, VX_TYPE_UINT8, 0);
    intermediate = vxCreateVirtualTensor(graph, 2, dims, VX_TYPE_UINT8, 0);
    output = vxCreateTensor(context, 2, dims, VX_TYPE_UINT8, 0);
    if (check_reference((vx_reference)input, "vxCreateTensor(input)") ||
        check_reference((vx_reference)intermediate, "vxCreateVirtualTensor") ||
        check_reference((vx_reference)output, "vxCreateTensor(output)")) goto cleanup;

    node_a = vxCreateGenericNode(graph, kernel_a);
    node_b = vxCreateGenericNode(graph, kernel_b);
    if (check_reference((vx_reference)node_a, "vxCreateGenericNode(A)") ||
        check_reference((vx_reference)node_b, "vxCreateGenericNode(B)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node_a, 0, (vx_reference)input),
                     "vxSetParameterByIndex(A input)") ||
        check_status(vxSetParameterByIndex(node_a, 1, (vx_reference)intermediate),
                     "vxSetParameterByIndex(A output)") ||
        check_status(vxSetParameterByIndex(node_b, 0, (vx_reference)intermediate),
                     "vxSetParameterByIndex(B input)") ||
        check_status(vxSetParameterByIndex(node_b, 1, (vx_reference)output),
                     "vxSetParameterByIndex(B output)") ||
        configure_node(node_a, elements) || configure_node(node_b, elements)) goto cleanup;

    vx_reference graph_inputs[1] = {(vx_reference)input};
    vx_reference graph_outputs[1] = {(vx_reference)output};
    if (check_status(vxIdentifyGraphInputsAndOutputs(
            graph, 1, graph_inputs, 1, graph_outputs),
            "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    vx_size nbg_size = 0;
    if (check_status(vxGenerateNBG(graph, NULL, &nbg_size),
                     "vxGenerateNBG(size)") || nbg_size == 0) goto cleanup;
    nbg = malloc(nbg_size);
    if (nbg == NULL || check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                                    "vxGenerateNBG(data)")) goto cleanup;
    if (write_binary(argv[3], nbg, nbg_size)) goto cleanup;
    printf("nbg,path=%s,bytes=%zu,kernels=2,nodes=2,elements=%zu\n",
           argv[3], (size_t)nbg_size, (size_t)elements);
    result = 0;

cleanup:
    free(nbg);
    if (node_b != NULL) vxReleaseNode(&node_b);
    if (node_a != NULL) vxReleaseNode(&node_a);
    if (output != NULL) vxReleaseTensor(&output);
    if (intermediate != NULL) vxReleaseTensor(&intermediate);
    if (input != NULL) vxReleaseTensor(&input);
    if (kernel_b != NULL) vxReleaseKernel(&kernel_b);
    if (kernel_a != NULL) vxReleaseKernel(&kernel_a);
    if (program_b != NULL) vxReleaseProgram(&program_b);
    if (program_a != NULL) vxReleaseProgram(&program_a);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(binary_b);
    free(binary_a);
    return result;
}
