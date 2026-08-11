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
    if (stream == NULL) {
        perror(path);
        return 1;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(stream);
        return 1;
    }
    long length = ftell(stream);
    if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        fprintf(stderr, "invalid input size: %ld\n", length);
        fclose(stream);
        return 1;
    }
    uint8_t *buffer = malloc((size_t) length);
    if (buffer == NULL || fread(buffer, 1, (size_t) length, stream) != (size_t) length) {
        fprintf(stderr, "failed to read %s\n", path);
        free(buffer);
        fclose(stream);
        return 1;
    }
    fclose(stream);
    *data = buffer;
    *size = (size_t) length;
    return 0;
}

static int write_binary(const char *path, const void *data, size_t size)
{
    FILE *stream = fopen(path, "wb");
    if (stream == NULL) {
        perror(path);
        return 1;
    }
    int failed = fwrite(data, 1, size, stream) != size || fclose(stream) != 0;
    if (failed) {
        fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    return 0;
}

static int check_status(vx_status status, const char *operation)
{
    if (status == VX_SUCCESS) {
        return 0;
    }
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

static vx_status VX_CALLBACK q1_validator(
    vx_node node, const vx_reference parameters[], vx_uint32 num,
    vx_meta_format metas[])
{
    (void) node;
    (void) parameters;
    (void) num;
    (void) metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_initializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void) node;
    (void) parameters;
    (void) num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_deinitializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void) node;
    (void) parameters;
    (void) num;
    return VX_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 5 && argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s kernel.clgcSL output.nb [kernel_name rows [columns [f32|q8]]]\n",
                argv[0]);
        return 2;
    }

    const char *kernel_name = "q1_vip_1x128";
    vx_size rows = 1;
    vx_size columns = 128;
    int q8_carrier = 0;
    if (argc >= 5) {
        char *end = NULL;
        unsigned long parsed_rows = strtoul(argv[4], &end, 10);
        if (end == argv[4] || *end != '\0' || parsed_rows == 0 ||
            parsed_rows > 65535 || strlen(argv[3]) >= VX_MAX_KERNEL_NAME) {
            fprintf(stderr, "invalid kernel_name/rows: %s %s\n", argv[3], argv[4]);
            return 2;
        }
        kernel_name = argv[3];
        rows = (vx_size) parsed_rows;
    }
    if (argc >= 6) {
        char *end = NULL;
        unsigned long parsed_columns = strtoul(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || parsed_columns == 0 ||
            parsed_columns > 65536 || parsed_columns % 128 != 0) {
            fprintf(stderr, "invalid columns: %s\n", argv[5]);
            return 2;
        }
        columns = (vx_size) parsed_columns;
    }
    if (argc == 7) {
        if (strcmp(argv[6], "q8") == 0) {
            q8_carrier = 1;
        } else if (strcmp(argv[6], "f32") != 0) {
            fprintf(stderr, "invalid activation carrier: %s\n", argv[6]);
            return 2;
        }
    }
    if (q8_carrier && rows % 4 != 0) {
        fprintf(stderr, "q8 EVIS output rows must be divisible by 4: %zu\n",
                (size_t) rows);
        return 2;
    }

    int result = 1;
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

    if (read_binary(argv[1], &program_binary, &program_size)) {
        goto cleanup;
    }

    context = vxCreateContext();
    if (check_reference((vx_reference) context, "vxCreateContext")) goto cleanup;
    report_phase("context");
    graph = vxCreateGraph(context);
    if (check_reference((vx_reference) graph, "vxCreateGraph")) goto cleanup;
    report_phase("graph");

    program = vxCreateProgramWithBinary(context, program_binary, program_size);
    if (check_reference((vx_reference) program, "vxCreateProgramWithBinary")) goto cleanup;
    report_phase("program");
    if (check_status(vxBuildProgram(program, "-cl-viv-vx-extension -D VX_VERSION=2"),
                     "vxBuildProgram")) goto cleanup;
    report_phase("program_built");

    vx_enum kernel_id = 0;
    if (check_status(vxAllocateUserKernelId(context, &kernel_id),
                     "vxAllocateUserKernelId")) goto cleanup;
    kernel = vxAddKernelInProgram(program, (vx_char *) kernel_name, kernel_id, 3,
                                  q1_validator, q1_initializer, q1_deinitializer);
    if (check_reference((vx_reference) kernel, "vxAddKernelInProgram")) goto cleanup;
    report_phase("kernel");
    if (check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(weights)")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(activation)")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(output)")) goto cleanup;
    if (check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;
    report_phase("kernel_finalized");

    const vx_size packed_row_bytes = columns / 128 * 18;
    const vx_size q8_row_bytes = columns / 128 * 136;
    const vx_size weight_dims[2] = {packed_row_bytes, rows};
    vx_size activation_dims[2] = {columns, 1};
    vx_enum activation_type = VX_TYPE_FLOAT32;
    int32_t block_count = (int32_t) (columns / 128);
    if (q8_carrier) {
        activation_dims[0] = q8_row_bytes;
        activation_type = VX_TYPE_UINT8;
    }
    const vx_size output_dims[2] = {rows, 1};
    weights = vxCreateTensor(context, 2, weight_dims, VX_TYPE_UINT8, 0);
    activation = vxCreateTensor(context, 2, activation_dims, activation_type, 0);
    output = vxCreateTensor(context, 2, output_dims, VX_TYPE_FLOAT32, 0);
    if (check_reference((vx_reference) weights, "vxCreateTensor(weights)")) goto cleanup;
    if (check_reference((vx_reference) activation, "vxCreateTensor(activation)")) goto cleanup;
    if (check_reference((vx_reference) output, "vxCreateTensor(output)")) goto cleanup;
    report_phase("tensors");

    node = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference) node, "vxCreateGenericNode")) goto cleanup;
    report_phase("node");
    if (check_status(vxSetParameterByIndex(node, 0, (vx_reference) weights),
                     "vxSetParameterByIndex(weights)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node, 1, (vx_reference) activation),
                     "vxSetParameterByIndex(activation)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(node, 2, (vx_reference) output),
                     "vxSetParameterByIndex(output)")) goto cleanup;

    if (q8_carrier) {
        uint32_t uni_dot_int8_16x1[16] = {
            0x55555555, 0x00000000, 0x76543210, 0xfedcba98,
            0x55555555, 0x76543210, 0xfedcba98, 0x00000400,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        };
        uint32_t uni_sum_int8_16x1[16] = {
            0x55555555, 0x00000000, 0x76543210, 0xfedcba98,
            0xaaaaaaaa, 0x00000000, 0x00000000, 0x00002400,
            0x00010001, 0x00010001, 0x00010001, 0x00010001,
            0x00010001, 0x00010001, 0x00010001, 0x00010001,
        };
        if (check_status(vxSetNodeUniform(node, (const vx_char *) "uniDotInt8_16x1",
                                          1, uni_dot_int8_16x1),
                         "vxSetNodeUniform(dot)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, (const vx_char *) "uniSumInt8_16x1",
                                          1, uni_sum_int8_16x1),
                         "vxSetNodeUniform(sum)")) goto cleanup;
        if (check_status(vxSetNodeUniform(node, (const vx_char *) "block_count",
                                          1, &block_count),
                         "vxSetNodeUniform(block_count)")) goto cleanup;
    }

    vx_kernel_execution_parameters_t execution = {0};
    execution.workDim = 1;
    execution.globalWorkScale[0] = 1;
    execution.localWorkSize[0] = 1;
    execution.globalWorkSize[0] = q8_carrier ? rows / 4 : rows;
    if (check_status(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &execution, sizeof(execution)),
                     "vxSetNodeAttribute(execution)")) goto cleanup;
    report_phase("node_configured");

    vx_reference graph_inputs[2] = {
        (vx_reference) weights,
        (vx_reference) activation,
    };
    vx_reference graph_outputs[1] = {(vx_reference) output};
    if (check_status(vxIdentifyGraphInputsAndOutputs(graph, 2, graph_inputs,
                                                     1, graph_outputs),
                     "vxIdentifyGraphInputsAndOutputs")) goto cleanup;
    report_phase("graph_io");

    vx_size nbg_size = 0;
    if (check_status(vxGenerateNBG(graph, NULL, &nbg_size),
                     "vxGenerateNBG(size)")) goto cleanup;
    report_phase("nbg_sized");
    if (nbg_size == 0) {
        fprintf(stderr, "vxGenerateNBG returned zero bytes\n");
        goto cleanup;
    }
    nbg = malloc(nbg_size);
    if (nbg == NULL) {
        fprintf(stderr, "failed to allocate %zu NBG bytes\n", (size_t) nbg_size);
        goto cleanup;
    }
    if (check_status(vxGenerateNBG(graph, nbg, &nbg_size),
                     "vxGenerateNBG(data)")) goto cleanup;
    report_phase("nbg_generated");
    if (write_binary(argv[2], nbg, nbg_size)) goto cleanup;
    printf("nbg,path=%s,bytes=%zu,program_bytes=%zu,kernel=%s,rows=%zu,columns=%zu,activation=%s\n",
           argv[2], (size_t) nbg_size, program_size, kernel_name,
           (size_t) rows, (size_t) columns, q8_carrier ? "q8" : "f32");
    result = 0;

cleanup:
    free(nbg);
    if (node != NULL) vxReleaseNode(&node);
    if (weights != NULL) vxReleaseTensor(&weights);
    if (activation != NULL) vxReleaseTensor(&activation);
    if (output != NULL) vxReleaseTensor(&output);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(program_binary);
    return result;
}
