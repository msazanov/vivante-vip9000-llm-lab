#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_nn.h>
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
    uint8_t *buffer = malloc((size_t)length);
    if (buffer == NULL || fread(buffer, 1, (size_t)length, stream) != (size_t)length) {
        fprintf(stderr, "failed to read %s\n", path);
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
    if (stream == NULL) {
        perror(path);
        return 1;
    }
    size_t written = fwrite(data, 1, size, stream);
    int close_status = fclose(stream);
    if (written != size || close_status != 0) {
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

static vx_status VX_CALLBACK q1_unpack_validator(
    vx_node node, const vx_reference parameters[], vx_uint32 num,
    vx_meta_format metas[])
{
    (void)node;
    (void)parameters;
    (void)num;
    (void)metas;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_unpack_initializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK q1_unpack_deinitializer(
    vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    (void)node;
    (void)parameters;
    (void)num;
    return VX_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s KERNEL.vxgcSL OUTPUT.nb ROWS [q1-as-batched-input]\n",
                argv[0]);
        return 2;
    }

    const int q1_as_batched_input = argc == 5 &&
        strcmp(argv[4], "q1-as-batched-input") == 0;
    if (argc == 5 && !q1_as_batched_input) {
        fprintf(stderr, "invalid mode: %s\n", argv[4]);
        return 2;
    }

    char *end = NULL;
    unsigned long parsed_rows = strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || parsed_rows == 0 || parsed_rows > 65535) {
        fprintf(stderr, "invalid rows: %s\n", argv[3]);
        return 2;
    }
    const vx_size rows = (vx_size)parsed_rows;
    vx_uint32 packed_dims[2] = {18, (vx_uint32)rows};
    vx_uint32 activation_dims[2] = {128, 1};
    vx_uint32 weight_dims[2] = {128, (vx_uint32)rows};
    vx_uint32 output_dims[2] = {(vx_uint32)rows, 1};
    if (q1_as_batched_input) {
        output_dims[0] = 1;
        output_dims[1] = (vx_uint32)rows;
    }

    int result = 1;
    uint8_t *program_binary = NULL;
    size_t program_size = 0;
    void *nbg = NULL;
    vx_context context = NULL;
    vx_graph graph = NULL;
    vx_program program = NULL;
    vx_kernel kernel = NULL;
    vx_node unpack = NULL;
    vx_node fc = NULL;
    vx_tensor packed_q1 = NULL;
    vx_tensor activation_u8 = NULL;
    vx_tensor scratch = NULL;
    vx_tensor dot_i16 = NULL;

    if (read_binary(argv[1], &program_binary, &program_size)) {
        goto cleanup;
    }
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
    kernel = vxAddKernelInProgram(program, (vx_char *)"q1_unpack_u8_evis", kernel_id,
                                   2, q1_unpack_validator, q1_unpack_initializer,
                                   q1_unpack_deinitializer);
    if (check_reference((vx_reference)kernel, "vxAddKernelInProgram")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(packed_q1)")) goto cleanup;
    if (check_status(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR,
                                            VX_PARAMETER_STATE_REQUIRED),
                     "vxAddParameterToKernel(scratch)")) goto cleanup;
    if (check_status(vxFinalizeKernel(kernel), "vxFinalizeKernel")) goto cleanup;

    vx_tensor_create_params_t packed_params = {0};
    packed_params.num_of_dims = 2;
    packed_params.sizes = packed_dims;
    packed_params.data_format = VX_TYPE_UINT8;
    packed_params.quant_format = VX_QUANT_NONE;
    packed_q1 = vxCreateTensor2(context, &packed_params, sizeof(packed_params));

    vx_tensor_quant_param aq = {0};
    aq.affine.scale = 1.0f;
    aq.affine.zeroPoint = 128;
    vx_tensor_create_params_t activation_params = {0};
    activation_params.num_of_dims = 2;
    activation_params.sizes = activation_dims;
    activation_params.data_format = VX_TYPE_UINT8;
    activation_params.quant_format = VX_QUANT_AFFINE_SCALE;
    activation_params.quant_data = aq;
    activation_u8 = vxCreateTensor2(context, &activation_params, sizeof(activation_params));

    vx_tensor_quant_param wq = {0};
    wq.affine.scale = 1.0f;
    wq.affine.zeroPoint = 1;
    vx_tensor_create_params_t wp = {0};
    wp.num_of_dims = 2;
    wp.sizes = weight_dims;
    wp.data_format = VX_TYPE_UINT8;
    wp.quant_format = VX_QUANT_AFFINE_SCALE;
    wp.quant_data = wq;
    scratch = vxCreateVirtualTensor2(graph, &wp, sizeof(wp));

    vx_tensor_quant_param oq = {0};
    oq.dfp.fixed_point_pos = 0;
    vx_tensor_create_params_t output_params = {0};
    output_params.num_of_dims = 2;
    output_params.sizes = output_dims;
    output_params.data_format = VX_TYPE_INT16;
    output_params.quant_format = VX_QUANT_DYNAMIC_FIXED_POINT;
    output_params.quant_data = oq;
    dot_i16 = vxCreateTensor2(context, &output_params, sizeof(output_params));
    if (check_reference((vx_reference)packed_q1, "vxCreateTensor2(packed_q1)")) goto cleanup;
    if (check_reference((vx_reference)activation_u8, "vxCreateTensor2(activation_u8)")) goto cleanup;
    if (check_reference((vx_reference)scratch, "vxCreateVirtualTensor2(scratch)")) goto cleanup;
    if (check_reference((vx_reference)dot_i16, "vxCreateTensor2(dot_i16)")) goto cleanup;

    unpack = vxCreateGenericNode(graph, kernel);
    if (check_reference((vx_reference)unpack, "vxCreateGenericNode")) goto cleanup;
    if (check_status(vxSetParameterByIndex(unpack, 0, (vx_reference)packed_q1),
                     "vxSetParameterByIndex(packed_q1)")) goto cleanup;
    if (check_status(vxSetParameterByIndex(unpack, 1, (vx_reference)scratch),
                     "vxSetParameterByIndex(scratch)")) goto cleanup;

    if (q1_as_batched_input) {
        fc = vxFullyConnectedLayer(graph, scratch, activation_u8, NULL,
                                   VX_CONVERT_POLICY_SATURATE,
                                   VX_ROUND_POLICY_TO_ZERO, dot_i16);
    } else {
        fc = vxFullyConnectedLayer(graph, activation_u8, scratch, NULL,
                                   VX_CONVERT_POLICY_SATURATE,
                                   VX_ROUND_POLICY_TO_ZERO, dot_i16);
    }
    if (check_reference((vx_reference)fc, "vxFullyConnectedLayer")) goto cleanup;

    vx_kernel_execution_parameters_t execution = {0};
    execution.workDim = 1;
    execution.globalWorkScale[0] = 1;
    execution.localWorkSize[0] = 1;
    execution.globalWorkSize[0] = rows * 8;
    if (check_status(vxSetNodeAttribute(unpack,
                                        VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &execution, sizeof(execution)),
                     "vxSetNodeAttribute(execution)")) goto cleanup;

    vx_reference graph_inputs[2] = {
        (vx_reference)packed_q1,
        (vx_reference)activation_u8,
    };
    vx_reference graph_outputs[1] = {(vx_reference)dot_i16};
    if (check_status(vxIdentifyGraphInputsAndOutputs(graph, 2, graph_inputs,
                                                     1, graph_outputs),
                     "vxIdentifyGraphInputsAndOutputs")) goto cleanup;

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
    if (write_binary(argv[2], nbg, nbg_size)) goto cleanup;
    if (q1_as_batched_input) {
        printf("mode=q1-as-batched-input\n");
    }
    printf("nbg,path=%s,bytes=%zu,program_bytes=%zu,rows=%zu\n",
           argv[2], (size_t)nbg_size, program_size, (size_t)rows);
    result = 0;

cleanup:
    free(nbg);
    if (fc != NULL) vxReleaseNode(&fc);
    if (unpack != NULL) vxReleaseNode(&unpack);
    if (dot_i16 != NULL) vxReleaseTensor(&dot_i16);
    if (scratch != NULL) vxReleaseTensor(&scratch);
    if (activation_u8 != NULL) vxReleaseTensor(&activation_u8);
    if (packed_q1 != NULL) vxReleaseTensor(&packed_q1);
    if (kernel != NULL) vxReleaseKernel(&kernel);
    if (program != NULL) vxReleaseProgram(&program);
    if (graph != NULL) vxReleaseGraph(&graph);
    if (context != NULL) vxReleaseContext(&context);
    free(program_binary);
    return result;
}
