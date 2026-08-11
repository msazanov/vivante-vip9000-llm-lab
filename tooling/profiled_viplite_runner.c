#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vip_lite.h>

#define MAX_IO 8

typedef struct {
    vip_network network;
    vip_buffer inputs[MAX_IO];
    vip_buffer outputs[MAX_IO];
    vip_uint32_t input_count;
    vip_uint32_t output_count;
    vip_uint32_t input_sizes[MAX_IO];
    vip_uint32_t output_sizes[MAX_IO];
    unsigned char *host_inputs[MAX_IO];
    unsigned char *host_outputs[MAX_IO];
    unsigned char *first_outputs[MAX_IO];
    int initialized;
    int prepared;
} runner_t;

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static double elapsed_us(uint64_t start, uint64_t end)
{
    return (double)(end - start) / 1000.0;
}

static int require_status(vip_status_e status, const char *operation, vip_uint32_t index)
{
    if (status == VIP_SUCCESS) {
        return 0;
    }
    if (index == UINT32_MAX) {
        fprintf(stderr, "%s failed: status=%d\n", operation, status);
    } else {
        fprintf(stderr, "%s[%u] failed: status=%d\n", operation, index, status);
    }
    return 1;
}

static int read_exact(const char *path, unsigned char *data, size_t size)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "open input failed: %s: %s\n", path, strerror(errno));
        return 1;
    }
    size_t count = fread(data, 1, size, file);
    int extra = fgetc(file);
    int close_status = fclose(file);
    if (count != size || extra != EOF || close_status != 0) {
        fprintf(stderr, "input size mismatch: %s got=%zu expected=%zu extra=%d\n",
                path, count, size, extra != EOF);
        return 1;
    }
    return 0;
}

static int write_exact(const char *path, const unsigned char *data, size_t size)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "open output failed: %s: %s\n", path, strerror(errno));
        return 1;
    }
    size_t count = fwrite(data, 1, size, file);
    int close_status = fclose(file);
    if (count != size || close_status != 0) {
        fprintf(stderr, "output write failed: %s wrote=%zu expected=%zu\n",
                path, count, size);
        return 1;
    }
    return 0;
}

static int query_buffer_params(vip_network network, int output, vip_uint32_t index,
                               vip_buffer_create_params_t *params)
{
    memset(params, 0, sizeof(*params));
    vip_status_e (*query)(vip_network, vip_uint32_t, vip_enum, void *) =
        output ? vip_query_output : vip_query_input;
    const char *kind = output ? "output" : "input";
    vip_status_e status;

    status = query(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &params->data_format);
    if (require_status(status, "query data format", index)) return 1;
    status = query(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &params->num_of_dims);
    if (require_status(status, "query dimension count", index)) return 1;
    if (params->num_of_dims > 6) {
        fprintf(stderr, "%s[%u] has unsupported dims=%u\n", kind, index, params->num_of_dims);
        return 1;
    }
    status = query(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, params->sizes);
    if (require_status(status, "query dimensions", index)) return 1;
    status = query(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &params->quant_format);
    if (require_status(status, "query quant format", index)) return 1;

    if (params->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        vip_uint8_t fixed_point = 0;
        status = query(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &fixed_point);
        if (require_status(status, "query fixed point", index)) return 1;
        params->quant_data.dfp.fixed_point_pos = fixed_point;
    } else if (params->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        status = query(network, index, VIP_BUFFER_PROP_TF_SCALE, &params->quant_data.affine.scale);
        if (require_status(status, "query quant scale", index)) return 1;
        status = query(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                       &params->quant_data.affine.zeroPoint);
        if (require_status(status, "query zero point", index)) return 1;
    }

    params->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    printf("tensor,kind=%s,index=%u,format=%d,quant=%d,dims=%u,shape=",
           kind, index, params->data_format, params->quant_format, params->num_of_dims);
    for (vip_uint32_t dim = 0; dim < params->num_of_dims; ++dim) {
        printf("%s%u", dim == 0 ? "" : "x", params->sizes[dim]);
    }
    if (params->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        printf(",scale=%.9g,zero_point=%d", params->quant_data.affine.scale,
               params->quant_data.affine.zeroPoint);
    }
    putchar('\n');
    return 0;
}

static int initialize(runner_t *runner, const char *network_path)
{
    memset(runner, 0, sizeof(*runner));
    uint64_t start = now_ns();
    vip_status_e status = vip_init();
    uint64_t end = now_ns();
    if (require_status(status, "vip_init", UINT32_MAX)) return 1;
    runner->initialized = 1;
    printf("phase,name=init,time_us=%.3f\n", elapsed_us(start, end));

    start = now_ns();
    status = vip_create_network(network_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &runner->network);
    end = now_ns();
    if (require_status(status, "vip_create_network", UINT32_MAX)) return 1;
    printf("phase,name=create_network,time_us=%.3f\n", elapsed_us(start, end));

    if (require_status(vip_query_network(runner->network, VIP_NETWORK_PROP_INPUT_COUNT,
                                         &runner->input_count),
                       "query input count", UINT32_MAX)) return 1;
    if (require_status(vip_query_network(runner->network, VIP_NETWORK_PROP_OUTPUT_COUNT,
                                         &runner->output_count),
                       "query output count", UINT32_MAX)) return 1;
    if (runner->input_count > MAX_IO || runner->output_count > MAX_IO) {
        fprintf(stderr, "too many tensors: inputs=%u outputs=%u max=%u\n",
                runner->input_count, runner->output_count, MAX_IO);
        return 1;
    }
    printf("network,inputs=%u,outputs=%u\n", runner->input_count, runner->output_count);

    start = now_ns();
    for (vip_uint32_t index = 0; index < runner->input_count; ++index) {
        vip_buffer_create_params_t params;
        if (query_buffer_params(runner->network, 0, index, &params)) return 1;
        status = vip_create_buffer(&params, sizeof(params), &runner->inputs[index]);
        if (require_status(status, "create input buffer", index)) return 1;
        runner->input_sizes[index] = vip_get_buffer_size(runner->inputs[index]);
        runner->host_inputs[index] = malloc(runner->input_sizes[index]);
        if (runner->host_inputs[index] == NULL) return 1;
        printf("buffer,kind=input,index=%u,bytes=%u\n", index, runner->input_sizes[index]);
    }
    for (vip_uint32_t index = 0; index < runner->output_count; ++index) {
        vip_buffer_create_params_t params;
        if (query_buffer_params(runner->network, 1, index, &params)) return 1;
        status = vip_create_buffer(&params, sizeof(params), &runner->outputs[index]);
        if (require_status(status, "create output buffer", index)) return 1;
        runner->output_sizes[index] = vip_get_buffer_size(runner->outputs[index]);
        runner->host_outputs[index] = malloc(runner->output_sizes[index]);
        runner->first_outputs[index] = malloc(runner->output_sizes[index]);
        if (runner->host_outputs[index] == NULL || runner->first_outputs[index] == NULL) return 1;
        printf("buffer,kind=output,index=%u,bytes=%u\n", index, runner->output_sizes[index]);
    }
    end = now_ns();
    printf("phase,name=create_buffers,time_us=%.3f\n", elapsed_us(start, end));

    start = now_ns();
    status = vip_prepare_network(runner->network);
    end = now_ns();
    if (require_status(status, "vip_prepare_network", UINT32_MAX)) return 1;
    runner->prepared = 1;
    printf("phase,name=prepare,time_us=%.3f\n", elapsed_us(start, end));

    for (vip_uint32_t index = 0; index < runner->input_count; ++index) {
        if (require_status(vip_set_input(runner->network, index, runner->inputs[index]),
                           "vip_set_input", index)) return 1;
    }
    for (vip_uint32_t index = 0; index < runner->output_count; ++index) {
        if (require_status(vip_set_output(runner->network, index, runner->outputs[index]),
                           "vip_set_output", index)) return 1;
    }
    return 0;
}

static int execute(runner_t *runner, unsigned iteration)
{
    uint64_t h2d_start = now_ns();
    for (vip_uint32_t index = 0; index < runner->input_count; ++index) {
        void *mapped = vip_map_buffer(runner->inputs[index]);
        if (mapped == NULL) {
            fprintf(stderr, "map input[%u] failed\n", index);
            return 1;
        }
        memcpy(mapped, runner->host_inputs[index], runner->input_sizes[index]);
        if (require_status(vip_unmap_buffer(runner->inputs[index]), "unmap input", index)) return 1;
        if (require_status(vip_flush_buffer(runner->inputs[index], VIP_BUFFER_OPER_TYPE_FLUSH),
                           "flush input", index)) return 1;
    }
    uint64_t h2d_end = now_ns();

    uint64_t run_start = now_ns();
    vip_status_e run_status = vip_run_network(runner->network);
    uint64_t run_end = now_ns();
    if (require_status(run_status, "vip_run_network", UINT32_MAX)) return 1;

    uint64_t d2h_start = now_ns();
    for (vip_uint32_t index = 0; index < runner->output_count; ++index) {
        if (require_status(vip_flush_buffer(runner->outputs[index], VIP_BUFFER_OPER_TYPE_INVALIDATE),
                           "invalidate output", index)) return 1;
        void *mapped = vip_map_buffer(runner->outputs[index]);
        if (mapped == NULL) {
            fprintf(stderr, "map output[%u] failed\n", index);
            return 1;
        }
        memcpy(runner->host_outputs[index], mapped, runner->output_sizes[index]);
        if (require_status(vip_unmap_buffer(runner->outputs[index]), "unmap output", index)) return 1;
    }
    uint64_t d2h_end = now_ns();

    vip_inference_profile_t profile = {0};
    vip_status_e profile_status = vip_query_network(runner->network, VIP_NETWORK_PROP_PROFILING,
                                                     &profile);
    int repeat_equal = 1;
    for (vip_uint32_t index = 0; index < runner->output_count; ++index) {
        if (iteration == 0) {
            memcpy(runner->first_outputs[index], runner->host_outputs[index],
                   runner->output_sizes[index]);
        } else if (memcmp(runner->first_outputs[index], runner->host_outputs[index],
                          runner->output_sizes[index]) != 0) {
            repeat_equal = 0;
        }
    }

    printf("iteration,index=%u,kind=%s,h2d_us=%.3f,run_us=%.3f,d2h_us=%.3f,"
           "device_profile_status=%d,device_us=%u,cycles=%u,repeat_equal=%d\n",
           iteration, iteration == 0 ? "first" : "steady",
           elapsed_us(h2d_start, h2d_end), elapsed_us(run_start, run_end),
           elapsed_us(d2h_start, d2h_end), profile_status, profile.inference_time,
           profile.total_cycle, repeat_equal);
    fflush(stdout);
    return 0;
}

static void destroy(runner_t *runner)
{
    if (runner->network != NULL && runner->prepared) vip_finish_network(runner->network);
    for (vip_uint32_t index = 0; index < runner->input_count; ++index) {
        if (runner->inputs[index] != NULL) vip_destroy_buffer(runner->inputs[index]);
        free(runner->host_inputs[index]);
    }
    for (vip_uint32_t index = 0; index < runner->output_count; ++index) {
        if (runner->outputs[index] != NULL) vip_destroy_buffer(runner->outputs[index]);
        free(runner->host_outputs[index]);
        free(runner->first_outputs[index]);
    }
    if (runner->network != NULL) vip_destroy_network(runner->network);
    if (runner->initialized) vip_destroy();
}

int main(int argc, char **argv)
{
    if (argc < 6 || strcmp(argv[1], "--iterations") != 0) {
        fprintf(stderr, "usage: %s --iterations N network.nb input... output...\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed == 0 || parsed > 10000) {
        fprintf(stderr, "invalid iteration count: %s\n", argv[2]);
        return 2;
    }

    runner_t runner;
    if (initialize(&runner, argv[3])) {
        destroy(&runner);
        return 1;
    }
    if ((unsigned)(argc - 4) != runner.input_count + runner.output_count) {
        fprintf(stderr, "expected %u IO paths, got %d\n",
                runner.input_count + runner.output_count, argc - 4);
        destroy(&runner);
        return 2;
    }
    for (vip_uint32_t index = 0; index < runner.input_count; ++index) {
        if (read_exact(argv[4 + index], runner.host_inputs[index], runner.input_sizes[index])) {
            destroy(&runner);
            return 1;
        }
    }
    for (unsigned iteration = 0; iteration < parsed; ++iteration) {
        if (execute(&runner, iteration)) {
            destroy(&runner);
            return 1;
        }
    }
    int result = 0;
    for (vip_uint32_t index = 0; index < runner.output_count; ++index) {
        if (write_exact(argv[4 + runner.input_count + index], runner.host_outputs[index],
                        runner.output_sizes[index])) result = 1;
    }
    destroy(&runner);
    return result;
}

