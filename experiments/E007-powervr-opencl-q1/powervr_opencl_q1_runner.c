#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define POWERVR_OPENCL_LIBRARY "/usr/lib/libPVROCL.so.1"

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint32_t cl_bool;
typedef intptr_t cl_context_properties;
typedef cl_ulong cl_device_type;
typedef cl_ulong cl_mem_flags;
typedef cl_ulong cl_command_queue_properties;
typedef uint32_t cl_platform_info;
typedef uint32_t cl_device_info;
typedef uint32_t cl_program_build_info;
typedef uint32_t cl_profiling_info;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

enum {
    CL_SUCCESS = 0,
    CL_TRUE = 1,
    CL_DEVICE_TYPE_GPU = 1 << 2,
    CL_QUEUE_PROFILING_ENABLE = 1 << 1,
    CL_MEM_WRITE_ONLY = 1 << 1,
    CL_MEM_READ_ONLY = 1 << 2,
    CL_PLATFORM_NAME = 0x0902,
    CL_DEVICE_NAME = 0x102b,
    CL_DRIVER_VERSION = 0x102d,
    CL_PROGRAM_BUILD_LOG = 0x1183,
    CL_PROFILING_COMMAND_START = 0x1282,
    CL_PROFILING_COMMAND_END = 0x1283,
};

struct opencl_api {
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    cl_context (*clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                  void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id,
                                              cl_command_queue_properties, cl_int *);
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char **,
                                             const size_t *, cl_int *);
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                             void (*)(cl_program, void *), void *);
    cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info,
                                    size_t, void *, size_t *);
    cl_kernel (*clCreateKernel)(cl_program, const char *, cl_int *);
    cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                   const void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
                                     const size_t *, const size_t *, const size_t *,
                                     cl_uint, const cl_event *, cl_event *);
    cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                  void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*clFinish)(cl_command_queue);
    cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void *, size_t *);
    cl_int (*clReleaseEvent)(cl_event);
    cl_int (*clReleaseMemObject)(cl_mem);
    cl_int (*clReleaseKernel)(cl_kernel);
    cl_int (*clReleaseProgram)(cl_program);
    cl_int (*clReleaseCommandQueue)(cl_command_queue);
    cl_int (*clReleaseContext)(cl_context);
};

struct sample_timing {
    double kernel_event_ms;
    double host_submit_wait_ms;
};

struct options {
    bool cpu_reference_only;
    const char *weights_path;
    const char *activation_path;
    const char *kernel_path;
    const char *output_jsonl_path;
    const char *output_f32_path;
    const char *run_id;
    uint64_t rows;
    uint64_t columns;
    uint64_t warmup;
    uint64_t iterations;
    uint64_t local_size;
};

static void print_contract(void) {
    puts("{\"schema\":\"powervr-opencl-q1-contract/v1\","
         "\"runtime\":\"/usr/lib/libPVROCL.so.1\","
         "\"generic_icd_allowed\":false,"
         "\"kernel_language\":\"OpenCL C 1.2\","
         "\"weight_layout\":\"Q1_0:fp16-scale+16-sign-bytes-per-128\","
         "\"activation_dtype\":\"F32\","
         "\"output_dtype\":\"F32\","
         "\"explicit_host_expanded_weight_allocation_bytes\":0,"
         "\"runtime_internal_expansion_observed\":\"unknown\","
         "\"allowed_local_sizes\":[32,64,128],"
         "\"qualified_steady_minimum_warmup\":1,"
         "\"qualified_steady_minimum_iterations\":50,"
         "\"timing_fields\":[\"weights_h2d_event_ms\",\"weights_h2d_host_ms\","
         "\"activation_h2d_event_ms\",\"activation_h2d_host_ms\","
         "\"d2h_event_ms\",\"d2h_host_ms\",\"host_total_ms\"],"
         "\"host_total_scope\":\"weights_h2d_start_through_final_d2h_finish\","
         "\"thermal_wrapper\":\"external\"}");
}

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "usage: %s [--cpu-reference-only] --weights FILE --activation FILE "
            "--m ROWS --k COLUMNS --local-size 32|64|128 "
            "--output-jsonl FILE --output-f32 FILE [--kernel FILE] "
            "[--warmup N] [--iterations N] [--run-id ID]\n",
            program);
}

static bool parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t) parsed;
    return true;
}

static bool valid_run_id(const char *text) {
    if (text == NULL || *text == '\0') {
        return false;
    }
    for (const unsigned char *cursor = (const unsigned char *) text; *cursor != '\0'; ++cursor) {
        const bool safe = (*cursor >= 'a' && *cursor <= 'z') ||
                          (*cursor >= 'A' && *cursor <= 'Z') ||
                          (*cursor >= '0' && *cursor <= '9') ||
                          *cursor == '.' || *cursor == '_' || *cursor == '-';
        if (!safe) {
            return false;
        }
    }
    return true;
}

static bool take_value(int argc, char **argv, int *index, const char **value) {
    if (*index + 1 >= argc) {
        return false;
    }
    *index += 1;
    *value = argv[*index];
    return true;
}

static int parse_options(int argc, char **argv, struct options *options) {
    *options = (struct options) {
        .run_id = "powervr-opencl-q1",
        .warmup = 1,
        .iterations = 50,
        .local_size = 64,
    };
    for (int index = 1; index < argc; ++index) {
        const char *arg = argv[index];
        const char *value = NULL;
        if (strcmp(arg, "--cpu-reference-only") == 0) {
            options->cpu_reference_only = true;
        } else if (strcmp(arg, "--weights") == 0) {
            if (!take_value(argc, argv, &index, &options->weights_path)) return 2;
        } else if (strcmp(arg, "--activation") == 0) {
            if (!take_value(argc, argv, &index, &options->activation_path)) return 2;
        } else if (strcmp(arg, "--kernel") == 0) {
            if (!take_value(argc, argv, &index, &options->kernel_path)) return 2;
        } else if (strcmp(arg, "--output-jsonl") == 0) {
            if (!take_value(argc, argv, &index, &options->output_jsonl_path)) return 2;
        } else if (strcmp(arg, "--output-f32") == 0) {
            if (!take_value(argc, argv, &index, &options->output_f32_path)) return 2;
        } else if (strcmp(arg, "--run-id") == 0) {
            if (!take_value(argc, argv, &index, &options->run_id)) return 2;
        } else if (strcmp(arg, "--m") == 0 || strcmp(arg, "--k") == 0 ||
                   strcmp(arg, "--warmup") == 0 || strcmp(arg, "--iterations") == 0 ||
                   strcmp(arg, "--local-size") == 0) {
            if (!take_value(argc, argv, &index, &value)) return 2;
            uint64_t parsed = 0;
            if (!parse_u64(value, &parsed)) {
                fprintf(stderr, "invalid integer for %s: %s\n", arg, value);
                return 2;
            }
            if (strcmp(arg, "--m") == 0) options->rows = parsed;
            if (strcmp(arg, "--k") == 0) options->columns = parsed;
            if (strcmp(arg, "--warmup") == 0) options->warmup = parsed;
            if (strcmp(arg, "--iterations") == 0) options->iterations = parsed;
            if (strcmp(arg, "--local-size") == 0) options->local_size = parsed;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg);
            return 2;
        }
    }

    if (options->weights_path == NULL || options->activation_path == NULL ||
        options->output_jsonl_path == NULL || options->output_f32_path == NULL) {
        fputs("weights, activation, output-jsonl, and output-f32 are required\n", stderr);
        return 2;
    }
    if (!options->cpu_reference_only && options->kernel_path == NULL) {
        fputs("kernel is required for a GPU run\n", stderr);
        return 2;
    }
    if (options->rows == 0 || options->columns == 0 ||
        options->rows % 128 != 0 || options->columns % 128 != 0) {
        fputs("m and k must be positive multiples of 128\n", stderr);
        return 2;
    }
    if (options->local_size != 32 && options->local_size != 64 && options->local_size != 128) {
        fputs("local size must be 32, 64, or 128\n", stderr);
        return 2;
    }
    if (options->warmup < 1) {
        fputs("warmup must be at least 1 for a qualified steady run\n", stderr);
        return 2;
    }
    if (options->iterations < 50) {
        fputs("iterations must be at least 50 for a qualified steady run\n", stderr);
        return 2;
    }
    if (options->warmup > 1000000 || options->iterations > 1000000) {
        fputs("warmup and iterations must not exceed 1000000\n", stderr);
        return 2;
    }
    if (!valid_run_id(options->run_id)) {
        fputs("run-id must contain only ASCII letters, digits, dot, underscore, or dash\n", stderr);
        return 2;
    }
    return 0;
}

static bool checked_mul_size(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static bool derive_sizes(const struct options *options, size_t *weights_bytes,
                         size_t *activation_bytes, size_t *output_bytes) {
    if (options->rows > SIZE_MAX || options->columns > SIZE_MAX) {
        return false;
    }
    const size_t rows = (size_t) options->rows;
    const size_t columns = (size_t) options->columns;
    size_t blocks = 0;
    if (!checked_mul_size(rows, columns / 128, &blocks) ||
        !checked_mul_size(blocks, 18, weights_bytes) ||
        !checked_mul_size(columns, sizeof(float), activation_bytes) ||
        !checked_mul_size(rows, sizeof(float), output_bytes)) {
        return false;
    }
    return true;
}

static bool read_exact_file(const char *path, size_t expected_size, void *buffer) {
    struct stat metadata;
    if (stat(path, &metadata) != 0) {
        fprintf(stderr, "stat failed for %s: %s\n", path, strerror(errno));
        return false;
    }
    if (metadata.st_size < 0 || (uintmax_t) metadata.st_size != (uintmax_t) expected_size) {
        fprintf(stderr, "unexpected size for %s: expected %zu, got %jd\n",
                path, expected_size, (intmax_t) metadata.st_size);
        return false;
    }
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "open failed for %s: %s\n", path, strerror(errno));
        return false;
    }
    const size_t count = fread(buffer, 1, expected_size, stream);
    bool ok = count == expected_size && ferror(stream) == 0;
    if (fclose(stream) != 0) {
        ok = false;
    }
    if (!ok) {
        fprintf(stderr, "read failed for %s\n", path);
    }
    return ok;
}

static float fp16_to_float(uint16_t value) {
    const uint32_t sign = ((uint32_t) value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t) value >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t) value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x03ffu;
            exponent = 113u - shift;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static bool scalar_golden(const uint8_t *weights, const float *activation,
                          size_t rows, size_t columns, float *output) {
    for (size_t column = 0; column < columns; ++column) {
        if (!isfinite(activation[column])) {
            fprintf(stderr, "activation contains a non-finite value at %zu\n", column);
            return false;
        }
    }
    const size_t blocks_per_row = columns / 128;
    for (size_t row = 0; row < rows; ++row) {
        float accumulator = 0.0f;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t base = (row * blocks_per_row + block) * 18;
            const uint16_t scale_bits = (uint16_t) weights[base] |
                                        (uint16_t) ((uint16_t) weights[base + 1] << 8);
            const float scale = fp16_to_float(scale_bits);
            if (!isfinite(scale)) {
                fprintf(stderr, "weight scale is non-finite at row %zu block %zu\n", row, block);
                return false;
            }
            float block_dot = 0.0f;
            for (size_t lane = 0; lane < 128; ++lane) {
                const uint8_t signs = weights[base + 2 + lane / 8];
                const float value = activation[block * 128 + lane];
                block_dot += ((signs >> (lane & 7)) & 1u) != 0u ? value : -value;
            }
            accumulator += scale * block_dot;
            if (!isfinite(accumulator)) {
                fprintf(stderr,
                        "non-finite scalar accumulator/output at row %zu block %zu\n",
                        row, block);
                return false;
            }
        }
        if (!isfinite(accumulator)) {
            fprintf(stderr, "non-finite scalar accumulator/output at row %zu\n", row);
            return false;
        }
        output[row] = accumulator;
    }
    return true;
}

static bool load_opencl_symbol(void *library, const char *name, void *destination,
                               size_t destination_size) {
    dlerror();
    void *symbol = dlsym(library, name);
    const char *error = dlerror();
    if (error != NULL || symbol == NULL) {
        fprintf(stderr, "dlsym(%s) failed: %s\n", name, error != NULL ? error : "null symbol");
        return false;
    }
    if (destination_size != sizeof(symbol)) {
        fprintf(stderr, "function pointer size mismatch for %s\n", name);
        return false;
    }
    memcpy(destination, &symbol, sizeof(symbol));
    return true;
}

#define LOAD_CL_SYMBOL(library, api, name) \
    load_opencl_symbol((library), #name, &(api)->name, sizeof((api)->name))

static bool load_opencl_api(void *library, struct opencl_api *api) {
    memset(api, 0, sizeof(*api));
    return LOAD_CL_SYMBOL(library, api, clGetPlatformIDs) &&
           LOAD_CL_SYMBOL(library, api, clGetPlatformInfo) &&
           LOAD_CL_SYMBOL(library, api, clGetDeviceIDs) &&
           LOAD_CL_SYMBOL(library, api, clGetDeviceInfo) &&
           LOAD_CL_SYMBOL(library, api, clCreateContext) &&
           LOAD_CL_SYMBOL(library, api, clCreateCommandQueue) &&
           LOAD_CL_SYMBOL(library, api, clCreateProgramWithSource) &&
           LOAD_CL_SYMBOL(library, api, clBuildProgram) &&
           LOAD_CL_SYMBOL(library, api, clGetProgramBuildInfo) &&
           LOAD_CL_SYMBOL(library, api, clCreateKernel) &&
           LOAD_CL_SYMBOL(library, api, clCreateBuffer) &&
           LOAD_CL_SYMBOL(library, api, clSetKernelArg) &&
           LOAD_CL_SYMBOL(library, api, clEnqueueWriteBuffer) &&
           LOAD_CL_SYMBOL(library, api, clEnqueueNDRangeKernel) &&
           LOAD_CL_SYMBOL(library, api, clEnqueueReadBuffer) &&
           LOAD_CL_SYMBOL(library, api, clFinish) &&
           LOAD_CL_SYMBOL(library, api, clGetEventProfilingInfo) &&
           LOAD_CL_SYMBOL(library, api, clReleaseEvent) &&
           LOAD_CL_SYMBOL(library, api, clReleaseMemObject) &&
           LOAD_CL_SYMBOL(library, api, clReleaseKernel) &&
           LOAD_CL_SYMBOL(library, api, clReleaseProgram) &&
           LOAD_CL_SYMBOL(library, api, clReleaseCommandQueue) &&
           LOAD_CL_SYMBOL(library, api, clReleaseContext);
}

static char *read_kernel_source(const char *path, size_t *length) {
    struct stat metadata;
    if (stat(path, &metadata) != 0) {
        fprintf(stderr, "stat failed for kernel %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (metadata.st_size <= 0 || metadata.st_size > 1024 * 1024) {
        fprintf(stderr, "kernel source size must be between 1 and 1048576 bytes: %s\n", path);
        return NULL;
    }
    *length = (size_t) metadata.st_size;
    char *source = malloc(*length + 1);
    if (source == NULL) {
        fputs("kernel source allocation failed\n", stderr);
        return NULL;
    }
    if (!read_exact_file(path, *length, source)) {
        free(source);
        return NULL;
    }
    source[*length] = '\0';
    return source;
}

static uint64_t monotonic_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    return (uint64_t) now.tv_sec * UINT64_C(1000000000) + (uint64_t) now.tv_nsec;
}

#define CL_CALL_OR_RETURN(status_lvalue, expression) do { \
    (status_lvalue) = (expression); \
    if ((status_lvalue) != CL_SUCCESS) { \
        fprintf(stderr, "%s failed with OpenCL status %d\n", #expression, (status_lvalue)); \
        return false; \
    } \
} while (0)

static bool query_platform_string(const struct opencl_api *api, cl_platform_id platform,
                                  cl_platform_info property, char **result) {
    size_t size = 0;
    cl_int status = CL_SUCCESS;
    CL_CALL_OR_RETURN(status, api->clGetPlatformInfo(platform, property, 0, NULL, &size));
    if (size == 0 || size > 65536) {
        fprintf(stderr, "clGetPlatformInfo returned invalid string size %zu\n", size);
        return false;
    }
    char *value = calloc(size, 1);
    if (value == NULL) {
        return false;
    }
    status = api->clGetPlatformInfo(platform, property, size, value, NULL);
    if (status != CL_SUCCESS || value[size - 1] != '\0') {
        fprintf(stderr, "clGetPlatformInfo(value) failed with status %d\n", status);
        free(value);
        return false;
    }
    *result = value;
    return true;
}

static bool query_device_string(const struct opencl_api *api, cl_device_id device,
                                cl_device_info property, char **result) {
    size_t size = 0;
    cl_int status = CL_SUCCESS;
    CL_CALL_OR_RETURN(status, api->clGetDeviceInfo(device, property, 0, NULL, &size));
    if (size == 0 || size > 65536) {
        fprintf(stderr, "clGetDeviceInfo returned invalid string size %zu\n", size);
        return false;
    }
    char *value = calloc(size, 1);
    if (value == NULL) {
        return false;
    }
    status = api->clGetDeviceInfo(device, property, size, value, NULL);
    if (status != CL_SUCCESS || value[size - 1] != '\0') {
        fprintf(stderr, "clGetDeviceInfo(value) failed with status %d\n", status);
        free(value);
        return false;
    }
    *result = value;
    return true;
}

static bool event_duration_ms(const struct opencl_api *api, cl_event event, double *duration_ms) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    cl_int status = CL_SUCCESS;
    CL_CALL_OR_RETURN(status, api->clGetEventProfilingInfo(
        event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
    CL_CALL_OR_RETURN(status, api->clGetEventProfilingInfo(
        event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL));
    if (end < start) {
        fputs("OpenCL event profiling end precedes start\n", stderr);
        return false;
    }
    *duration_ms = (double) (end - start) / 1000000.0;
    return true;
}

static bool compare_golden(const float *golden, const float *actual, size_t count,
                           double *max_abs_error, double *mean_abs_error) {
    const double absolute_tolerance = 1e-3;
    const double relative_tolerance = 1e-5;
    double maximum = 0.0;
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(actual[index])) {
            fprintf(stderr, "GPU output is non-finite at %zu\n", index);
            return false;
        }
        const double difference = fabs((double) actual[index] - (double) golden[index]);
        const double limit = absolute_tolerance + relative_tolerance * fabs((double) golden[index]);
        if (difference > maximum) {
            maximum = difference;
        }
        sum += difference;
        if (difference > limit) {
            fprintf(stderr,
                    "golden mismatch at %zu: gpu=%.9g cpu=%.9g abs=%.9g limit=%.9g\n",
                    index, (double) actual[index], (double) golden[index], difference, limit);
            return false;
        }
    }
    *max_abs_error = maximum;
    *mean_abs_error = count == 0 ? 0.0 : sum / (double) count;
    return true;
}

static bool json_string(FILE *stream, const char *value) {
    if (fputc('"', stream) == EOF) return false;
    for (const unsigned char *cursor = (const unsigned char *) value; *cursor != '\0'; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') {
            if (fputc('\\', stream) == EOF || fputc(*cursor, stream) == EOF) return false;
        } else if (*cursor >= 0x20 && *cursor < 0x7f) {
            if (fputc(*cursor, stream) == EOF) return false;
        } else {
            if (fprintf(stream, "\\u%04x", (unsigned int) *cursor) < 0) return false;
        }
    }
    return fputc('"', stream) != EOF;
}

static char *temporary_path(const char *final_path) {
    const size_t length = strlen(final_path);
    if (length > SIZE_MAX - 48) {
        return NULL;
    }
    char *path = malloc(length + 48);
    if (path == NULL) {
        return NULL;
    }
    const int written = snprintf(path, length + 48, "%s.tmp.%ld", final_path, (long) getpid());
    if (written < 0 || (size_t) written >= length + 48) {
        free(path);
        return NULL;
    }
    return path;
}

static bool path_absent(const char *path) {
    if (access(path, F_OK) == 0) {
        fprintf(stderr, "refusing to overwrite existing output: %s\n", path);
        return false;
    }
    if (errno != ENOENT) {
        fprintf(stderr, "cannot inspect output path %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static bool publish_pair_no_replace(const char *output_temp, const char *output_final,
                                    const char *json_temp, const char *json_final) {
    bool output_published = false;
    bool json_published = false;
    if (link(output_temp, output_final) != 0) {
        fprintf(stderr, "refusing to replace competing output %s: %s\n",
                output_final, strerror(errno));
        return false;
    }
    output_published = true;
    if (link(json_temp, json_final) != 0) {
        fprintf(stderr, "refusing to replace competing output %s: %s\n",
                json_final, strerror(errno));
        goto rollback;
    }
    json_published = true;
    if (unlink(output_temp) != 0 || unlink(json_temp) != 0) {
        fputs("failed to remove temporary names after publishing output pair\n", stderr);
        goto rollback;
    }
    return true;

rollback:
    if (json_published && unlink(json_final) != 0) {
        fprintf(stderr, "failed to roll back %s: %s\n", json_final, strerror(errno));
    }
    if (output_published && unlink(output_final) != 0) {
        fprintf(stderr, "failed to roll back %s: %s\n", output_final, strerror(errno));
    }
    return false;
}

static bool write_cpu_outputs(const struct options *options, const float *output,
                              size_t weights_bytes, size_t activation_bytes,
                              size_t output_bytes) {
    if (!path_absent(options->output_jsonl_path) || !path_absent(options->output_f32_path)) {
        return false;
    }
    char *json_temp = temporary_path(options->output_jsonl_path);
    char *output_temp = temporary_path(options->output_f32_path);
    if (json_temp == NULL || output_temp == NULL) {
        fputs("failed to allocate temporary output paths\n", stderr);
        free(json_temp);
        free(output_temp);
        return false;
    }

    bool ok = false;
    FILE *binary = fopen(output_temp, "wbx");
    if (binary == NULL) {
        fprintf(stderr, "cannot create %s: %s\n", output_temp, strerror(errno));
        goto cleanup;
    }
    bool binary_ok = fwrite(output, 1, output_bytes, binary) == output_bytes;
    if (fclose(binary) != 0) {
        binary_ok = false;
    }
    binary = NULL;
    if (!binary_ok) {
        fputs("failed to write CPU golden output\n", stderr);
        goto cleanup;
    }

    FILE *jsonl = fopen(json_temp, "wx");
    if (jsonl == NULL) {
        fprintf(stderr, "cannot create %s: %s\n", json_temp, strerror(errno));
        goto cleanup;
    }
    const int identity = fprintf(
        jsonl,
        "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"identity\","
        "\"run_id\":\"%s\",\"backend\":\"scalar-c11-reference\","
        "\"shape\":{\"m\":%" PRIu64 ",\"k\":%" PRIu64 "},"
        "\"local_size\":%" PRIu64 ",\"warmup\":%" PRIu64 ","
        "\"iterations\":%" PRIu64 "}\n",
        options->run_id, options->rows, options->columns, options->local_size,
        options->warmup, options->iterations);
    const int memory = fprintf(
        jsonl,
        "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"memory\","
        "\"packed_weight_bytes\":%zu,"
        "\"explicit_host_expanded_weight_allocation_bytes\":0,"
        "\"runtime_internal_expansion_observed\":\"unknown\","
        "\"activation_f32_bytes\":%zu,\"output_f32_bytes\":%zu,"
        "\"resident_weight_upload_count\":0}\n",
        weights_bytes, activation_bytes, output_bytes);
    const int result = fprintf(
        jsonl,
        "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"result\","
        "\"golden\":\"scalar-c11-q1_0-fp32\",\"golden_pass\":true,"
        "\"cpu_reference_only\":true}\n");
    bool json_ok = identity >= 0 && memory >= 0 && result >= 0;
    if (fclose(jsonl) != 0) {
        json_ok = false;
    }
    jsonl = NULL;
    if (!json_ok) {
        fputs("failed to write JSONL output\n", stderr);
        goto cleanup;
    }

    if (!publish_pair_no_replace(output_temp, options->output_f32_path,
                                 json_temp, options->output_jsonl_path)) {
        goto cleanup;
    }
    ok = true;

cleanup:
    unlink(json_temp);
    unlink(output_temp);
    free(json_temp);
    free(output_temp);
    return ok;
}

static bool write_gpu_outputs(const struct options *options, const float *output,
                              size_t weights_bytes, size_t activation_bytes,
                              size_t output_bytes, const char *platform_name,
                              const char *device_name, const char *driver_version,
                              double build_ms, double weights_h2d_event_ms,
                              double weights_h2d_host_ms,
                              double activation_h2d_event_ms,
                              double activation_h2d_host_ms,
                              double d2h_event_ms, double d2h_host_ms,
                              double host_total_ms, double max_abs_error,
                              double mean_abs_error, const struct sample_timing *samples) {
    if (!path_absent(options->output_jsonl_path) || !path_absent(options->output_f32_path)) {
        return false;
    }
    char *json_temp = temporary_path(options->output_jsonl_path);
    char *output_temp = temporary_path(options->output_f32_path);
    if (json_temp == NULL || output_temp == NULL) {
        fputs("failed to allocate temporary output paths\n", stderr);
        free(json_temp);
        free(output_temp);
        return false;
    }
    bool ok = false;
    FILE *binary = fopen(output_temp, "wbx");
    if (binary == NULL) {
        fprintf(stderr, "cannot create %s: %s\n", output_temp, strerror(errno));
        goto cleanup;
    }
    bool binary_ok = fwrite(output, 1, output_bytes, binary) == output_bytes;
    if (fclose(binary) != 0) {
        binary_ok = false;
    }
    binary = NULL;
    if (!binary_ok) {
        fputs("failed to write GPU output\n", stderr);
        goto cleanup;
    }

    FILE *jsonl = fopen(json_temp, "wx");
    if (jsonl == NULL) {
        fprintf(stderr, "cannot create %s: %s\n", json_temp, strerror(errno));
        goto cleanup;
    }
    if (fputs("{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"identity\","
              "\"run_id\":", jsonl) == EOF || !json_string(jsonl, options->run_id) ||
        fputs(",\"backend\":\"powervr-opencl-direct\",\"runtime\":\""
              POWERVR_OPENCL_LIBRARY "\",\"platform\":", jsonl) == EOF ||
        !json_string(jsonl, platform_name) || fputs(",\"device\":", jsonl) == EOF ||
        !json_string(jsonl, device_name) || fputs(",\"driver\":", jsonl) == EOF ||
        !json_string(jsonl, driver_version) ||
        fprintf(jsonl,
                ",\"kernel_language\":\"OpenCL C 1.2\","
                "\"shape\":{\"m\":%" PRIu64 ",\"k\":%" PRIu64 "},"
                "\"local_size\":%" PRIu64 ",\"warmup\":%" PRIu64 ","
                "\"iterations\":%" PRIu64 "}\n",
                options->rows, options->columns, options->local_size,
                options->warmup, options->iterations) < 0) {
        fputs("failed to write identity JSONL record\n", stderr);
        fclose(jsonl);
        goto cleanup;
    }
    if (fprintf(jsonl,
                "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"memory\","
                "\"packed_weight_bytes\":%zu,"
                "\"explicit_host_expanded_weight_allocation_bytes\":0,"
                "\"runtime_internal_expansion_observed\":\"unknown\","
                "\"activation_f32_bytes\":%zu,\"output_f32_bytes\":%zu,"
                "\"resident_weight_upload_count\":1,\"cpu_fallback_count\":0}\n",
                weights_bytes, activation_bytes, output_bytes) < 0 ||
        fprintf(jsonl,
                "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"transfer\","
                "\"build_ms\":%.9f,\"weights_h2d_event_ms\":%.9f,"
                "\"weights_h2d_host_ms\":%.9f,"
                "\"activation_h2d_event_ms\":%.9f,"
                "\"activation_h2d_host_ms\":%.9f,"
                "\"h2d_event_ms\":%.9f,\"h2d_host_ms\":%.9f}\n",
                build_ms, weights_h2d_event_ms, weights_h2d_host_ms,
                activation_h2d_event_ms, activation_h2d_host_ms,
                weights_h2d_event_ms + activation_h2d_event_ms,
                weights_h2d_host_ms + activation_h2d_host_ms) < 0) {
        fputs("failed to write memory/transfer JSONL records\n", stderr);
        fclose(jsonl);
        goto cleanup;
    }
    for (uint64_t index = 0; index < options->iterations; ++index) {
        if (fprintf(jsonl,
                    "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"sample\","
                    "\"index\":%" PRIu64 ",\"kernel_event_ms\":%.9f,"
                    "\"host_submit_wait_ms\":%.9f}\n",
                    index, samples[index].kernel_event_ms,
                    samples[index].host_submit_wait_ms) < 0) {
            fputs("failed to write sample JSONL record\n", stderr);
            fclose(jsonl);
            goto cleanup;
        }
    }
    const int result_record = fprintf(jsonl,
                "{\"schema\":\"powervr-opencl-q1-run/v1\",\"record\":\"result\","
                "\"golden\":\"scalar-c11-q1_0-fp32\",\"golden_pass\":true,"
                "\"atol\":0.001,\"rtol\":0.00001,\"max_abs_error\":%.9g,"
                "\"mean_abs_error\":%.9g,\"d2h_event_ms\":%.9f,"
                "\"d2h_host_ms\":%.9f,\"host_total_ms\":%.9f,"
                "\"cpu_reference_only\":false}\n",
                max_abs_error, mean_abs_error, d2h_event_ms, d2h_host_ms,
                host_total_ms);
    bool json_ok = result_record >= 0;
    if (fclose(jsonl) != 0) {
        json_ok = false;
    }
    jsonl = NULL;
    if (!json_ok) {
        fputs("failed to finish JSONL output\n", stderr);
        goto cleanup;
    }

    if (!publish_pair_no_replace(output_temp, options->output_f32_path,
                                 json_temp, options->output_jsonl_path)) {
        goto cleanup;
    }
    ok = true;

cleanup:
    unlink(json_temp);
    unlink(output_temp);
    free(json_temp);
    free(output_temp);
    return ok;
}

static int run_cpu_reference(const struct options *options) {
    size_t weights_bytes = 0;
    size_t activation_bytes = 0;
    size_t output_bytes = 0;
    if (!derive_sizes(options, &weights_bytes, &activation_bytes, &output_bytes)) {
        fputs("shape byte count overflows size_t\n", stderr);
        return 1;
    }
    uint8_t *weights = malloc(weights_bytes);
    float *activation = malloc(activation_bytes);
    float *output = malloc(output_bytes);
    if (weights == NULL || activation == NULL || output == NULL) {
        fputs("host allocation failed\n", stderr);
        free(weights);
        free(activation);
        free(output);
        return 1;
    }
    const bool ok = read_exact_file(options->weights_path, weights_bytes, weights) &&
                    read_exact_file(options->activation_path, activation_bytes, activation) &&
                    scalar_golden(weights, activation, (size_t) options->rows,
                                  (size_t) options->columns, output) &&
                    write_cpu_outputs(options, output, weights_bytes, activation_bytes, output_bytes);
    free(weights);
    free(activation);
    free(output);
    return ok ? 0 : 1;
}

#define CL_CALL(status_lvalue, expression) do { \
    (status_lvalue) = (expression); \
    if ((status_lvalue) != CL_SUCCESS) { \
        failed_call = #expression; \
        goto cl_failure; \
    } \
} while (0)

static bool release_status(const char *name, cl_int status) {
    if (status == CL_SUCCESS) {
        return true;
    }
    fprintf(stderr, "%s failed during cleanup with OpenCL status %d\n", name, status);
    return false;
}

static void print_program_build_log(const struct opencl_api *api, cl_program program,
                                    cl_device_id device) {
    if (api->clGetProgramBuildInfo == NULL || program == NULL || device == NULL) {
        return;
    }
    size_t size = 0;
    cl_int status = api->clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    if (status != CL_SUCCESS || size == 0 || size > 16 * 1024 * 1024) {
        return;
    }
    char *log = calloc(size, 1);
    if (log == NULL) {
        return;
    }
    status = api->clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
    if (status == CL_SUCCESS) {
        fprintf(stderr, "OpenCL build log:\n%s\n", log);
    }
    free(log);
}

static int run_gpu(const struct options *options) {
    int result = 1;
    bool core_success = false;
    bool cleanup_success = true;
    const char *failed_call = NULL;
    cl_int status = CL_SUCCESS;
    size_t weights_bytes = 0;
    size_t activation_bytes = 0;
    size_t output_bytes = 0;
    size_t kernel_length = 0;
    uint8_t *weights = NULL;
    float *activation = NULL;
    float *golden = NULL;
    float *output = NULL;
    char *kernel_source = NULL;
    struct sample_timing *samples = NULL;
    char *platform_name = NULL;
    char *device_name = NULL;
    char *driver_version = NULL;
    void *library = NULL;
    struct opencl_api api;
    memset(&api, 0, sizeof(api));
    cl_platform_id *platforms = NULL;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem weights_buffer = NULL;
    cl_mem activation_buffer = NULL;
    cl_mem output_buffer = NULL;
    cl_event event = NULL;
    double build_ms = 0.0;
    double weights_h2d_event_ms = 0.0;
    double weights_h2d_host_ms = 0.0;
    double activation_h2d_event_ms = 0.0;
    double activation_h2d_host_ms = 0.0;
    double d2h_event_ms = 0.0;
    double d2h_host_ms = 0.0;
    double host_total_ms = 0.0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;

    if (!derive_sizes(options, &weights_bytes, &activation_bytes, &output_bytes)) {
        fputs("shape byte count overflows size_t\n", stderr);
        goto cleanup;
    }
    if (!path_absent(options->output_jsonl_path) || !path_absent(options->output_f32_path)) {
        goto cleanup;
    }
    if (options->iterations > SIZE_MAX / sizeof(*samples)) {
        fputs("sample timing allocation overflows size_t\n", stderr);
        goto cleanup;
    }
    weights = malloc(weights_bytes);
    activation = malloc(activation_bytes);
    golden = malloc(output_bytes);
    output = malloc(output_bytes);
    samples = calloc((size_t) options->iterations, sizeof(*samples));
    kernel_source = read_kernel_source(options->kernel_path, &kernel_length);
    if (weights == NULL || activation == NULL || golden == NULL || output == NULL ||
        samples == NULL || kernel_source == NULL) {
        fputs("host allocation or kernel read failed\n", stderr);
        goto cleanup;
    }
    if (!read_exact_file(options->weights_path, weights_bytes, weights) ||
        !read_exact_file(options->activation_path, activation_bytes, activation) ||
        !scalar_golden(weights, activation, (size_t) options->rows,
                       (size_t) options->columns, golden)) {
        goto cleanup;
    }

    library = dlopen(POWERVR_OPENCL_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", POWERVR_OPENCL_LIBRARY, dlerror());
        goto cleanup;
    }
    if (!load_opencl_api(library, &api)) {
        goto cleanup;
    }

    cl_uint platform_count = 0;
    CL_CALL(status, api.clGetPlatformIDs(0, NULL, &platform_count));
    if (platform_count == 0 || platform_count > 64) {
        fprintf(stderr, "direct PowerVR runtime returned invalid platform count %u\n", platform_count);
        goto cleanup;
    }
    platforms = calloc(platform_count, sizeof(*platforms));
    if (platforms == NULL) {
        fputs("platform list allocation failed\n", stderr);
        goto cleanup;
    }
    CL_CALL(status, api.clGetPlatformIDs(platform_count, platforms, NULL));
    platform = platforms[0];

    cl_uint device_count = 0;
    CL_CALL(status, api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_count));
    if (device_count != 1) {
        fprintf(stderr, "direct PowerVR platform must expose exactly one GPU, got %u\n", device_count);
        goto cleanup;
    }
    CL_CALL(status, api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    if (!query_platform_string(&api, platform, CL_PLATFORM_NAME, &platform_name) ||
        !query_device_string(&api, device, CL_DEVICE_NAME, &device_name) ||
        !query_device_string(&api, device, CL_DRIVER_VERSION, &driver_version)) {
        goto cleanup;
    }

    CL_CALL(status, ((context = api.clCreateContext(NULL, 1, &device, NULL, NULL, &status)), status));
    if (context == NULL) {
        fputs("clCreateContext returned null with success status\n", stderr);
        goto cleanup;
    }
    CL_CALL(status, ((queue = api.clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status)), status));
    if (queue == NULL) {
        fputs("clCreateCommandQueue returned null with success status\n", stderr);
        goto cleanup;
    }
    const char *source_pointer = kernel_source;
    CL_CALL(status, ((program = api.clCreateProgramWithSource(context, 1, &source_pointer, &kernel_length, &status)), status));
    if (program == NULL) {
        fputs("clCreateProgramWithSource returned null with success status\n", stderr);
        goto cleanup;
    }
    const uint64_t build_start_ns = monotonic_ns();
    CL_CALL(status, api.clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL));
    const uint64_t build_end_ns = monotonic_ns();
    if (build_start_ns == 0 || build_end_ns < build_start_ns) {
        fputs("monotonic clock failed while timing build\n", stderr);
        goto cleanup;
    }
    build_ms = (double) (build_end_ns - build_start_ns) / 1000000.0;
    CL_CALL(status, ((kernel = api.clCreateKernel(program, "q1_packed_gemv", &status)), status));
    if (kernel == NULL) {
        fputs("clCreateKernel returned null with success status\n", stderr);
        goto cleanup;
    }

    CL_CALL(status, ((weights_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY, weights_bytes, NULL, &status)), status));
    CL_CALL(status, ((activation_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY, activation_bytes, NULL, &status)), status));
    CL_CALL(status, ((output_buffer = api.clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_bytes, NULL, &status)), status));
    if (weights_buffer == NULL || activation_buffer == NULL || output_buffer == NULL) {
        fputs("clCreateBuffer returned null with success status\n", stderr);
        goto cleanup;
    }

    const cl_ulong rows_argument = options->rows;
    const cl_ulong columns_argument = options->columns;
    CL_CALL(status, api.clSetKernelArg(kernel, 0, sizeof(weights_buffer), &weights_buffer));
    CL_CALL(status, api.clSetKernelArg(kernel, 1, sizeof(activation_buffer), &activation_buffer));
    CL_CALL(status, api.clSetKernelArg(kernel, 2, sizeof(rows_argument), &rows_argument));
    CL_CALL(status, api.clSetKernelArg(kernel, 3, sizeof(columns_argument), &columns_argument));
    CL_CALL(status, api.clSetKernelArg(kernel, 4, sizeof(output_buffer), &output_buffer));

    const size_t global_size = (size_t) options->rows;
    const size_t local_size = (size_t) options->local_size;
    const uint64_t weights_h2d_host_start_ns = monotonic_ns();
    const uint64_t host_total_start_ns = weights_h2d_host_start_ns;
    if (weights_h2d_host_start_ns == 0) {
        fputs("monotonic clock failed before H2D\n", stderr);
        goto cleanup;
    }
    CL_CALL(status, api.clEnqueueWriteBuffer(queue, weights_buffer, CL_TRUE, 0, weights_bytes, weights, 0, NULL, &event));
    const uint64_t weights_h2d_host_end_ns = monotonic_ns();
    if (weights_h2d_host_end_ns < weights_h2d_host_start_ns) {
        fputs("invalid weights H2D host timing\n", stderr);
        goto cleanup;
    }
    weights_h2d_host_ms =
        (double) (weights_h2d_host_end_ns - weights_h2d_host_start_ns) / 1000000.0;
    if (event == NULL || !event_duration_ms(&api, event, &weights_h2d_event_ms)) goto cleanup;
    CL_CALL(status, api.clReleaseEvent(event));
    event = NULL;
    const uint64_t activation_h2d_host_start_ns = monotonic_ns();
    if (activation_h2d_host_start_ns == 0) {
        fputs("monotonic clock failed before activation H2D\n", stderr);
        goto cleanup;
    }
    CL_CALL(status, api.clEnqueueWriteBuffer(queue, activation_buffer, CL_TRUE, 0, activation_bytes, activation, 0, NULL, &event));
    const uint64_t activation_h2d_host_end_ns = monotonic_ns();
    if (activation_h2d_host_end_ns < activation_h2d_host_start_ns) {
        fputs("invalid activation H2D host timing\n", stderr);
        goto cleanup;
    }
    activation_h2d_host_ms =
        (double) (activation_h2d_host_end_ns - activation_h2d_host_start_ns) / 1000000.0;
    if (event == NULL || !event_duration_ms(&api, event, &activation_h2d_event_ms)) goto cleanup;
    CL_CALL(status, api.clReleaseEvent(event));
    event = NULL;
    CL_CALL(status, api.clFinish(queue));

    for (uint64_t index = 0; index < options->warmup; ++index) {
        double ignored_ms = 0.0;
        CL_CALL(status, api.clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event));
        CL_CALL(status, api.clFinish(queue));
        if (event == NULL || !event_duration_ms(&api, event, &ignored_ms)) goto cleanup;
        CL_CALL(status, api.clReleaseEvent(event));
        event = NULL;
    }
    for (uint64_t index = 0; index < options->iterations; ++index) {
        const uint64_t host_start_ns = monotonic_ns();
        if (host_start_ns == 0) {
            fputs("monotonic clock failed before kernel launch\n", stderr);
            goto cleanup;
        }
        CL_CALL(status, api.clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event));
        CL_CALL(status, api.clFinish(queue));
        const uint64_t host_end_ns = monotonic_ns();
        if (host_end_ns < host_start_ns || event == NULL ||
            !event_duration_ms(&api, event, &samples[index].kernel_event_ms)) {
            fputs("invalid kernel timing\n", stderr);
            goto cleanup;
        }
        samples[index].host_submit_wait_ms = (double) (host_end_ns - host_start_ns) / 1000000.0;
        CL_CALL(status, api.clReleaseEvent(event));
        event = NULL;
    }
    const uint64_t d2h_host_start_ns = monotonic_ns();
    if (d2h_host_start_ns == 0) {
        fputs("monotonic clock failed before D2H\n", stderr);
        goto cleanup;
    }
    CL_CALL(status, api.clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, output_bytes, output, 0, NULL, &event));
    const uint64_t d2h_host_end_ns = monotonic_ns();
    if (d2h_host_end_ns < d2h_host_start_ns) {
        fputs("invalid D2H host timing\n", stderr);
        goto cleanup;
    }
    d2h_host_ms = (double) (d2h_host_end_ns - d2h_host_start_ns) / 1000000.0;
    if (event == NULL || !event_duration_ms(&api, event, &d2h_event_ms)) goto cleanup;
    CL_CALL(status, api.clReleaseEvent(event));
    event = NULL;
    CL_CALL(status, api.clFinish(queue));
    const uint64_t host_total_end_ns = monotonic_ns();
    if (host_total_end_ns < host_total_start_ns) {
        fputs("invalid host total timing\n", stderr);
        goto cleanup;
    }
    host_total_ms = (double) (host_total_end_ns - host_total_start_ns) / 1000000.0;
    if (!compare_golden(golden, output, (size_t) options->rows,
                        &max_abs_error, &mean_abs_error)) {
        goto cleanup;
    }
    core_success = true;
    goto cleanup;

cl_failure:
    fprintf(stderr, "%s failed with OpenCL status %d\n",
            failed_call != NULL ? failed_call : "OpenCL call", status);
    if (failed_call != NULL && strstr(failed_call, "clBuildProgram") != NULL) {
        print_program_build_log(&api, program, device);
    }

cleanup:
    if (event != NULL && api.clReleaseEvent != NULL) {
        cleanup_success &= release_status("clReleaseEvent", api.clReleaseEvent(event));
    }
    if (output_buffer != NULL && api.clReleaseMemObject != NULL) {
        cleanup_success &= release_status("clReleaseMemObject(output)", api.clReleaseMemObject(output_buffer));
    }
    if (activation_buffer != NULL && api.clReleaseMemObject != NULL) {
        cleanup_success &= release_status("clReleaseMemObject(activation)", api.clReleaseMemObject(activation_buffer));
    }
    if (weights_buffer != NULL && api.clReleaseMemObject != NULL) {
        cleanup_success &= release_status("clReleaseMemObject(weights)", api.clReleaseMemObject(weights_buffer));
    }
    if (kernel != NULL && api.clReleaseKernel != NULL) {
        cleanup_success &= release_status("clReleaseKernel", api.clReleaseKernel(kernel));
    }
    if (program != NULL && api.clReleaseProgram != NULL) {
        cleanup_success &= release_status("clReleaseProgram", api.clReleaseProgram(program));
    }
    if (queue != NULL && api.clReleaseCommandQueue != NULL) {
        cleanup_success &= release_status("clReleaseCommandQueue", api.clReleaseCommandQueue(queue));
    }
    if (context != NULL && api.clReleaseContext != NULL) {
        cleanup_success &= release_status("clReleaseContext", api.clReleaseContext(context));
    }
    if (library != NULL && dlclose(library) != 0) {
        fprintf(stderr, "dlclose(%s) failed: %s\n", POWERVR_OPENCL_LIBRARY, dlerror());
        cleanup_success = false;
    }
    if (core_success && cleanup_success &&
        write_gpu_outputs(options, output, weights_bytes, activation_bytes, output_bytes,
                          platform_name, device_name, driver_version, build_ms,
                          weights_h2d_event_ms, weights_h2d_host_ms,
                          activation_h2d_event_ms, activation_h2d_host_ms,
                          d2h_event_ms, d2h_host_ms, host_total_ms,
                          max_abs_error, mean_abs_error, samples)) {
        result = 0;
    }
    free(platforms);
    free(platform_name);
    free(device_name);
    free(driver_version);
    free(kernel_source);
    free(samples);
    free(weights);
    free(activation);
    free(golden);
    free(output);
    return result;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--print-contract") == 0) {
        print_contract();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(stdout, argv[0]);
        return 0;
    }
    struct options options;
    const int parse_status = parse_options(argc, argv, &options);
    if (parse_status != 0) {
        usage(stderr, argv[0]);
        return parse_status;
    }
    if (options.cpu_reference_only) {
        return run_cpu_reference(&options);
    }
    return run_gpu(&options);
}
