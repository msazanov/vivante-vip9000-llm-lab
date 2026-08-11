#include "q1_i16_evis_gemm_contract.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int parse_u32(const char *text, uint32_t *value)
{
    uint64_t parsed = 0U;
    const char *cursor = text;
    if (text == NULL || *text == '\0') return 1;
    while (*cursor != '\0') {
        uint32_t digit;
        if (*cursor < '0' || *cursor > '9') return 1;
        digit = (uint32_t)(*cursor - '0');
        if (parsed > (UINT32_MAX - digit) / 10U) return 1;
        parsed = parsed * 10U + digit;
        ++cursor;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static void print_contract(const q1_i16_evis_gemm_contract *c)
{
    (void)printf(
        "{\"schema\":\"vip9000-q1-i16-evis-gemm-contract/v1\","
        "\"kernel\":\"com.vivantecorp.extension.evis.gemm_I16I16toI16\","
        "\"m\":%" PRIu32 ",\"k\":%" PRIu32 ",\"n\":%" PRIu32 ","
        "\"a_dims\":[%" PRIu32 ",%" PRIu32 "],"
        "\"b_dims\":[%" PRIu32 ",%" PRIu32 "],"
        "\"c_dims\":[%" PRIu32 ",%" PRIu32 "],"
        "\"dfp_fixed_point_pos\":%" PRIu32 ","
        "\"parameter_count\":%" PRIu32 ",\"scalar_count\":%" PRIu32 ","
        "\"uniform_count\":%" PRIu32 ","
        "\"global_scale\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "],"
        "\"global_size\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "],"
        "\"input0_zero_point\":%" PRId32 ",\"input1_zero_point\":%" PRId32 ","
        "\"output_zero_point\":%.1f,\"output_scale\":%.1f,"
        "\"multiplier0\":%" PRIu32 ",\"multiplier1\":%" PRIu32 ","
        "\"post_shift0\":%" PRId32 ",\"post_shift1\":%" PRId32 ","
        "\"ac2zero\":0,\"bc2zero\":0,"
        "\"physical_max_abs_dot\":%" PRIu64 ",\"int16_safe\":%s}\n",
        c->m, c->k, c->n, c->a_dims[0], c->a_dims[1], c->b_dims[0], c->b_dims[1],
        c->c_dims[0], c->c_dims[1], c->dfp_fixed_point_pos, c->parameter_count,
        c->scalar_count, c->uniform_count, c->global_scale[0], c->global_scale[1],
        c->global_scale[2], c->global_size[0], c->global_size[1], c->global_size[2],
        c->input0_zero_point, c->input1_zero_point, c->output_zero_point,
        c->output_scale, c->multiplier0, c->multiplier1, c->post_shift0,
        c->post_shift1, c->physical_max_abs_dot, c->int16_safe ? "true" : "false");
}

int main(int argc, char **argv)
{
    uint32_t m, k, n = 1U;
    q1_i16_evis_gemm_contract contract;
    char error[80];
    if (argc != 3 && argc != 4) return 2;
    if (parse_u32(argv[1], &m) != 0 || m == 0U) return 2;
    if (parse_u32(argv[2], &k) != 0) return 2;
    if (argc == 4 && parse_u32(argv[3], &n) != 0) return 2;
    if (q1_i16_evis_gemm_contract_build(m, k, n, &contract, error, sizeof(error)) != 0) {
        (void)fprintf(stderr, "%s\n", error);
        return 2;
    }
    print_contract(&contract);
    return 0;
}
