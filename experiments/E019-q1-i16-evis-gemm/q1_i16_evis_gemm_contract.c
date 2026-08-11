#include "q1_i16_evis_gemm_contract.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message)
{
    if (error != NULL && error_size != 0U) {
        (void)snprintf(error, error_size, "%s", message);
    }
}

int q1_i16_evis_gemm_contract_build(
    uint32_t m,
    uint32_t k,
    uint32_t n,
    q1_i16_evis_gemm_contract *contract,
    char *error,
    size_t error_size)
{
    const uint32_t dfp_fixed_point_pos = 8U;
    const uint32_t global_scale_x = 4U;
    const uint32_t global_scale_y = 4U;
    const uint32_t q_fixed_for_pow2_scale = UINT32_C(1) << 30;
    const uint32_t multiplier = q_fixed_for_pow2_scale >> 15;
    const int32_t post_shift = 15 + (int32_t)dfp_fixed_point_pos;
    uint64_t physical_max_abs_dot;

    if (contract == NULL) {
        set_error(error, error_size, "contract output must not be null");
        return 1;
    }
    if (m == 0U) {
        set_error(error, error_size, "M must be positive");
        return 1;
    }
    if (n != 1U) {
        set_error(error, error_size, "N must be 1");
        return 1;
    }
    if (k != 32U && k != 128U) {
        set_error(error, error_size, "K must be 32 or 128");
        return 1;
    }

    (void)memset(contract, 0, sizeof(*contract));
    contract->m = m;
    contract->k = k;
    contract->n = n;
    contract->a_dims[0] = k;
    contract->a_dims[1] = m;
    contract->b_dims[0] = n;
    contract->b_dims[1] = k;
    contract->c_dims[0] = n;
    contract->c_dims[1] = m;
    contract->dfp_fixed_point_pos = dfp_fixed_point_pos;
    contract->parameter_count = 10U;
    contract->scalar_count = 7U;
    contract->uniform_count = 9U;
    contract->global_scale[0] = global_scale_x;
    contract->global_scale[1] = global_scale_y;
    contract->global_scale[2] = 1U;
    contract->global_size[0] = global_scale_x;
    contract->global_size[1] =
        ((m + global_scale_y - 1U) / global_scale_y + 3U) & ~UINT32_C(3);
    contract->global_size[2] = 1U;
    contract->input0_zero_point = 0;
    contract->input1_zero_point = 0;
    contract->output_zero_point = 0.0;
    contract->output_scale = 256.0;
    contract->multiplier0 = multiplier;
    contract->multiplier1 = multiplier;
    contract->post_shift0 = post_shift;
    contract->post_shift1 = post_shift;
    physical_max_abs_dot = (uint64_t)k * UINT64_C(128);
    contract->physical_max_abs_dot = physical_max_abs_dot;
    contract->int16_safe = physical_max_abs_dot <= (uint64_t)INT16_MAX;
    return 0;
}
