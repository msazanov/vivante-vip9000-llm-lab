#ifndef Q1_I16_EVIS_GEMM_CONTRACT_H
#define Q1_I16_EVIS_GEMM_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t a_dims[2];
    uint32_t b_dims[2];
    uint32_t c_dims[2];
    uint32_t dfp_fixed_point_pos;
    uint32_t parameter_count;
    uint32_t scalar_count;
    uint32_t uniform_count;
    uint32_t global_scale[3];
    uint32_t global_size[3];
    int32_t input0_zero_point;
    int32_t input1_zero_point;
    double output_zero_point;
    double output_scale;
    uint32_t multiplier0;
    uint32_t multiplier1;
    int32_t post_shift0;
    int32_t post_shift1;
    uint64_t physical_max_abs_dot;
    int int16_safe;
} q1_i16_evis_gemm_contract;

int q1_i16_evis_gemm_contract_build(
    uint32_t m,
    uint32_t k,
    uint32_t n,
    q1_i16_evis_gemm_contract *contract,
    char *error,
    size_t error_size);

#endif
