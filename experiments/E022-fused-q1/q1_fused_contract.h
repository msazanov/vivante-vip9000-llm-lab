#ifndef E022_Q1_FUSED_CONTRACT_H
#define E022_Q1_FUSED_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

/*
 * Exact reference for one packed Q1 row and one signed INT8 activation row.
 * Q1 bits are LSB-first: zero -> -1, one -> +1.
 * The caller must provide K divisible by 8 and enough packed bytes.
 */
static inline int32_t q1_fused_dot_from_bits(
    const int8_t *q, const uint8_t *packed, size_t k)
{
    int32_t selected = 0;
    int32_t total = 0;
    size_t i;
    for (i = 0; i < k; ++i) {
        const int32_t value = (int32_t)q[i];
        const uint8_t bit = (uint8_t)((packed[i >> 3] >> (i & 7u)) & 1u);
        selected += bit ? value : 0;
        total += value;
    }
    return (selected * 2) - total;
}

#endif
