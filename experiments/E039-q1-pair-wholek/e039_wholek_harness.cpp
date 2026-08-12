// E039 whole-K target microharness.
//
// This is intended to run on the AArch64 DOTPROD target after cross-build.
// It packs the native Prism layouts directly, invokes the assembly helper,
// checks against an independent scalar native-order oracle, and times only
// the paired helper versus two native-equivalent NEON/DOTPROD group calls.

#include <algorithm>
#include <array>
#include <arm_neon.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

struct block_q1_0x4 {
    uint16_t d[4];
    int8_t   qs[64];
};
static_assert(sizeof(block_q1_0x4) == 72, "native block_q1_0x4 must be 72 bytes");

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "native block_q8_0 must be 34 bytes");

extern "C" void e039_q1_pair_wholek(float * dst,
                                      const block_q1_0x4 * b0,
                                      const block_q1_0x4 * b1,
                                      const block_q8_0 * q8,
                                      const uint64_t * table_q1_signs,
                                      uint32_t nb);

// The upstream ARM DOTPROD path uses this exact table shape.  It is initialized
// from make_table() before any target kernel or baseline call.
alignas(16) static uint64_t g_table_q1_signs[256];

static float half_to_float(uint16_t h) {
    const unsigned sign = h >> 15;
    const unsigned exp  = (h >> 10) & 0x1f;
    const unsigned frac = h & 0x3ff;
    float value;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(frac), -24);
    } else if (exp == 0x1f) {
        value = frac ? std::numeric_limits<float>::quiet_NaN()
                     : std::numeric_limits<float>::infinity();
    } else {
        value = std::ldexp(1024.0f + static_cast<float>(frac),
                           static_cast<int>(exp) - 25);
    }
    return sign ? -value : value;
}

// This is the ARM implementation selected by GGML_CPU_FP16_TO_FP32 in
// simd-mappings.h (neon_compute_fp16_to_fp32), kept local so this fixture has
// no dependency on the full ggml headers.
static float neon_fp16_to_float(uint16_t h) {
    __fp16 tmp;
    std::memcpy(&tmp, &h, sizeof(h));
    return static_cast<float>(tmp);
}

static uint64_t make_table_entry(unsigned bits) {
    uint64_t result = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
        const uint64_t sign = (bits & (1u << bit)) ? 0x01u : 0xffu;
        result |= sign << (8u * bit);
    }
    return result;
}

static std::array<uint64_t, 256> make_table() {
    std::array<uint64_t, 256> table{};
    for (unsigned i = 0; i < table.size(); ++i) {
        table[i] = make_table_entry(i);
    }
    return table;
}

static uint32_t next_u32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static uint16_t random_finite_half(uint32_t & state) {
    uint16_t h = static_cast<uint16_t>(next_u32(state));
    if (((h >> 10) & 0x1f) == 0x1f) {
        h = static_cast<uint16_t>((h & 0x83ffu) | 0x7800u); // finite exponent 30
    }
    return h;
}

static bool sign_value(unsigned pattern, unsigned group, unsigned block,
                       unsigned row, unsigned value) {
    switch (pattern % 6) {
        case 0: return false;
        case 1: return true;
        case 2: return ((value + row + block + group) & 1u) != 0;
        case 3: return ((value + row * 3u + block) & 3u) == 0;
        case 4: return ((0x55u >> ((value + row + group) & 7u)) & 1u) != 0;
        default: {
            uint32_t x = 0x9e3779b9u ^ (group * 0x45d9f3bu)
                       ^ (block * 0x119de1f3u);
            x ^= row * 0x27d4eb2du + value * 0x165667b1u;
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            return (x & 1u) != 0;
        }
    }
}

static int8_t q8_value(unsigned pattern, unsigned index, uint32_t & state) {
    static constexpr int8_t extremes[] = {-128, -127, -1, 0, 1, 126, 127};
    switch (pattern % 6) {
        case 0: return 0;
        case 1: return 1;
        case 2: return (index & 1u) ? -3 : 5;
        case 3: return extremes[index % (sizeof(extremes) / sizeof(extremes[0]))];
        case 4: return static_cast<int8_t>(static_cast<int>(next_u32(state) % 255u) - 127);
        default: return extremes[(index * 5u + pattern) % (sizeof(extremes) / sizeof(extremes[0]))];
    }
}

static void pack_fixture(int K, unsigned pattern,
                         std::vector<block_q1_0x4> & b0,
                         std::vector<block_q1_0x4> & b1,
                         std::vector<block_q8_0> & q8) {
    assert(K > 0 && K % 128 == 0);
    const int nb = K / 128;
    b0.resize(nb);
    b1.resize(nb);
    q8.resize(static_cast<size_t>(nb) * 4);
    uint32_t state = 0xe0390000u ^ static_cast<uint32_t>(K) ^ pattern * 0x10001u;

    for (int l = 0; l < nb; ++l) {
        for (unsigned row = 0; row < 4; ++row) {
            // Independent Q1 sequences for the two adjacent four-row groups.
            b0[l].d[row] = (pattern < 4)
                ? std::array<uint16_t, 8>{0x0000, 0x3c00, 0x3555, 0x3a00,
                                          0x4000, 0xbc00, 0x7bff, 0x0400}[(row + l) & 7]
                : random_finite_half(state);
            b1[l].d[row] = (pattern < 4)
                ? std::array<uint16_t, 8>{0x3c00, 0x3800, 0x3e00, 0x4200,
                                          0x3000, 0xbc00, 0x7bff, 0x0001}[(row + 3u * l + 1u) & 7]
                : random_finite_half(state);
        }

        for (unsigned k = 0; k < 4; ++k) {
            for (unsigned tile = 0; tile < 8; ++tile) {
                uint8_t b0_lo = 0, b0_hi = 0, b1_lo = 0, b1_hi = 0;
                for (unsigned p = 0; p < 4; ++p) {
                    const unsigned value = k * 32u + tile * 4u + p;
                    if (sign_value(pattern, 0, static_cast<unsigned>(l), 0, value)) b0_lo |= static_cast<uint8_t>(1u << p);
                    if (sign_value(pattern, 0, static_cast<unsigned>(l), 1, value)) b0_lo |= static_cast<uint8_t>(1u << (4u + p));
                    if (sign_value(pattern, 0, static_cast<unsigned>(l), 2, value)) b0_hi |= static_cast<uint8_t>(1u << p);
                    if (sign_value(pattern, 0, static_cast<unsigned>(l), 3, value)) b0_hi |= static_cast<uint8_t>(1u << (4u + p));
                    if (sign_value(pattern, 1, static_cast<unsigned>(l), 0, value)) b1_lo |= static_cast<uint8_t>(1u << p);
                    if (sign_value(pattern, 1, static_cast<unsigned>(l), 1, value)) b1_lo |= static_cast<uint8_t>(1u << (4u + p));
                    if (sign_value(pattern, 1, static_cast<unsigned>(l), 2, value)) b1_hi |= static_cast<uint8_t>(1u << p);
                    if (sign_value(pattern, 1, static_cast<unsigned>(l), 3, value)) b1_hi |= static_cast<uint8_t>(1u << (4u + p));
                }
                b0[l].qs[k * 16u + tile * 2u + 0u] = static_cast<int8_t>(b0_lo);
                b0[l].qs[k * 16u + tile * 2u + 1u] = static_cast<int8_t>(b0_hi);
                b1[l].qs[k * 16u + tile * 2u + 0u] = static_cast<int8_t>(b1_lo);
                b1[l].qs[k * 16u + tile * 2u + 1u] = static_cast<int8_t>(b1_hi);
            }

            block_q8_0 & a = q8[static_cast<size_t>(l) * 4u + k];
            a.d = (pattern < 4)
                ? std::array<uint16_t, 8>{0x3c00, 0x3800, 0x3e00, 0x4200,
                                          0x3000, 0xbc00, 0x7bff, 0x0001}[(k + 2u * l + pattern) & 7]
                : random_finite_half(state);
            for (unsigned i = 0; i < 32; ++i) {
                a.qs[i] = q8_value(pattern, i + k * 2u + l * 3u, state);
            }
        }
    }
}

static void scalar_native_group(float * dst, const block_q1_0x4 * b,
                                const block_q8_0 * q8, int nb) {
    float acc[4] = {0, 0, 0, 0};
    for (int l = 0; l < nb; ++l) {
        float accb[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; ++k) {
            int dot[4] = {0, 0, 0, 0};
            const block_q8_0 & a = q8[l * 4 + k];
            for (int tile = 0; tile < 8; ++tile) {
                const uint8_t * bits = reinterpret_cast<const uint8_t *>(b[l].qs) + k * 16 + tile * 2;
                for (int row = 0; row < 4; ++row) {
                    const unsigned shift = (row & 1) ? 4u : 0u;
                    for (int p = 0; p < 4; ++p) {
                        const int sign = (bits[row / 2] & (1u << (shift + p))) ? 1 : -1;
                        dot[row] += sign * static_cast<int>(a.qs[tile * 4 + p]);
                    }
                }
            }
            const float ad = half_to_float(a.d);
            for (int row = 0; row < 4; ++row) {
                accb[row] = std::fmaf(static_cast<float>(dot[row]), ad, accb[row]);
            }
        }
        for (int row = 0; row < 4; ++row) {
            acc[row] = std::fmaf(accb[row], half_to_float(b[l].d[row]), acc[row]);
        }
    }
    std::memcpy(dst, acc, sizeof(acc));
}

static inline int8x16_t e039_q1_unpack_pair(uint8_t bits0, uint8_t bits1) {
    return vreinterpretq_s8_u8(vcombine_u8(vcreate_u8(g_table_q1_signs[bits0]),
                                           vcreate_u8(g_table_q1_signs[bits1])));
}

// Literal native-equivalent baseline from the upstream ARM q1_0 4-row loop:
//   ggml/src/ggml-cpu/arch/arm/repack.cpp:1862-1898
// Keep this noinline/noclone so the benchmark cannot replace two native calls
// with the paired assembly call or eliminate their work.
__attribute__((noinline, noclone, used))
static void native_simd_group(float * dst, const block_q1_0x4 * b,
                              const block_q8_0 * q8, int nb) {
    const block_q1_0x4 * b_ptr = b;
    const block_q8_0 * a_ptr = q8;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int l = 0; l < nb; ++l) {
        const float32x4_t b_d = vcvt_f32_f16(
            vld1_f16(reinterpret_cast<const float16_t *>(b_ptr[l].d)));
        float32x4_t accb = vdupq_n_f32(0.0f);

        for (int k = 0; k < 4; ++k) {
            const block_q8_0 * a_blk = a_ptr + l * 4 + k;
            const float ad = neon_fp16_to_float(a_blk->d);
            const uint8_t * b_qs = reinterpret_cast<const uint8_t *>(b_ptr[l].qs) + k * 16;
            int32x4_t ret = vdupq_n_s32(0);

            for (int tile = 0; tile < 8; tile += 4) {
                const int8x16_t signs0 = e039_q1_unpack_pair(
                    b_qs[2 * (tile + 0) + 0], b_qs[2 * (tile + 0) + 1]);
                const int8x16_t signs1 = e039_q1_unpack_pair(
                    b_qs[2 * (tile + 1) + 0], b_qs[2 * (tile + 1) + 1]);
                const int8x16_t signs2 = e039_q1_unpack_pair(
                    b_qs[2 * (tile + 2) + 0], b_qs[2 * (tile + 2) + 1]);
                const int8x16_t signs3 = e039_q1_unpack_pair(
                    b_qs[2 * (tile + 3) + 0], b_qs[2 * (tile + 3) + 1]);
                const int8x16_t q_tiles = vld1q_s8(a_blk->qs + tile * 4);

                ret = vdotq_laneq_s32(ret, signs0, q_tiles, 0);
                ret = vdotq_laneq_s32(ret, signs1, q_tiles, 1);
                ret = vdotq_laneq_s32(ret, signs2, q_tiles, 2);
                ret = vdotq_laneq_s32(ret, signs3, q_tiles, 3);
            }

            accb = vfmaq_n_f32(accb, vcvtq_f32_s32(ret), ad);
        }
        acc = vfmaq_f32(acc, accb, b_d);
    }
    vst1q_f32(dst, acc);
}

static bool close_enough(float got, float want) {
    if (std::memcmp(&got, &want, sizeof(float)) == 0) return true;
    if (!std::isfinite(got) || !std::isfinite(want)) return false;
    const float scale = std::max(1.0f, std::max(std::fabs(got), std::fabs(want)));
    return std::fabs(got - want) <= 4.0f * std::numeric_limits<float>::epsilon() * scale;
}

static bool bit_equal(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static void check_case(int K, unsigned pattern, const uint64_t * table) {
    const int nb = K / 128;
    std::vector<block_q1_0x4> b0, b1;
    std::vector<block_q8_0> q8;
    pack_fixture(K, pattern, b0, b1, q8);
    const auto b0_before = b0;
    const auto b1_before = b1;
    const auto q8_before = q8;

    float got[8] = {};
    float want[8] = {};
    float simd[8] = {};
    scalar_native_group(want + 0, b0.data(), q8.data(), nb);
    scalar_native_group(want + 4, b1.data(), q8.data(), nb);
    native_simd_group(simd + 0, b0.data(), q8.data(), nb);
    native_simd_group(simd + 4, b1.data(), q8.data(), nb);
    bool scalar_simd_bit_exact = true;
    bool paired_scalar_bit_exact = true;
    bool paired_simd_bit_exact = true;
    for (int i = 0; i < 8; ++i) {
        scalar_simd_bit_exact &= bit_equal(simd[i], want[i]);
        if (!close_enough(simd[i], want[i])) {
            std::fprintf(stderr, "native SIMD mismatch K=%d pattern=%u row=%d simd=%a scalar=%a\n",
                         K, pattern, i, simd[i], want[i]);
            std::abort();
        }
    }
    e039_q1_pair_wholek(got, b0.data(), b1.data(), q8.data(), table, nb);
    for (int i = 0; i < 8; ++i) {
        paired_scalar_bit_exact &= bit_equal(got[i], want[i]);
        paired_simd_bit_exact &= bit_equal(got[i], simd[i]);
        if (!close_enough(got[i], want[i])) {
            std::fprintf(stderr, "mismatch K=%d pattern=%u row=%d got=%a want=%a\n",
                         K, pattern, i, got[i], want[i]);
            std::abort();
        }
        if (!bit_equal(got[i], simd[i])) {
            std::fprintf(stderr, "paired/SIMD mismatch K=%d pattern=%u row=%d paired=%a simd=%a\n",
                         K, pattern, i, got[i], simd[i]);
            std::abort();
        }
    }
    std::printf("CASE K=%d pattern=%u scalar_vs_simd_bit_exact=%s paired_vs_scalar_bit_exact=%s paired_vs_simd_bit_exact=%s\n",
                K, pattern, scalar_simd_bit_exact ? "yes" : "no",
                paired_scalar_bit_exact ? "yes" : "no",
                paired_simd_bit_exact ? "yes" : "no");
    assert(std::memcmp(b0.data(), b0_before.data(), b0.size() * sizeof(b0[0])) == 0);
    assert(std::memcmp(b1.data(), b1_before.data(), b1.size() * sizeof(b1[0])) == 0);
    assert(std::memcmp(q8.data(), q8_before.data(), q8.size() * sizeof(q8[0])) == 0);
}

static volatile float g_bench_sink = 0.0f;

struct bench_result {
    double ns_per_call;
    uint64_t calls;
};

static uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
}

static bench_result bench_paired(float * out, const block_q1_0x4 * b0,
                                 const block_q1_0x4 * b1,
                                 const block_q8_0 * q8,
                                 const uint64_t * table, int nb,
                                 uint64_t budget_ns) {
    uint64_t calls = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        e039_q1_pair_wholek(out, b0, b1, q8, table, nb);
        g_bench_sink += out[0] + out[4];
        ++calls;
    } while (elapsed_ns(start) < budget_ns);
    return {static_cast<double>(elapsed_ns(start)) / calls, calls};
}

static bench_result bench_two_native_simd(float * out, const block_q1_0x4 * b0,
                                          const block_q1_0x4 * b1,
                                          const block_q8_0 * q8, int nb,
                                          uint64_t budget_ns) {
    uint64_t calls = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        native_simd_group(out + 0, b0, q8, nb);
        native_simd_group(out + 4, b1, q8, nb);
        // Consume both output vectors so neither noinline call can be removed.
        g_bench_sink += out[0] + out[4];
        ++calls;
    } while (elapsed_ns(start) < budget_ns);
    return {static_cast<double>(elapsed_ns(start)) / calls, calls};
}

static void microgate(int K, unsigned pattern, const uint64_t * table) {
    const int nb = K / 128;
    std::vector<block_q1_0x4> b0, b1;
    std::vector<block_q8_0> q8;
    pack_fixture(K, pattern, b0, b1, q8);
    float out[8] = {};
    constexpr uint64_t warmup_calls = 128;
    constexpr uint64_t budget_ns = 25'000'000; // 25 ms per side and round

    for (uint64_t i = 0; i < warmup_calls; ++i) {
        if (i & 1u) {
            native_simd_group(out + 0, b0.data(), q8.data(), nb);
            native_simd_group(out + 4, b1.data(), q8.data(), nb);
        } else {
            e039_q1_pair_wholek(out, b0.data(), b1.data(), q8.data(), table, nb);
        }
        g_bench_sink += out[0] + out[4];
    }

    bench_result pair_results[4]{};
    bench_result native_results[4]{};
    for (unsigned round = 0; round < 4; ++round) {
        if ((round & 1u) == 0) {
            pair_results[round] = bench_paired(out, b0.data(), b1.data(), q8.data(),
                                               table, nb, budget_ns);
            native_results[round] = bench_two_native_simd(out, b0.data(), b1.data(),
                                                          q8.data(), nb, budget_ns);
        } else {
            native_results[round] = bench_two_native_simd(out, b0.data(), b1.data(),
                                                          q8.data(), nb, budget_ns);
            pair_results[round] = bench_paired(out, b0.data(), b1.data(), q8.data(),
                                               table, nb, budget_ns);
        }
        std::printf("MICROGATE K=%d pattern=%u round=%u order=%s paired_asm_ns=%.1f(%llu) two_native_simd_ns=%.1f(%llu)\n",
                    K, pattern, round, (round & 1u) ? "native-first" : "paired-first",
                    pair_results[round].ns_per_call,
                    static_cast<unsigned long long>(pair_results[round].calls),
                    native_results[round].ns_per_call,
                    static_cast<unsigned long long>(native_results[round].calls));
    }
    std::array<double, 4> pair_times{
        pair_results[0].ns_per_call, pair_results[1].ns_per_call,
        pair_results[2].ns_per_call, pair_results[3].ns_per_call};
    std::array<double, 4> native_times{
        native_results[0].ns_per_call, native_results[1].ns_per_call,
        native_results[2].ns_per_call, native_results[3].ns_per_call};
    std::sort(pair_times.begin(), pair_times.end());
    std::sort(native_times.begin(), native_times.end());
    std::printf("MICROGATE_SUMMARY K=%d paired_median_ns=%.1f two_native_simd_median_ns=%.1f sink=%a\n",
                K, (pair_times[1] + pair_times[2]) * 0.5,
                (native_times[1] + native_times[2]) * 0.5,
                static_cast<float>(g_bench_sink));
}

int main() {
    const auto table = make_table();
    std::memcpy(g_table_q1_signs, table.data(), sizeof(g_table_q1_signs));
    assert(table[0] == 0xffffffffffffffffULL);
    assert(table[255] == 0x0101010101010101ULL);
    // Invalid signed-negative counts (bit 31 set) are rejected before input
    // dereferences and produce the documented zero result.
    std::array<float, 8> invalid{};
    std::memset(invalid.data(), 0xa5, sizeof(invalid));
    e039_q1_pair_wholek(invalid.data(), nullptr, nullptr, nullptr,
                        table.data(), UINT32_MAX);
    for (float value : invalid) {
        assert(bit_equal(value, 0.0f));
    }
    for (int K : {128, 256, 5120}) {
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
            check_case(K, pattern, table.data());
        }
        microgate(K, 5, table.data());
    }
    std::puts("PASS E039 whole-K packed-Q1 oracle: K=128,256,5120 patterns=6 no mutation");
    return 0;
}
