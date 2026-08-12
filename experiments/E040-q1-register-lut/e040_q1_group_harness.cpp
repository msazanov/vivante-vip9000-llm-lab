// E040 standalone contract and microbenchmark harness.
//
// The helper under test processes one native block_q1_0x4 output group over
// whole K.  Q1 bytes remain packed in the native Prism layout; the candidate
// expands signs only in NEON registers through a 64-byte nibble LUT.

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

extern "C" void e040_q1_group_wholek(float * dst,
                                       const block_q1_0x4 * b,
                                       const block_q8_0 * q8,
                                       const uint8_t * nibble_lut,
                                       uint32_t nb);

alignas(16) static uint64_t g_table_q1_signs[256];

static float half_to_float(uint16_t h) {
    const unsigned sign = h >> 15;
    const unsigned exp = (h >> 10) & 0x1f;
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

static std::array<uint64_t, 256> make_sign_table() {
    std::array<uint64_t, 256> table{};
    for (unsigned i = 0; i < table.size(); ++i) {
        table[i] = make_table_entry(i);
    }
    return table;
}

// T[b][n] is the exact byte sign for bit b of nibble n.  The four 16-byte
// vectors are concatenated in memory because the assembly uses one TBL table.
static std::array<uint8_t, 64> make_nibble_lut() {
    std::array<uint8_t, 64> lut{};
    for (unsigned bit = 0; bit < 4; ++bit) {
        for (unsigned nibble = 0; nibble < 16; ++nibble) {
            lut[bit * 16 + nibble] = (nibble & (1u << bit)) ? 0x01 : 0xff;
        }
    }
    return lut;
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
        h = static_cast<uint16_t>((h & 0x83ffu) | 0x7800u);
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

static int8_t q8_value(unsigned pattern, unsigned value, uint32_t & state) {
    switch (pattern % 6) {
        case 0: return 0;
        case 1: return 1;
        case 2: return static_cast<int8_t>((value & 1u) ? 127 : -128);
        case 3: return static_cast<int8_t>((value * 17u + 3u) & 0x7f);
        case 4: return static_cast<int8_t>((value & 1u) ? -127 : 127);
        default: return static_cast<int8_t>(next_u32(state));
    }
}

static void pack_fixture(int K, unsigned pattern,
                         std::vector<block_q1_0x4> & b,
                         std::vector<block_q8_0> & q8) {
    const int nb = K / 128;
    b.assign(static_cast<size_t>(nb), {});
    q8.assign(static_cast<size_t>(nb) * 4, {});
    uint32_t state = 0x12345678u ^ static_cast<uint32_t>(K * 17 + pattern);

    for (int l = 0; l < nb; ++l) {
        for (unsigned row = 0; row < 4; ++row) {
            b[l].d[row] = (pattern == 4 && row == 3) ? 0x0001
                         : static_cast<uint16_t>(0x3000u + row * 0x400u
                                                  + (l + pattern) * 0x31u);
        }
        for (unsigned k = 0; k < 4; ++k) {
            for (unsigned tile = 0; tile < 8; ++tile) {
                uint8_t bytes[2] = {0, 0};
                for (unsigned row = 0; row < 4; ++row) {
                    uint8_t & packed = bytes[row / 2];
                    for (unsigned p = 0; p < 4; ++p) {
                        if (sign_value(pattern, static_cast<unsigned>(l),
                                       k, row, tile * 4 + p)) {
                            packed |= static_cast<uint8_t>(1u
                                      << ((row & 1u) * 4u + p));
                        }
                    }
                }
                b[l].qs[k * 16 + tile * 2 + 0] = static_cast<int8_t>(bytes[0]);
                b[l].qs[k * 16 + tile * 2 + 1] = static_cast<int8_t>(bytes[1]);
            }
        }
        for (unsigned k = 0; k < 4; ++k) {
            block_q8_0 & a = q8[static_cast<size_t>(l) * 4 + k];
            a.d = (pattern == 4 && k == 3) ? 0x7bff
                 : static_cast<uint16_t>(0x3000u + k * 0x300u
                                          + (l + pattern) * 0x17u);
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
                const uint8_t * bits = reinterpret_cast<const uint8_t *>(b[l].qs)
                                     + k * 16 + tile * 2;
                for (int row = 0; row < 4; ++row) {
                    const unsigned shift = (row & 1) ? 4u : 0u;
                    for (int p = 0; p < 4; ++p) {
                        const int sign = (bits[row / 2] & (1u << (shift + p)))
                                       ? 1 : -1;
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

static inline int8x16_t native_unpack_pair(uint8_t bits0, uint8_t bits1) {
    return vreinterpretq_s8_u8(vcombine_u8(vreinterpret_u8_u64(vcreate_u64(
        g_table_q1_signs[bits0])), vreinterpret_u8_u64(vcreate_u64(
        g_table_q1_signs[bits1]))));
}

// Literal native-equivalent one-group path from Prism repack.cpp.  Kept
// noinline/noclone so the timing comparison remains a real baseline.
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
            const uint8_t * b_qs = reinterpret_cast<const uint8_t *>(b_ptr[l].qs)
                                 + k * 16;
            int32x4_t ret = vdupq_n_s32(0);
            for (int tile = 0; tile < 8; tile += 4) {
                const int8x16_t s0 = native_unpack_pair(b_qs[2 * (tile + 0) + 0],
                                                         b_qs[2 * (tile + 0) + 1]);
                const int8x16_t s1 = native_unpack_pair(b_qs[2 * (tile + 1) + 0],
                                                         b_qs[2 * (tile + 1) + 1]);
                const int8x16_t s2 = native_unpack_pair(b_qs[2 * (tile + 2) + 0],
                                                         b_qs[2 * (tile + 2) + 1]);
                const int8x16_t s3 = native_unpack_pair(b_qs[2 * (tile + 3) + 0],
                                                         b_qs[2 * (tile + 3) + 1]);
                const int8x16_t q = vld1q_s8(a_blk->qs + tile * 4);
                ret = vdotq_laneq_s32(ret, s0, q, 0);
                ret = vdotq_laneq_s32(ret, s1, q, 1);
                ret = vdotq_laneq_s32(ret, s2, q, 2);
                ret = vdotq_laneq_s32(ret, s3, q, 3);
            }
            accb = vfmaq_n_f32(accb, vcvtq_f32_s32(ret), ad);
        }
        acc = vfmaq_f32(acc, accb, b_d);
    }
    vst1q_f32(dst, acc);
}

static bool bit_equal(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static void require_exact(const char * label, int K, unsigned pattern,
                          const float * got, const float * want) {
    for (unsigned i = 0; i < 4; ++i) {
        if (!bit_equal(got[i], want[i])) {
            std::fprintf(stderr,
                         "FAIL %s K=%d pattern=%u row=%u got=%a want=%a\n",
                         label, K, pattern, i, got[i], want[i]);
            std::abort();
        }
    }
}

static void check_case(int K, unsigned pattern,
                       const std::array<uint8_t, 64> & lut) {
    const int nb = K / 128;
    std::vector<block_q1_0x4> b;
    std::vector<block_q8_0> q8;
    pack_fixture(K, pattern, b, q8);
    const auto b_before = b;
    const auto q8_before = q8;
    float oracle[4] = {}, native[4] = {}, got[4] = {};
    scalar_native_group(oracle, b.data(), q8.data(), nb);
    native_simd_group(native, b.data(), q8.data(), nb);
    e040_q1_group_wholek(got, b.data(), q8.data(), lut.data(), static_cast<uint32_t>(nb));
    require_exact("oracle", K, pattern, got, oracle);
    require_exact("native", K, pattern, got, native);
    if (std::memcmp(b.data(), b_before.data(), b.size() * sizeof(b[0])) != 0 ||
        std::memcmp(q8.data(), q8_before.data(), q8.size() * sizeof(q8[0])) != 0) {
        std::fprintf(stderr, "FAIL mutation K=%d pattern=%u\n", K, pattern);
        std::abort();
    }
}

static void exhaustive_unpack(const std::array<uint8_t, 64> & lut) {
    std::vector<block_q1_0x4> b;
    std::vector<block_q8_0> q8;
    pack_fixture(128, 5, b, q8);
    const block_q1_0x4 pristine = b[0];
    const block_q8_0 q8_block = q8[0];
    for (unsigned byte_offset = 0; byte_offset < 64; ++byte_offset) {
        for (unsigned value = 0; value < 256; ++value) {
            block_q1_0x4 fixture = pristine;
            for (unsigned i = 0; i < 64; ++i) {
                fixture.qs[i] = 0;
            }
            fixture.qs[byte_offset] = static_cast<int8_t>(value);
            float oracle[4] = {}, native[4] = {}, got[4] = {};
            scalar_native_group(oracle, &fixture, &q8_block, 1);
            native_simd_group(native, &fixture, &q8_block, 1);
            e040_q1_group_wholek(got, &fixture, &q8_block, lut.data(), 1);
            require_exact("exhaustive-unpack-oracle", 128, value, got, oracle);
            require_exact("exhaustive-unpack-native", 128, value, got, native);
        }
    }
    std::puts("PASS exhaustive packed-Q1 byte values=256 offsets=64");
}

static constexpr uint64_t kCanaryA = 0x13579bdf2468ace0ULL;
static constexpr uint64_t kCanaryB = 0xfdb97531eca86420ULL;

struct guarded_dst {
    uint64_t pre[2];
    float data[4];
    uint64_t post[2];
};

struct guarded_q1 {
    uint64_t pre[2];
    block_q1_0x4 data;
    uint64_t post[2];
};

struct guarded_q8 {
    uint64_t pre[2];
    block_q8_0 data[4];
    uint64_t post[2];
};

struct guarded_lut {
    uint64_t pre[2];
    alignas(16) uint8_t data[64];
    uint64_t post[2];
};

static void init_canary(uint64_t (&zone)[2]) {
    zone[0] = kCanaryA;
    zone[1] = kCanaryB;
}

static void assert_canary(const char * label, const uint64_t (&zone)[2]) {
    if (zone[0] != kCanaryA || zone[1] != kCanaryB) {
        std::fprintf(stderr, "FAIL red-zone overwrite: %s\n", label);
        std::abort();
    }
}

static void early_exit_checks() {
    guarded_dst dst{};
    init_canary(dst.pre);
    init_canary(dst.post);
    for (float & value : dst.data) {
        value = 17.0f;
    }

    // These calls must not dereference any input pointer.  Passing null makes
    // an accidental read an immediate, reproducible failure on the target.
    e040_q1_group_wholek(dst.data, nullptr, nullptr, nullptr, 0);
    for (float value : dst.data) {
        if (!bit_equal(value, 0.0f)) {
            std::fprintf(stderr, "FAIL nb=0 nonzero output: %a\n", value);
            std::abort();
        }
    }
    assert_canary("dst nb=0", dst.pre);
    assert_canary("dst nb=0", dst.post);

    for (float & value : dst.data) {
        value = -19.0f;
    }
    e040_q1_group_wholek(dst.data, nullptr, nullptr, nullptr, 0x80000000u);
    for (float value : dst.data) {
        if (!bit_equal(value, 0.0f)) {
            std::fprintf(stderr, "FAIL high-bit nb nonzero output: %a\n", value);
            std::abort();
        }
    }
    assert_canary("dst high-bit nb", dst.pre);
    assert_canary("dst high-bit nb", dst.post);
    std::puts("PASS early-exit nb=0/high-bit with null inputs");
}

static void red_zone_checks(const std::array<uint8_t, 64> & lut) {
    std::vector<block_q1_0x4> b;
    std::vector<block_q8_0> q8;
    pack_fixture(128, 5, b, q8);

    guarded_dst dst{};
    guarded_q1 b_guard{};
    guarded_q8 q8_guard{};
    guarded_lut lut_guard{};
    init_canary(dst.pre);
    init_canary(dst.post);
    init_canary(b_guard.pre);
    init_canary(b_guard.post);
    init_canary(q8_guard.pre);
    init_canary(q8_guard.post);
    init_canary(lut_guard.pre);
    init_canary(lut_guard.post);
    std::memcpy(&b_guard.data, b.data(), sizeof(b_guard.data));
    std::memcpy(q8_guard.data, q8.data(), sizeof(q8_guard.data));
    std::memcpy(lut_guard.data, lut.data(), sizeof(lut_guard.data));

    float oracle[4] = {};
    scalar_native_group(oracle, &b_guard.data, q8_guard.data, 1);
    e040_q1_group_wholek(dst.data, &b_guard.data, q8_guard.data,
                         lut_guard.data, 1);
    require_exact("red-zone", 128, 5, dst.data, oracle);
    assert_canary("dst pre", dst.pre);
    assert_canary("dst post", dst.post);
    assert_canary("q1 pre", b_guard.pre);
    assert_canary("q1 post", b_guard.post);
    assert_canary("q8 pre", q8_guard.pre);
    assert_canary("q8 post", q8_guard.post);
    assert_canary("lut pre", lut_guard.pre);
    assert_canary("lut post", lut_guard.post);
    if (std::memcmp(&b_guard.data, b.data(), sizeof(b_guard.data)) != 0 ||
        std::memcmp(q8_guard.data, q8.data(), sizeof(q8_guard.data)) != 0 ||
        std::memcmp(lut_guard.data, lut.data(), sizeof(lut_guard.data)) != 0) {
        std::fprintf(stderr, "FAIL input mutation in red-zone case\n");
        std::abort();
    }
    std::puts("PASS red-zone/canary dst/q1/q8/lut nb=1");
}

static volatile float g_sink = 0.0f;

static uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
}

static double bench_e040(float * out, const block_q1_0x4 * b,
                         const block_q8_0 * q8, const uint8_t * lut,
                         int nb, uint64_t budget_ns, uint64_t & calls) {
    calls = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        e040_q1_group_wholek(out, b, q8, lut, static_cast<uint32_t>(nb));
        g_sink += out[0];
        ++calls;
    } while (elapsed_ns(start) < budget_ns);
    return static_cast<double>(elapsed_ns(start)) / calls;
}

static double bench_native(float * out, const block_q1_0x4 * b,
                           const block_q8_0 * q8, int nb,
                           uint64_t budget_ns, uint64_t & calls) {
    calls = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        native_simd_group(out, b, q8, nb);
        g_sink += out[0];
        ++calls;
    } while (elapsed_ns(start) < budget_ns);
    return static_cast<double>(elapsed_ns(start)) / calls;
}

static void timed_gate(int K, const std::array<uint8_t, 64> & lut) {
    const int nb = K / 128;
    std::vector<block_q1_0x4> b;
    std::vector<block_q8_0> q8;
    pack_fixture(K, 5, b, q8);
    float out[4] = {};
    for (unsigned i = 0; i < 128; ++i) {
        if (i & 1u) {
            native_simd_group(out, b.data(), q8.data(), nb);
        } else {
            e040_q1_group_wholek(out, b.data(), q8.data(), lut.data(), nb);
        }
        g_sink += out[0];
    }
    constexpr uint64_t budget_ns = 25'000'000;
    std::array<double, 4> e040{}, native{};
    for (unsigned round = 0; round < 4; ++round) {
        uint64_t ec = 0, nc = 0;
        if ((round & 1u) == 0) {
            e040[round] = bench_e040(out, b.data(), q8.data(), lut.data(), nb,
                                     budget_ns, ec);
            native[round] = bench_native(out, b.data(), q8.data(), nb,
                                          budget_ns, nc);
        } else {
            native[round] = bench_native(out, b.data(), q8.data(), nb,
                                          budget_ns, nc);
            e040[round] = bench_e040(out, b.data(), q8.data(), lut.data(), nb,
                                     budget_ns, ec);
        }
        std::printf("TIMING K=%d round=%u order=%s e040_ns=%.1f(%llu) native_ns=%.1f(%llu)\n",
                    K, round, (round & 1u) ? "native-first" : "e040-first",
                    e040[round], static_cast<unsigned long long>(ec),
                    native[round], static_cast<unsigned long long>(nc));
    }
    std::sort(e040.begin(), e040.end());
    std::sort(native.begin(), native.end());
    std::printf("TIMING_SUMMARY K=%d e040_median_ns=%.1f native_median_ns=%.1f sink=%a\n",
                K, (e040[1] + e040[2]) * 0.5,
                (native[1] + native[2]) * 0.5, static_cast<float>(g_sink));
}

int main() {
    const auto sign_table = make_sign_table();
    std::memcpy(g_table_q1_signs, sign_table.data(), sizeof(g_table_q1_signs));
    const auto lut = make_nibble_lut();
    assert(sign_table[0] == 0xffffffffffffffffULL);
    assert(sign_table[255] == 0x0101010101010101ULL);
    for (int K : {128, 256, 5120}) {
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
            check_case(K, pattern, lut);
        }
    }
    early_exit_checks();
    red_zone_checks(lut);
    exhaustive_unpack(lut);
    timed_gate(5120, lut);
    std::puts("PASS E040 register-nibble Q1 whole-K: exact/oracle/native/no-mutation");
    return 0;
}
