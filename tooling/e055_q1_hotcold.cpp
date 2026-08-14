// E055 bounded Q1 hot/cold microbenchmark for the A733.
//
// The reference math/traversal is included verbatim from E039's stock
// block_q1_0x4 4x4 NEON/DOTPROD path.  This file adds controls around it; it
// does not change the Q1 representation, quantisation, lane order, or
// accumulation order.  It deliberately does not open a model or contain a
// model payload.

#define main e055_embedded_e039_reference_main
#include "../experiments/E039-q1-pair-wholek/e039_wholek_harness.cpp"
#undef main

#include <fcntl.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {

constexpr int kMarkerFd = 9;
constexpr int kAckFd = 8;
constexpr uint64_t kDefaultBudgetMs = 250;
constexpr uint64_t kDefaultThrashBytes = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kCacheLineBytes = 64;
constexpr uint64_t kNativeCarrierBytes = 208;
constexpr uint64_t kCanarySeed = UINT64_C(0xe055a7339e039040);

volatile uint64_t g_checksum = kCanarySeed;
volatile float g_float_sink = 0.0f;

struct options {
    std::string mode = "full_dotprod";
    std::string cache_state = "hot_repeat";
    uint64_t target_bytes = 64ULL * 1024ULL;
    uint64_t iterations = 0;
    uint64_t budget_ms = kDefaultBudgetMs;
    uint64_t thrash_bytes = kDefaultThrashBytes;
    uint64_t warmup = 16;
    int cpu_label = -1;
    bool sync = false;
    bool self_test = false;
    bool iterations_explicit = false;
    bool budget_explicit = false;
    const char *json_out = nullptr;
};

static void usage(const char *program) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --mode packed_stream|unpack_scale|full_dotprod\n"
                 "  --cache-state hot_repeat|cold_conditioned\n"
                 "  --working-set-bytes N  (64K, 128K, 256K, 512K, 1M, 4M, 12.5M)\n"
                 "  --iterations N         exact calls; default uses --budget-ms\n"
                 "  --budget-ms N          hot repeated-call budget (default 250)\n"
                 "  --thrash-bytes N       cold conditioning buffer (default 64M)\n"
                 "  --warmup N             hot calls before PMU marker (default 16)\n"
                 "  --cpu N                recorded CPU label (otherwise sched_getcpu)\n"
                 "  --sync                 use fixed fd9 S / fd8 ACK / fd9 E marker\n"
                 "  --json-out PATH        duplicate the final JSON object to PATH\n"
                 "  --self-test            run only the exact stock golden cases\n"
                 "  --help\n",
                 program);
}

static bool parse_u64(const char *text, uint64_t &value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

static bool parse_int(const char *text, int &value) {
    uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed > static_cast<uint64_t>(INT_MAX)) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_options(int argc, char **argv, options &out) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return false;
        }
        if (std::strcmp(arg, "--self-test") == 0) {
            out.self_test = true;
            continue;
        }
        if (std::strcmp(arg, "--sync") == 0) {
            out.sync = true;
            continue;
        }
        const char *value = nullptr;
        if (std::strcmp(arg, "--mode") == 0) {
            value = require_value(arg);
            if (value == nullptr) return false;
            out.mode = value;
        } else if (std::strcmp(arg, "--cache-state") == 0) {
            value = require_value(arg);
            if (value == nullptr) return false;
            out.cache_state = value;
        } else if (std::strcmp(arg, "--working-set-bytes") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_u64(value, out.target_bytes)) return false;
        } else if (std::strcmp(arg, "--iterations") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_u64(value, out.iterations)) return false;
            out.iterations_explicit = true;
        } else if (std::strcmp(arg, "--budget-ms") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_u64(value, out.budget_ms)) return false;
            out.budget_explicit = true;
        } else if (std::strcmp(arg, "--thrash-bytes") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_u64(value, out.thrash_bytes)) return false;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_u64(value, out.warmup)) return false;
        } else if (std::strcmp(arg, "--cpu") == 0) {
            value = require_value(arg);
            if (value == nullptr || !parse_int(value, out.cpu_label)) return false;
        } else if (std::strcmp(arg, "--json-out") == 0) {
            value = require_value(arg);
            if (value == nullptr) return false;
            out.json_out = value;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            usage(argv[0]);
            return false;
        }
    }
    if (out.mode != "packed_stream" && out.mode != "unpack_scale" &&
        out.mode != "full_dotprod") {
        std::fprintf(stderr, "invalid --mode: %s\n", out.mode.c_str());
        return false;
    }
    if (out.cache_state != "hot_repeat" && out.cache_state != "cold_conditioned") {
        std::fprintf(stderr, "invalid --cache-state: %s\n", out.cache_state.c_str());
        return false;
    }
    if (out.target_bytes == 0 || out.thrash_bytes < kCacheLineBytes) {
        std::fprintf(stderr, "working set and thrash buffer must be positive\n");
        return false;
    }
    if (out.target_bytes > std::numeric_limits<uint64_t>::max() -
                               (kNativeCarrierBytes - 1)) {
        std::fprintf(stderr, "working set rounding overflow\n");
        return false;
    }
    if (out.thrash_bytes % kCacheLineBytes != 0) {
        std::fprintf(stderr, "thrash buffer must be 64-byte aligned\n");
        return false;
    }
    if (out.cache_state == "cold_conditioned") {
        // A cold sample is one explicitly conditioned traversal.  Repeating
        // after it would turn the rest of the PMU window into a hot sample.
        if ((out.iterations_explicit && out.iterations != 1) ||
            (out.budget_explicit && out.budget_ms != 0)) {
            std::fprintf(stderr,
                         "cold_conditioned requires --iterations 1 and --budget-ms 0\n");
            return false;
        }
        out.iterations = 1;
        out.budget_ms = 0;
        out.warmup = 0;
    }
    return true;
}

static uint64_t monotonic_ns() {
    struct timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * UINT64_C(1000000000) +
           static_cast<uint64_t>(ts.tv_nsec);
}

static uint64_t mix_checksum(uint64_t hash, uint64_t value) {
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    hash ^= hash >> 29;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 31;
    return hash;
}

// The following function is deliberately a memory-only control.  It walks
// Q1/Q8 in the stock block order.  Four bounded vector accumulators consume
// the loads; only one final reduction and one global sink update are observable.
// Thus the control has no serial per-byte hash, although vector XOR/add and the
// final reduction remain measurable control overhead.
__attribute__((noinline, noclone, used))
static uint64_t packed_stream_only(const block_q1_0x4 *b,
                                   const block_q8_0 *q8, int nb) {
    uint8x16_t acc0 = vdupq_n_u8(0x15);
    uint8x16_t acc1 = vdupq_n_u8(0x2a);
    uint8x16_t acc2 = vdupq_n_u8(0x51);
    uint8x16_t acc3 = vdupq_n_u8(0xa2);
    uint64_t scale_bits = kCanarySeed;
    for (int l = 0; l < nb; ++l) {
        uint64_t q1_scale_bits = 0;
        std::memcpy(&q1_scale_bits, b[l].d, sizeof(q1_scale_bits));
        scale_bits ^= q1_scale_bits;
        const uint8_t *b_qs = reinterpret_cast<const uint8_t *>(b[l].qs);
        for (int k = 0; k < 4; ++k) {
            const block_q8_0 *a = q8 + l * 4 + k;
            uint16_t q8_scale_bits = 0;
            std::memcpy(&q8_scale_bits, &a->d, sizeof(q8_scale_bits));
            scale_bits += q8_scale_bits;
            acc0 = veorq_u8(acc0, vld1q_u8(b_qs + k * 16));
            acc1 = vaddq_u8(acc1, vld1q_u8(reinterpret_cast<const uint8_t *>(a->qs)));
            acc2 = veorq_u8(acc2, vld1q_u8(reinterpret_cast<const uint8_t *>(a->qs + 16)));
            acc3 = vaddq_u8(acc3, veorq_u8(acc0, acc2));
        }
    }
    const uint64_t reduced = static_cast<uint64_t>(vaddvq_u8(acc0)) |
                             (static_cast<uint64_t>(vaddvq_u8(acc1)) << 8) |
                             (static_cast<uint64_t>(vaddvq_u8(acc2)) << 16) |
                             (static_cast<uint64_t>(vaddvq_u8(acc3)) << 24);
    const uint64_t hash = mix_checksum(scale_bits, reduced);
    g_checksum = mix_checksum(g_checksum, hash);
    return hash;
}

// Register unpack and scale control.  It executes the same two-byte packed
// sign loads and LUT expansion as the stock path, loads every Q8 vector and
// both scale streams, but replaces SDOT/FMA with four bounded vector consumers.
// There is one final vector reduction and one externally observable sink.
// PMU values are event counts; values joined by the runner remain counts,
// never bytes.
__attribute__((noinline, noclone, used))
static uint64_t unpack_scale_only(const block_q1_0x4 *b,
                                  const block_q8_0 *q8, int nb) {
    int8x16_t sign_acc0 = vdupq_n_s8(1);
    int8x16_t sign_acc1 = vdupq_n_s8(3);
    int8x16_t q8_acc0 = vdupq_n_s8(5);
    int8x16_t q8_acc1 = vdupq_n_s8(7);
    float32x4_t scale_acc = vdupq_n_f32(0.0f);
    float q8_scale_acc = 0.0f;
    for (int l = 0; l < nb; ++l) {
        const uint8_t *b_qs = reinterpret_cast<const uint8_t *>(b[l].qs);
        for (int k = 0; k < 4; ++k) {
            const block_q8_0 *a_blk = q8 + l * 4 + k;
            q8_scale_acc += neon_fp16_to_float(a_blk->d);
            for (int tile = 0; tile < 8; tile += 4) {
                const int8x16_t s0 = e039_q1_unpack_pair(
                    b_qs[k * 16 + 2 * (tile + 0) + 0],
                    b_qs[k * 16 + 2 * (tile + 0) + 1]);
                const int8x16_t s1 = e039_q1_unpack_pair(
                    b_qs[k * 16 + 2 * (tile + 1) + 0],
                    b_qs[k * 16 + 2 * (tile + 1) + 1]);
                const int8x16_t s2 = e039_q1_unpack_pair(
                    b_qs[k * 16 + 2 * (tile + 2) + 0],
                    b_qs[k * 16 + 2 * (tile + 2) + 1]);
                const int8x16_t s3 = e039_q1_unpack_pair(
                    b_qs[k * 16 + 2 * (tile + 3) + 0],
                    b_qs[k * 16 + 2 * (tile + 3) + 1]);
                const int8x16_t q = vld1q_s8(a_blk->qs + tile * 4);
                sign_acc0 = veorq_s8(sign_acc0, veorq_s8(s0, s2));
                sign_acc1 = vaddq_s8(sign_acc1, veorq_s8(s1, s3));
                if (tile == 0) {
                    q8_acc0 = veorq_s8(q8_acc0, q);
                } else {
                    q8_acc1 = vaddq_s8(q8_acc1, q);
                }
            }
        }
        scale_acc = vaddq_f32(scale_acc, vcvt_f32_f16(
            vld1_f16(reinterpret_cast<const float16_t *>(b[l].d))));
    }
    const uint8x16_t folded = veorq_u8(
        vreinterpretq_u8_s8(veorq_s8(sign_acc0, sign_acc1)),
        vreinterpretq_u8_s8(veorq_s8(q8_acc0, q8_acc1)));
    const float scale_sum = vaddvq_f32(scale_acc) + q8_scale_acc;
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale_sum, sizeof(scale_bits));
    const uint64_t reduced = static_cast<uint64_t>(vaddvq_u8(folded)) |
                             (static_cast<uint64_t>(scale_bits) << 16);
    const uint64_t hash = mix_checksum(kCanarySeed, reduced);
    g_checksum = mix_checksum(g_checksum, hash);
    return hash;
}

static uint64_t run_one(const std::string &mode,
                        const block_q1_0x4 *b,
                        const block_q8_0 *q8,
                        int nb) {
    if (mode == "packed_stream") {
        return packed_stream_only(b, q8, nb);
    }
    if (mode == "unpack_scale") {
        return unpack_scale_only(b, q8, nb);
    }
    float out[4] = {};
    native_simd_group(out, b, q8, nb);
    uint64_t hash = kCanarySeed;
    for (float value : out) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash = mix_checksum(hash, bits);
    }
    g_float_sink += out[0];
    g_checksum = mix_checksum(g_checksum, hash);
    return hash;
}

static bool bit_equal(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static bool run_golden(unsigned &case_count) {
    case_count = 0;
    for (int K : {128, 256, 5120}) {
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
            std::vector<block_q1_0x4> b0, b1;
            std::vector<block_q8_0> q8;
            pack_fixture(K, pattern, b0, b1, q8);
            const std::vector<block_q1_0x4> before_b = b0;
            const std::vector<block_q8_0> before_q8 = q8;
            float scalar[4] = {};
            float simd[4] = {};
            scalar_native_group(scalar, b0.data(), q8.data(), K / 128);
            native_simd_group(simd, b0.data(), q8.data(), K / 128);
            for (unsigned row = 0; row < 4; ++row) {
                if (!bit_equal(scalar[row], simd[row])) {
                    std::fprintf(stderr,
                                 "golden mismatch K=%d pattern=%u row=%u scalar=%a simd=%a\n",
                                 K, pattern, row, scalar[row], simd[row]);
                    return false;
                }
            }
            if (std::memcmp(b0.data(), before_b.data(), before_b.size() * sizeof(b0[0])) != 0 ||
                std::memcmp(q8.data(), before_q8.data(), before_q8.size() * sizeof(q8[0])) != 0) {
                std::fprintf(stderr, "golden input mutation K=%d pattern=%u\n", K, pattern);
                return false;
            }
            ++case_count;
        }
    }
    return true;
}

struct cold_result {
    uint64_t requested_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t lines = 0;
    uint64_t warmup_calls = 0;
    uint64_t checksum = 0;
    bool verified_touched = false;
};

static cold_result condition_cold(std::vector<uint8_t> &buffer) {
    cold_result result;
    result.requested_bytes = static_cast<uint64_t>(buffer.size());
    result.actual_bytes = static_cast<uint64_t>(buffer.size());
    result.lines = result.actual_bytes / kCacheLineBytes;
    uint64_t checksum = kCanarySeed;
    for (uint64_t line = 0; line < result.lines; ++line) {
        uint8_t *base = buffer.data() + line * kCacheLineBytes;
        for (uint64_t word = 0; word < kCacheLineBytes; word += sizeof(uint64_t)) {
            uint64_t value = kCanarySeed ^ (line * UINT64_C(0x9e3779b1)) ^ word;
            std::memcpy(base + word, &value, sizeof(value));
        }
    }
    for (uint64_t line = 0; line < result.lines; ++line) {
        const uint8_t *base = buffer.data() + line * kCacheLineBytes;
        for (uint64_t word = 0; word < kCacheLineBytes; word += sizeof(uint64_t)) {
            uint64_t value = 0;
            std::memcpy(&value, base + word, sizeof(value));
            const uint64_t expected = kCanarySeed ^ (line * UINT64_C(0x9e3779b1)) ^ word;
            if (value != expected) {
                return result;
            }
            checksum = mix_checksum(checksum, value);
        }
    }
    result.checksum = checksum;
    result.verified_touched = true;
    return result;
}

static bool fd_is_open(int fd) {
    return fcntl(fd, F_GETFD) >= 0;
}

static bool marker_start(bool enabled) {
    if (!enabled) return true;
    if (!fd_is_open(kMarkerFd) || !fd_is_open(kAckFd)) {
        std::fprintf(stderr, "sync requested but fd9/fd8 are unavailable\n");
        return false;
    }
    const char start = 'S';
    if (write(kMarkerFd, &start, 1) != 1) {
        std::perror("write marker S");
        return false;
    }
    char ack = 0;
    if (read(kAckFd, &ack, 1) != 1 || ack != 'A') {
        std::fprintf(stderr, "invalid PMU ACK marker\n");
        return false;
    }
    return true;
}

static bool marker_end(bool enabled) {
    if (!enabled) return true;
    const char end = 'E';
    if (write(kMarkerFd, &end, 1) != 1) {
        std::perror("write marker E");
        return false;
    }
    return true;
}

struct logical_byte_counts {
    uint64_t q1_packed_bytes;
    uint64_t q8_bytes;
    uint64_t total_input_bytes;
    uint64_t output_bytes;
    uint64_t dot_products;
};

static logical_byte_counts logical_bytes_for_blocks(int blocks) {
    const uint64_t count = static_cast<uint64_t>(blocks);
    return {
        count * 72,
        count * 4 * 34,
        count * 208,
        16,
        count * 512,
    };
}

static std::string json_result(const options &opt,
                               int cpu,
                               unsigned golden_cases,
                               const cold_result &cold,
                               uint64_t actual_bytes,
                               int blocks,
                               uint64_t iterations,
                               uint64_t calls,
                               uint64_t elapsed_ns,
                               uint64_t first_call_ns,
                               uint64_t checksum,
                               bool marker_started,
                               bool marker_ack,
                               bool marker_ended) {
    const auto bytes = logical_bytes_for_blocks(blocks);
    const bool hot_conditioned = opt.cache_state == "hot_repeat" && opt.warmup > 0;
    const bool conditioning_verified = hot_conditioned || cold.verified_touched;
    const char *conditioning_strategy = hot_conditioned
        ? "verified_kernel_warmup" : "verified_write_read_each_64B_line";
    char buffer[8192] = {};
    const double calls_per_second = elapsed_ns == 0
        ? 0.0 : static_cast<double>(calls) * 1.0e9 / static_cast<double>(elapsed_ns);
    const int written = std::snprintf(
        buffer, sizeof(buffer),
        "{\"schema\":\"e055-q1-hot-cold-harness/v1\","
        "\"mode\":\"%s\",\"cache_state\":\"%s\",\"cpu\":%d,"
        "\"q1_layout\":\"E039 stock native block_q1_0x4 4x4 DOTPROD\","
        "\"golden_pass\":true,\"golden_cases\":%u,"
        "\"target_working_set_bytes\":%" PRIu64 ","
        "\"actual_working_set_bytes\":%" PRIu64 ",\"blocks\":%d,"
        "\"iterations\":%" PRIu64 ",\"calls\":%" PRIu64 ","
        "\"elapsed_ns\":%" PRIu64 ",\"first_call_ns\":%" PRIu64 ","
        "\"calls_per_second\":%.9f,"
        "\"logical_bytes_per_call\":{\"q1_packed_bytes\":%" PRIu64 ","
        "\"q8_bytes\":%" PRIu64 ",\"total_input_bytes\":%" PRIu64 ","
        "\"output_bytes\":%" PRIu64 ",\"dot_products\":%" PRIu64 "},"
        "\"checksum\":\"0x%" PRIx64 "\","
        "\"cold_conditioning\":{\"strategy\":\"%s\","
        "\"requested_bytes\":%" PRIu64 ",\"actual_bytes\":%" PRIu64 ","
        "\"line_bytes\":%" PRIu64 ","
        "\"lines_touched\":%" PRIu64 ",\"checksum\":\"0x%" PRIx64 "\","
        "\"verified_touched\":%s,\"warmup_calls\":%" PRIu64 "},"
        "\"sync\":{\"requested\":%s,\"started\":%s,"
        "\"acknowledged\":%s,\"ended\":%s,\"sequence\":\"S/A/E\"},"
        "\"qualification\":\"unqualified_harness_output_requires_E049c_join\"}\n",
        opt.mode.c_str(), opt.cache_state.c_str(), cpu,
        golden_cases, opt.target_bytes, actual_bytes, blocks,
        iterations, calls, elapsed_ns, first_call_ns, calls_per_second,
        bytes.q1_packed_bytes, bytes.q8_bytes, bytes.total_input_bytes,
        bytes.output_bytes, bytes.dot_products, checksum, conditioning_strategy,
        cold.requested_bytes, cold.actual_bytes, kCacheLineBytes, cold.lines,
        cold.checksum, conditioning_verified ? "true" : "false", cold.warmup_calls,
        opt.sync ? "true" : "false", marker_started ? "true" : "false",
        marker_ack ? "true" : "false", marker_ended ? "true" : "false");
    if (written < 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
        return "{\"schema\":\"e055-q1-hot-cold-harness/v1\",\"error\":\"json buffer overflow\"}\n";
    }
    return std::string(buffer, static_cast<size_t>(written));
}

}  // namespace

int main(int argc, char **argv) {
    options opt;
    if (!parse_options(argc, argv, opt)) {
        return 2;
    }
    const auto sign_table = make_table();
    std::memcpy(g_table_q1_signs, sign_table.data(), sizeof(g_table_q1_signs));
    unsigned golden_cases = 0;
    if (!run_golden(golden_cases)) {
        return 3;
    }
    if (opt.self_test) {
        std::printf("{\"schema\":\"e055-q1-hot-cold-harness/v1\",\"self_test\":true,"
                    "\"golden_pass\":true,\"golden_cases\":%u}\n", golden_cases);
        return 0;
    }

    const uint64_t blocks_u64 =
        std::max<uint64_t>(1, (opt.target_bytes + kNativeCarrierBytes - 1) /
                              kNativeCarrierBytes);
    if (blocks_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        blocks_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max() / 128)) {
        std::fprintf(stderr, "working set has too many native blocks\n");
        return 2;
    }
    const int blocks = static_cast<int>(blocks_u64);
    const uint64_t actual_bytes = blocks_u64 * kNativeCarrierBytes;
    std::vector<block_q1_0x4> b0, b1;
    std::vector<block_q8_0> q8;
    pack_fixture(blocks * 128, 5, b0, b1, q8);
    std::vector<uint8_t> thrash;
    cold_result cold;
    if (opt.cache_state == "cold_conditioned") {
        try {
            thrash.resize(static_cast<size_t>(opt.thrash_bytes));
        } catch (...) {
            std::fprintf(stderr, "cannot allocate thrash buffer\n");
            return 4;
        }
        cold = condition_cold(thrash);
        if (!cold.verified_touched) {
            std::fprintf(stderr, "thrash verification failed\n");
            return 5;
        }
    }

    if (opt.cache_state == "hot_repeat") {
        cold.requested_bytes = actual_bytes;
        cold.actual_bytes = actual_bytes;
        cold.lines = (actual_bytes + kCacheLineBytes - 1) / kCacheLineBytes;
        cold.warmup_calls = opt.warmup;
        cold.checksum = kCanarySeed;
        for (uint64_t i = 0; i < opt.warmup; ++i) {
            cold.checksum = mix_checksum(
                cold.checksum, run_one(opt.mode, b0.data(), q8.data(), blocks));
        }
        cold.verified_touched = opt.warmup > 0;
    }

    const bool marker_requested = opt.sync;
    bool marker_started = false;
    bool marker_ack = false;
    bool marker_ended = false;
    if (marker_requested) {
        marker_started = true;
        if (!marker_start(true)) {
            return 6;
        }
        marker_ack = true;
    }

    uint64_t calls = 0;
    uint64_t elapsed_ns = 0;
    uint64_t first_call_ns = 0;
    uint64_t checksum = kCanarySeed;
    if (opt.iterations > 0) {
        const uint64_t start = monotonic_ns();
        for (uint64_t i = 0; i < opt.iterations; ++i) {
            const uint64_t call_start = monotonic_ns();
            checksum = mix_checksum(checksum,
                                    run_one(opt.mode, b0.data(), q8.data(), blocks));
            const uint64_t call_end = monotonic_ns();
            if (calls == 0) first_call_ns = call_end - call_start;
            ++calls;
        }
        elapsed_ns = monotonic_ns() - start;
    } else {
        const uint64_t start = monotonic_ns();
        do {
            checksum = mix_checksum(checksum,
                                    run_one(opt.mode, b0.data(), q8.data(), blocks));
            ++calls;
        } while (monotonic_ns() - start < opt.budget_ms * UINT64_C(1000000));
        elapsed_ns = monotonic_ns() - start;
        first_call_ns = elapsed_ns / calls;
    }
    if (marker_requested) {
        marker_ended = marker_end(true);
        if (!marker_ended) return 7;
    }

    const int detected_cpu = opt.cpu_label >= 0 ? opt.cpu_label : sched_getcpu();
    const uint64_t published_checksum = mix_checksum(
        checksum, static_cast<uint64_t>(g_checksum));
    const std::string json = json_result(
        opt, detected_cpu, golden_cases, cold, actual_bytes, blocks,
        opt.iterations > 0 ? opt.iterations : calls, calls, elapsed_ns,
        first_call_ns, published_checksum, marker_started, marker_ack,
        marker_ended);
    std::fputs(json.c_str(), stdout);
    if (opt.json_out != nullptr) {
        FILE *file = std::fopen(opt.json_out, "wb");
        if (file == nullptr || std::fwrite(json.data(), 1, json.size(), file) != json.size()) {
            if (file != nullptr) std::fclose(file);
            std::fprintf(stderr, "cannot write --json-out\n");
            return 8;
        }
        if (std::fclose(file) != 0) {
            std::fprintf(stderr, "cannot close --json-out\n");
            return 8;
        }
    }
    return 0;
}
