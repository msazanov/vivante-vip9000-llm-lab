#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef Q1_STRICT_MODE
#define Q1_STRICT_MODE 1
#endif

namespace fs = std::filesystem;

namespace {
constexpr const char * kSchema = "q1-cpu-operator-run/v1";

struct Options {
    std::string run_id;
    fs::path weights;
    fs::path activation;
    fs::path output_jsonl;
    fs::path output_f32;
    fs::path output_q8;
    fs::path fixture_dir;
    fs::path output_dir;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int threads = 1;
    int warmup = 1;
    int iterations = 50;
    bool self_test = false;
    bool help = false;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void usage(FILE * stream) {
    std::fprintf(stream,
        "q1_cpu_operator_runner --run-id ID --weights weights.q1_0.bin --activation activation.f32.bin\n"
        "  --ne0 K --ne1 M --threads N --warmup 1 --iterations 50\n"
        "  --output-jsonl runner.jsonl --output-f32 output.f32.bin\n"
        "  --output-q8 activation.q8_0.bin\n"
        "Self-test: --self-test --fixture-dir DIR --iterations N --output-dir DIR\n");
}

const char * next_arg(int & index, int argc, char ** argv, const char * option) {
    if (++index >= argc) {
        fail(std::string("missing value for ") + option);
    }
    return argv[index];
}

int parse_int(const char * text, const char * option, int minimum) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum || value > 100000000) {
        fail(std::string("invalid ") + option);
    }
    return static_cast<int>(value);
}

bool valid_run_id(const std::string & value) {
    if (value.empty() || !(std::isalnum(static_cast<unsigned char>(value.front())))) return false;
    for (const unsigned char character : value) {
        if (!(std::isalnum(character) || character == '.' || character == '_' || character == '-')) return false;
    }
    return true;
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            options.help = true;
        } else if (option == "--run-id") {
            options.run_id = next_arg(index, argc, argv, "--run-id");
        } else if (option == "--self-test") {
            options.self_test = true;
        } else if (option == "--weights") {
            options.weights = next_arg(index, argc, argv, "--weights");
        } else if (option == "--activation") {
            options.activation = next_arg(index, argc, argv, "--activation");
        } else if (option == "--fixture-dir") {
            options.fixture_dir = next_arg(index, argc, argv, "--fixture-dir");
        } else if (option == "--output-dir") {
            options.output_dir = next_arg(index, argc, argv, "--output-dir");
        } else if (option == "--output-jsonl") {
            options.output_jsonl = next_arg(index, argc, argv, "--output-jsonl");
        } else if (option == "--output-f32") {
            options.output_f32 = next_arg(index, argc, argv, "--output-f32");
        } else if (option == "--output-q8") {
            options.output_q8 = next_arg(index, argc, argv, "--output-q8");
        } else if (option == "--ne0") {
            options.ne0 = parse_int(next_arg(index, argc, argv, "--ne0"), "--ne0", 1);
        } else if (option == "--ne1") {
            options.ne1 = parse_int(next_arg(index, argc, argv, "--ne1"), "--ne1", 1);
        } else if (option == "--threads") {
            options.threads = parse_int(next_arg(index, argc, argv, "--threads"), "--threads", 1);
        } else if (option == "--warmup") {
            options.warmup = parse_int(next_arg(index, argc, argv, "--warmup"), "--warmup", 0);
        } else if (option == "--iterations") {
            options.iterations = parse_int(next_arg(index, argc, argv, "--iterations"), "--iterations", 1);
        } else {
            fail("unknown option: " + option);
        }
    }
    if (options.help) {
        return options;
    }
    if (options.self_test) {
        if (options.fixture_dir.empty() || options.output_dir.empty()) {
            fail("--self-test requires --fixture-dir and --output-dir");
        }
        options.weights = options.fixture_dir / "weights.q1_0.bin";
        options.activation = options.fixture_dir / "activation.f32.bin";
        options.output_dir = fs::absolute(options.output_dir);
        options.output_jsonl = options.output_dir / "runner.jsonl";
        options.output_f32 = options.output_dir / "output.f32.bin";
        options.output_q8 = options.output_dir / "activation.q8_0.bin";
        if (options.ne0 == 0 && options.ne1 == 0) {
            options.ne0 = 128;
            options.ne1 = 16;
        }
        if (options.run_id.empty()) options.run_id = "q1-self-test-" + std::to_string(static_cast<long long>(::getpid()));
    } else if (options.weights.empty() || options.activation.empty() || options.output_jsonl.empty() ||
               options.output_f32.empty() || options.output_q8.empty() || options.ne0 <= 0 || options.ne1 <= 0) {
        fail("normal mode requires weights, activation, ne0, ne1, and all output paths");
    }
    if (!valid_run_id(options.run_id)) fail("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]*");
    if (options.ne0 % 128 != 0 || options.ne1 % 16 != 0) {
        fail("Q1_0 dimensions must be ne0 divisible by 128 and ne1 divisible by 16");
    }
    if (options.self_test && (options.ne0 != 128 || options.ne1 != 16)) {
        fail("--self-test requires the literal K=128, M=16 fixture");
    }
    if (options.self_test && options.warmup != 1) {
        fail("--self-test requires --warmup 1");
    }
    if (!options.self_test && (options.warmup != 1 || options.iterations != 50)) {
        fail("normal mode requires --warmup 1 and --iterations 50");
    }
    return options;
}

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) fail("cannot open " + path.string());
    const auto size = stream.tellg();
    if (size < 0) fail("cannot stat " + path.string());
    std::vector<uint8_t> data(static_cast<size_t>(size));
    stream.seekg(0);
    if (!data.empty()) stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) fail("cannot read " + path.string());
    return data;
}

void write_fd_all(int descriptor, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(descriptor, bytes + offset, size - offset);
        if (written <= 0) fail("short output write");
        offset += static_cast<size_t>(written);
    }
}

struct OutputFiles {
    std::array<fs::path, 3> finals;
    std::array<fs::path, 3> temps;
    std::array<bool, 3> published{{false, false, false}};
    bool committed = false;
    static inline uint64_t serial = 0;

    explicit OutputFiles(const Options & options) : finals{{options.output_jsonl, options.output_f32, options.output_q8}} {
        try {
            for (size_t i = 0; i < finals.size(); ++i) {
                for (size_t j = 0; j < i; ++j) if (finals[i] == finals[j]) fail("output paths must be distinct");
                if (fs::exists(finals[i])) fail("refusing to overwrite existing output: " + finals[i].string());
                if (!finals[i].parent_path().empty()) fs::create_directories(finals[i].parent_path());
                temps[i] = finals[i].parent_path() / ("." + finals[i].filename().string() + ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." + std::to_string(++serial));
                const int descriptor = ::open(temps[i].c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                if (descriptor < 0) fail("cannot create temporary output: " + temps[i].string());
                ::close(descriptor);
            }
        } catch (...) {
            for (const fs::path & temp : temps) if (!temp.empty()) ::unlink(temp.c_str());
            throw;
        }
    }
    void write(size_t index, const void * data, size_t size) {
        const int descriptor = ::open(temps[index].c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC);
        if (descriptor < 0) fail("cannot open temporary output");
        try { write_fd_all(descriptor, data, size); if (::fsync(descriptor) != 0) fail("cannot sync temporary output"); }
        catch (...) { ::close(descriptor); throw; }
        ::close(descriptor);
    }
    void publish() {
        for (size_t i = 0; i < finals.size(); ++i) {
            if (::link(temps[i].c_str(), finals[i].c_str()) != 0) fail("refusing to overwrite output: " + finals[i].string());
            published[i] = true;
            if (::unlink(temps[i].c_str()) != 0) fail("cannot publish output: " + finals[i].string());
        }
        committed = true;
    }
    ~OutputFiles() {
        for (size_t i = 0; i < finals.size(); ++i) {
            if (!committed && published[i]) ::unlink(finals[i].c_str());
            if (!temps[i].empty()) ::unlink(temps[i].c_str());
        }
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string & text) : text_(text) {}

    void object(const std::function<void(const std::string &)> & field) {
        ws(); expect('{'); ws();
        if (take('}')) return;
        while (true) {
            const std::string key = string(); ws(); expect(':'); field(key); ws();
            if (take('}')) return;
            expect(','); ws();
        }
    }

    std::string string() {
        ws(); expect('"'); std::string result;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return result;
            if (c == '\\') {
                if (pos_ >= text_.size()) fail("fixture JSON has truncated escape");
                const char escaped = text_[pos_++];
                switch (escaped) {
                    case '"': case '\\': case '/': result.push_back(escaped); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    default: fail("fixture JSON has unsupported string escape");
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                fail("fixture JSON has control character in string");
            } else result.push_back(c);
        }
        fail("fixture JSON has unterminated string");
    }

    int64_t integer() {
        ws(); const size_t begin = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        const size_t digits = pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (digits == pos_) fail("fixture JSON expected integer");
        try { return std::stoll(text_.substr(begin, pos_ - begin)); }
        catch (...) { fail("fixture JSON integer is out of range"); }
    }

    std::vector<int64_t> integer_array() {
        ws(); expect('['); ws(); std::vector<int64_t> values;
        if (take(']')) return values;
        while (true) {
            values.push_back(integer()); ws();
            if (take(']')) return values;
            expect(','); ws();
        }
    }

    void finish() { ws(); if (pos_ != text_.size()) fail("fixture JSON has trailing data"); }

private:
    void ws() { while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
    bool take(char expected) { if (pos_ < text_.size() && text_[pos_] == expected) { ++pos_; return true; } return false; }
    void expect(char expected) { if (!take(expected)) fail("fixture JSON has unexpected token"); }
    const std::string & text_;
    size_t pos_ = 0;
};

struct Sha256 {
    std::array<uint32_t, 8> state{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<uint8_t, 64> block{};
    size_t used = 0;
    uint64_t total = 0;
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void transform(const uint8_t * input) {
        static constexpr uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t(input[i*4]) << 24) | (uint32_t(input[i*4+1]) << 16) | (uint32_t(input[i*4+2]) << 8) | input[i*4+3];
        for (int i = 16; i < 64; ++i) { const uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3); const uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; ++i) { const uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g), t1=h+S1+ch+k[i]+w[i]; const uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c), t2=S0+maj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    void update(const uint8_t * data, size_t size) { total += size; while (size) { const size_t take = std::min(size, block.size() - used); std::memcpy(block.data()+used, data, take); used += take; data += take; size -= take; if (used == block.size()) { transform(block.data()); used = 0; } } }
    std::string finish() { const uint64_t bits = total * 8; block[used++] = 0x80; if (used > 56) { while (used < 64) block[used++] = 0; transform(block.data()); used = 0; } while (used < 56) block[used++] = 0; for (int i = 7; i >= 0; --i) block[used++] = static_cast<uint8_t>(bits >> (i*8)); transform(block.data()); char output[65]; for (int i=0;i<8;++i) std::snprintf(output+i*8,9,"%08x",state[i]); output[64]='\0'; return output; }
};

std::string sha256(const std::vector<uint8_t> & data) { Sha256 hash; hash.update(data.data(), data.size()); return hash.finish(); }

struct Metadata {
    std::string model_id = "prism-ml/Ternary-Bonsai-27B-gguf";
    std::string model_sha;
    int64_t model_size = 0;
    std::string model_filename;
    std::string tensor_name;
    std::string tensor_sha;
    std::string tensor_file;
    std::string tensor_type;
    std::vector<int64_t> tensor_shape;
    int64_t tensor_size = 0;
    std::string activation_file;
    std::string activation_sha;
    std::string activation_type;
    std::vector<int64_t> activation_shape;
    int64_t activation_size = 0;
};

void require_field(bool present, const char * name) { if (!present) fail(std::string("fixture.json missing ") + name); }
void reject_field(const std::string & key, const char * object) { fail("fixture.json unknown " + std::string(object) + " field: " + key); }

Metadata load_metadata(const fs::path & fixture_dir, const Options & options,
                       const std::vector<uint8_t> & weights, const std::vector<uint8_t> & activation) {
    std::ifstream input(fixture_dir / "fixture.json");
    if (!input) fail("normal mode requires fixture.json");
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    Metadata metadata; bool schema=false, model=false, tensor=false, activation_object=false;
    JsonParser parser(text);
    parser.object([&](const std::string & key) {
        if (key == "schema") { if (parser.string() != "q1-cpu-operator-fixture/v1") fail("fixture.json schema mismatch"); schema=true; }
        else if (key == "model") { bool filename=false,sha=false,size=false; parser.object([&](const std::string & field) { if(field=="filename"){metadata.model_filename=parser.string();filename=true;} else if(field=="sha256"){metadata.model_sha=parser.string();sha=true;} else if(field=="size_bytes"){metadata.model_size=parser.integer();size=true;} else reject_field(field,"model"); }); require_field(filename,"model.filename"); require_field(sha,"model.sha256"); require_field(size,"model.size_bytes"); model=true; }
        else if (key == "tensor") { bool name=false,sha=false,file=false,type=false,shape=false,size=false; parser.object([&](const std::string & field) { if(field=="name"){metadata.tensor_name=parser.string();name=true;} else if(field=="sha256"){metadata.tensor_sha=parser.string();sha=true;} else if(field=="file"){metadata.tensor_file=parser.string();file=true;} else if(field=="ggml_type"){metadata.tensor_type=parser.string();type=true;} else if(field=="shape"){metadata.tensor_shape=parser.integer_array();shape=true;} else if(field=="size_bytes"){metadata.tensor_size=parser.integer();size=true;} else if(field=="source_offset_bytes" || field=="index"){(void)parser.integer();} else reject_field(field,"tensor"); }); require_field(name,"tensor.name"); require_field(sha,"tensor.sha256"); require_field(file,"tensor.file"); require_field(type,"tensor.ggml_type"); require_field(shape,"tensor.shape"); require_field(size,"tensor.size_bytes"); tensor=true; }
        else if (key == "activation") { bool dtype=false,file=false,sha=false,shape=false,size=false; parser.object([&](const std::string & field) { if(field=="dtype"){metadata.activation_type=parser.string();dtype=true;} else if(field=="file"){metadata.activation_file=parser.string();file=true;} else if(field=="sha256"){metadata.activation_sha=parser.string();sha=true;} else if(field=="shape"){metadata.activation_shape=parser.integer_array();shape=true;} else if(field=="size_bytes"){metadata.activation_size=parser.integer();size=true;} else if(field=="prng"||field=="seed"||field=="transform"){ if(field=="seed")(void)parser.integer(); else (void)parser.string(); } else reject_field(field,"activation"); }); require_field(dtype,"activation.dtype"); require_field(file,"activation.file"); require_field(sha,"activation.sha256"); require_field(shape,"activation.shape"); require_field(size,"activation.size_bytes"); activation_object=true; }
        else reject_field(key,"root");
    });
    parser.finish(); require_field(schema,"schema"); require_field(model,"model"); require_field(tensor,"tensor"); require_field(activation_object,"activation");
    if (metadata.model_filename != "Bonsai-27B-Q1_0.gguf" || metadata.model_sha != "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0" || metadata.model_size != 3803452480) fail("fixture model is not the pinned Bonsai model");
    if (metadata.tensor_type != "Q1_0" || metadata.tensor_shape.size() != 2 || metadata.tensor_shape[0] != options.ne0 || metadata.tensor_shape[1] != options.ne1 || metadata.tensor_size != static_cast<int64_t>(weights.size())) fail("fixture tensor metadata does not match inputs");
    if (metadata.activation_type != "F32" || metadata.activation_shape.size() != 1 || metadata.activation_shape[0] != options.ne0 || metadata.activation_size != static_cast<int64_t>(activation.size())) fail("fixture activation metadata does not match inputs");
    if (metadata.tensor_file != "weights.q1_0.bin" || metadata.activation_file != "activation.f32.bin") fail("fixture file names are not canonical");
    if (sha256(weights) != metadata.tensor_sha || sha256(activation) != metadata.activation_sha) fail("fixture payload SHA-256 mismatch");
    if (!options.self_test) {
        const bool gate = options.ne0 == 5120 && options.ne1 == 17408;
        const bool down = options.ne0 == 17408 && options.ne1 == 5120;
        const std::string expected_name = gate ? "blk.0.ffn_gate.weight" : (down ? "blk.0.ffn_down.weight" : "");
        const std::string expected_sha = gate ? "0f42ca3b81099f540ed67941809ee7fdc0a672563bf852b87db75034135fc6fe" : (down ? "ab3ef165a5940b14ff1279843af15cbde0fa6cffc2e8eb063be70f7541f0fd04" : "");
        if (expected_name.empty() || metadata.tensor_name != expected_name || metadata.tensor_sha != expected_sha) fail("normal mode requires a pinned production tensor");
    }
    return metadata;
}

double monotonic_raw_us() {
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) fail("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    return static_cast<double>(timestamp.tv_sec) * 1.0e6 + static_cast<double>(timestamp.tv_nsec) / 1.0e3;
}

void phase(const char * event) {
    const char * path = std::getenv("VIP9000_PHASE_FILE");
    if (!path || !*path) return;
    static uint64_t step = 0;
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) fail("clock_gettime(CLOCK_MONOTONIC) failed");
    const uint64_t monotonic_ns = static_cast<uint64_t>(timestamp.tv_sec) * 1000000000ULL + static_cast<uint64_t>(timestamp.tv_nsec);
    const std::string row = "{\"event\":\"" + std::string(event) + "\",\"monotonic_ns\":" + std::to_string(monotonic_ns) + ",\"step\":" + std::to_string(++step) + "}\n";
    const int descriptor = ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (descriptor < 0) fail("cannot open VIP9000_PHASE_FILE");
    size_t offset = 0;
    while (offset < row.size()) {
        const ssize_t written = ::write(descriptor, row.data() + offset, row.size() - offset);
        if (written <= 0) { ::close(descriptor); fail("cannot write VIP9000_PHASE_FILE"); }
        offset += static_cast<size_t>(written);
    }
    if (::fdatasync(descriptor) != 0) { ::close(descriptor); fail("cannot sync VIP9000_PHASE_FILE"); }
    ::close(descriptor);
}

void json_common(std::ostream & stream, const char * record) {
    stream << "{\"schema_version\":\"" << kSchema << "\",\"record\":\"" << record << "\"";
}

ggml_backend_buffer_type_t find_cpu_repack(ggml_backend_t cpu_backend) {
    ggml_backend_dev_t device = ggml_backend_get_device(cpu_backend);
    ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
    auto getter = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
    if (!getter) fail("CPU backend does not expose ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t * extra = getter(device);
    if (!extra) fail("CPU backend returned no extra buffer types");
    for (size_t index = 0; extra[index] != nullptr; ++index) {
        if (std::strcmp(ggml_backend_buft_name(extra[index]), "CPU_REPACK") == 0) return extra[index];
    }
    fail("CPU_REPACK buffer type is unavailable");
}

void emit_identity(std::ostream & stream, const Metadata & metadata, const Options & options) {
    json_common(stream, "identity");
    stream << ",\"run_id\":\"" << options.run_id << "\""
           << ",\"model\":{\"id\":\"" << metadata.model_id << "\",\"sha256\":\"" << metadata.model_sha
           << "\",\"size_bytes\":" << metadata.model_size << ",\"quantization\":\"Q1_0\"}"
           << ",\"tensor\":{\"name\":\"" << metadata.tensor_name << "\",\"sha256\":\"" << metadata.tensor_sha
           << "\",\"shape\":[" << options.ne0 << "," << options.ne1 << "],\"packed_bytes\":"
           << (options.ne0 * options.ne1 / 128 * 18) << "}"
           << ",\"layout\":{\"logical_type\":\"GGML_TYPE_Q1_0\",\"source_layout\":\"GGUF_Q1_0/v1\",\"executor_layout\":\"CPU_REPACK_Q1_0_4x4\"}"
           << ",\"workload\":{\"activation_f32_elements\":" << options.ne0 << ",\"output_f32_elements\":" << options.ne1 << "}"
           << ",\"weight_buffer_type\":\"CPU_REPACK\",\"backend\":\"ggml-cpu\",\"kernel\":\"q1_0_4x4_q8_0\",\"strict_mode\":"
           << (Q1_STRICT_MODE ? "true" : "false") << ",\"threads\":" << options.threads << ",\"logical_gemv_count\":1}\n";
}

void emit_memory(std::ostream & stream, const Options & options) {
    const int64_t canonical = options.ne0 * options.ne1 / 128 * 18;
    const int64_t q8_bytes = options.ne0 / 32 * 34;
    json_common(stream, "memory");
    stream << ",\"declared_canonical_q1_bytes\":" << canonical
           << ",\"declared_cpu_repack_bytes\":" << canonical
           << ",\"expanded_weight_ddr_bytes\":0"
           << ",\"activation_f32_bytes\":" << options.ne0 * static_cast<int64_t>(sizeof(float))
           << ",\"activation_q8_bytes\":" << q8_bytes
           << ",\"output_f32_bytes\":" << options.ne1 * static_cast<int64_t>(sizeof(float))
           << ",\"observed_ddr_read_bytes\":null,\"observed_ddr_write_bytes\":null"
           << ",\"physical_resident_weight_copies\":1,\"cpu_fallback_count\":0}\n";
}

void emit_sample(std::ostream & stream, const char * kind, int iteration, double host_us, double compute_us) {
    json_common(stream, "sample");
    stream << ",\"kind\":\"" << kind << "\",\"iteration\":" << iteration
           << ",\"host_us\":" << host_us << ",\"compute_us\":" << compute_us << "}\n";
}

void emit_result(std::ostream & stream, const Options & options) {
    json_common(stream, "result");
    stream << ",\"output_f32_bytes\":" << options.ne1 * static_cast<int64_t>(sizeof(float)) << "}\n";
}

struct BackendDeleter { void operator()(ggml_backend * value) const { if (value) ggml_backend_free(value); } };
struct ContextDeleter { void operator()(ggml_context * value) const { if (value) ggml_free(value); } };
struct BufferDeleter { void operator()(ggml_backend_buffer * value) const { if (value) ggml_backend_buffer_free(value); } };

void run(const Options & options) {
    phase("setup_begin");
    std::vector<uint8_t> canonical_input = read_bytes(options.weights);
    const std::vector<uint8_t> activation_bytes = read_bytes(options.activation);
    const size_t expected_weight_bytes = static_cast<size_t>(options.ne0 * options.ne1 / 128 * 18);
    if (canonical_input.size() != expected_weight_bytes) fail("weights size does not match Q1_0 dimensions");
    if (activation_bytes.size() != static_cast<size_t>(options.ne0) * sizeof(float)) fail("activation size does not match ne0");
    const fs::path fixture_dir = options.fixture_dir.empty() ? options.weights.parent_path() : options.fixture_dir;
    const Metadata metadata = load_metadata(fixture_dir, options, canonical_input, activation_bytes);
    OutputFiles outputs(options);
    std::vector<float> activation(static_cast<size_t>(options.ne0));
    std::memcpy(activation.data(), activation_bytes.data(), activation_bytes.size());
    const size_t q8_size = static_cast<size_t>(options.ne0 / 32 * 34);
    std::vector<uint8_t> q8(q8_size);
    if (ggml_quantize_chunk(GGML_TYPE_Q8_0, activation.data(), q8.data(), 0, 1, options.ne0, nullptr) != q8_size) fail("ggml_quantize_chunk returned an unexpected Q8_0 size");

    std::unique_ptr<ggml_backend, BackendDeleter> cpu_backend(ggml_backend_cpu_init());
    if (!cpu_backend) fail("ggml_backend_cpu_init failed");
    ggml_backend_cpu_set_n_threads(cpu_backend.get(), options.threads);
    ggml_backend_buffer_type_t repack_buft = find_cpu_repack(cpu_backend.get());

    ggml_init_params weight_params{ggml_tensor_overhead() * 2, nullptr, true};
    std::unique_ptr<ggml_context, ContextDeleter> weight_ctx(ggml_init(weight_params));
    if (!weight_ctx) fail("weight context allocation failed");
    ggml_tensor * weight = ggml_new_tensor_2d(weight_ctx.get(), GGML_TYPE_Q1_0, options.ne0, options.ne1);
    if (!weight) fail("weight tensor allocation failed");
    phase("repack_begin");
    std::unique_ptr<ggml_backend_buffer, BufferDeleter> weight_buffer(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx.get(), repack_buft));
    if (!weight_buffer) fail("CPU_REPACK allocation failed");
    if (!weight->extra) fail("CPU_REPACK does not provide a Q1_0 repack kernel on this CPU");
    ggml_backend_tensor_set(weight, canonical_input.data(), 0, canonical_input.size());
    phase("repack_end");

    ggml_init_params compute_params{ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(2, false), nullptr, true};
    std::unique_ptr<ggml_context, ContextDeleter> compute_ctx(ggml_init(compute_params));
    if (!compute_ctx) fail("compute context allocation failed");
    ggml_tensor * input = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, options.ne0, 1);
    ggml_tensor * output = ggml_mul_mat(compute_ctx.get(), weight, input);
    if (!input || !output) fail("graph tensor construction failed");
    std::unique_ptr<ggml_backend_buffer, BufferDeleter> compute_buffer(ggml_backend_alloc_ctx_tensors(compute_ctx.get(), cpu_backend.get()));
    if (!compute_buffer) fail("CPU activation/output allocation failed");
    ggml_backend_tensor_set(input, activation.data(), 0, activation_bytes.size());
    if (!ggml_backend_supports_op(cpu_backend.get(), output)) fail("ggml-cpu does not support Q1_0 GGML_OP_MUL_MAT with CPU_REPACK");
    if (!weight->buffer || std::strcmp(ggml_backend_buffer_name(weight->buffer), "CPU_REPACK") != 0) fail("Q1 source tensor is not bound to CPU_REPACK");
    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), 2, false);
    if (!graph) fail("graph allocation failed");
    ggml_build_forward_expand(graph, output);
    std::vector<uint8_t>().swap(canonical_input);

    std::ostringstream records;
    records << std::setprecision(17);
    emit_identity(records, metadata, options);
    emit_memory(records, options);
    phase("warmup_begin");
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        const double host_begin = monotonic_raw_us();
        const double compute_begin = monotonic_raw_us();
        if (ggml_backend_graph_compute(cpu_backend.get(), graph) != GGML_STATUS_SUCCESS) fail("warmup graph compute failed");
        const double compute_end = monotonic_raw_us();
        emit_sample(records, "warmup", iteration, compute_end - host_begin, compute_end - compute_begin);
    }
    phase("warmup_end");
    phase("compute_begin");
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        const double host_begin = monotonic_raw_us();
        const double compute_begin = monotonic_raw_us();
        if (ggml_backend_graph_compute(cpu_backend.get(), graph) != GGML_STATUS_SUCCESS) fail("measured graph compute failed");
        const double compute_end = monotonic_raw_us();
        emit_sample(records, "measured", iteration, compute_end - host_begin, compute_end - compute_begin);
    }
    phase("compute_end");
    phase("output_read_begin");
    std::vector<float> result(static_cast<size_t>(options.ne1));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    phase("output_read_end");
    emit_result(records, options);
    const std::string record_text = records.str();
    outputs.write(0, record_text.data(), record_text.size());
    outputs.write(1, result.data(), result.size() * sizeof(float));
    outputs.write(2, q8.data(), q8.size());
    phase("teardown");
    outputs.publish();
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            usage(stdout);
            return 0;
        }
        run(options);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "q1_cpu_operator_runner: %s\n", error.what());
        return 2;
    }
}
