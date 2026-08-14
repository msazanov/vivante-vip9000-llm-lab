#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>

#include <unistd.h>

// E049d measurement gate. When LLAMA_E049D_STEADY_PMU is absent this object
// performs no I/O. In enabled mode it emits S and waits for A before the first
// single-token graph, then emits E after exactly three successful single-token
// graphs. The fixed descriptors match a733_pmu_exec.
class e049d_steady_pmu {
public:
    e049d_steady_pmu() {
        const char * value = std::getenv("LLAMA_E049D_STEADY_PMU");
        enabled_ = value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    bool before_graph(uint32_t n_tokens, bool & measured) {
        measured = false;
        if (!enabled_ || n_tokens != 1 || measured_count_ >= measured_limit_) {
            return true;
        }
        if (measured_count_ == 0 && (!write_marker('S') || !wait_for_ack())) {
            return false;
        }
        measured = true;
        return true;
    }

    bool after_graph(bool measured) {
        if (!measured) {
            return true;
        }
        ++measured_count_;
        return measured_count_ != measured_limit_ || write_marker('E');
    }

private:
    static constexpr int ack_fd_ = 8;
    static constexpr int marker_fd_ = 9;
    static constexpr uint32_t measured_limit_ = 3;

    static bool write_marker(char marker) {
        ssize_t bytes;
        do {
            bytes = write(marker_fd_, &marker, 1);
        } while (bytes < 0 && errno == EINTR);
        return bytes == 1;
    }

    static bool wait_for_ack() {
        char ack = '\0';
        ssize_t bytes;
        do {
            bytes = read(ack_fd_, &ack, 1);
        } while (bytes < 0 && errno == EINTR);
        return bytes == 1 && ack == 'A';
    }

    bool enabled_ = false;
    uint32_t measured_count_ = 0;
};
