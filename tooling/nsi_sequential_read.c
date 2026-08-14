/*
 * E049: exact-byte sequential read stream for sunxi-nsi calibration.
 *
 * Аргумент: <buffer_mib>. Helper first-touches and calibrates the buffer
 * before announcing READY.  The controller then sends
 * "<planned_bytes> <absolute_deadline_ns>\n" through E049_START_FD.
 * Only the exact planned architected load bytes may be reported as DONE.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MIB (1024u * 1024u)
#define LOAD_WIDTH_BYTES ((size_t)sizeof(uint64_t))
#define DEADLINE_CHUNK_BYTES 4096u
#define DEADLINE_GUARD_NS UINT64_C(1000000)
#define CALIBRATION_BYTES (16u * MIB)
#define MAX_PLANNED_BYTES (16ull * 1024ull * 1024ull * 1024ull)

static uint64_t monotonic_ns(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0)
#endif
    {
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            return 0;
        }
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int parse_positive(const char *text, unsigned long long *result) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        return 0;
    }
    *result = value;
    return 1;
}

static int read_exact(
    volatile const uint64_t *buffer,
    size_t buffer_bytes,
    uint64_t planned_bytes,
    uint64_t deadline_ns,
    volatile uint64_t *sink,
    uint64_t *completed_bytes
) {
    uint64_t completed = 0;
    size_t offset_bytes = 0;

    while (completed < planned_bytes) {
        uint64_t remaining = planned_bytes - completed;
        size_t chunk = remaining < DEADLINE_CHUNK_BYTES
            ? (size_t)remaining : DEADLINE_CHUNK_BYTES;
        if (deadline_ns != 0) {
            uint64_t now = monotonic_ns();
            if (now == 0 || now >= deadline_ns || deadline_ns - now <= DEADLINE_GUARD_NS) {
                *completed_bytes = completed;
                return 0;
            }
        }
        for (size_t local = 0; local < chunk; local += LOAD_WIDTH_BYTES) {
            *sink += buffer[(offset_bytes + local) / LOAD_WIDTH_BYTES];
        }
        completed += (uint64_t)chunk;
        offset_bytes += chunk;
        if (offset_bytes == buffer_bytes) {
            offset_bytes = 0;
        }
    }
    *completed_bytes = completed;
    return 1;
}

int main(int argc, char **argv) {
    unsigned long long size_mib;
    size_t buffer_bytes;
    unsigned char *allocation = NULL;
    volatile const uint64_t *read_buffer;
    volatile uint64_t sink = 0;
    uint64_t calibration_bytes = CALIBRATION_BYTES;
    uint64_t calibration_done = 0;
    uint64_t calibration_start_ns;
    uint64_t calibration_end_ns;
    uint64_t planned_bytes;
    uint64_t deadline_ns;
    uint64_t workload_start_ns;
    uint64_t workload_end_ns;
    uint64_t completed_bytes = 0;
    const char *start_fd_text;
    unsigned long long start_fd;
    FILE *start_stream = NULL;
    int completed;

    if (argc != 2 || !parse_positive(argv[1], &size_mib) || size_mib > 4096) {
        fprintf(stderr, "usage: %s <buffer_mib 1..4096>\n", argv[0]);
        return 2;
    }
    if (size_mib > (unsigned long long)(SIZE_MAX / MIB)) {
        fprintf(stderr, "buffer size overflow\n");
        return 2;
    }
    buffer_bytes = (size_t)size_mib * MIB;
    if (posix_memalign((void **)&allocation, 64u, buffer_bytes) != 0 || allocation == NULL) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", buffer_bytes);
        return 3;
    }
    for (size_t offset = 0; offset < buffer_bytes; offset += 64u) {
        allocation[offset] = (unsigned char)((offset >> 6u) ^ 0x5aU);
    }
    read_buffer = (volatile const uint64_t *)(const void *)allocation;

    /* Calibration is deliberately before READY and therefore outside PMU. */
    calibration_start_ns = monotonic_ns();
    if (!read_exact(read_buffer, buffer_bytes, calibration_bytes, 0, &sink, &calibration_done)) {
        fprintf(stderr, "unexpected calibration failure\n");
        free(allocation);
        return 4;
    }
    calibration_end_ns = monotonic_ns();
    if (calibration_start_ns == 0 || calibration_end_ns <= calibration_start_ns) {
        fprintf(stderr, "invalid calibration clock\n");
        free(allocation);
        return 4;
    }

    start_fd_text = getenv("E049_START_FD");
    if (start_fd_text == NULL || !parse_positive(start_fd_text, &start_fd) || start_fd > 1024) {
        fprintf(stderr, "E049_START_FD is required\n");
        free(allocation);
        return 4;
    }
    start_stream = fdopen((int)start_fd, "r");
    if (start_stream == NULL) {
        fprintf(stderr, "fdopen(E049_START_FD) failed\n");
        free(allocation);
        return 4;
    }
    if (printf(
            "READY {\"buffer_bytes\":%zu,\"calibration_bytes\":%" PRIu64
            ",\"calibration_elapsed_ns\":%" PRIu64
            ",\"deadline_chunk_bytes\":%u,\"deadline_guard_ns\":%" PRIu64 "}\n",
            buffer_bytes, calibration_done, calibration_end_ns - calibration_start_ns,
            DEADLINE_CHUNK_BYTES, DEADLINE_GUARD_NS) < 0 || fflush(stdout) != 0) {
        fprintf(stderr, "READY write failed\n");
        fclose(start_stream);
        free(allocation);
        return 4;
    }
    if (fscanf(start_stream, "%" SCNu64 " %" SCNu64, &planned_bytes, &deadline_ns) != 2) {
        fprintf(stderr, "invalid synchronized start command\n");
        fclose(start_stream);
        free(allocation);
        return 4;
    }
    fclose(start_stream);
    if (planned_bytes == 0 || planned_bytes > MAX_PLANNED_BYTES ||
        planned_bytes % LOAD_WIDTH_BYTES != 0 || deadline_ns == 0) {
        fprintf(stderr, "invalid planned_bytes/deadline\n");
        free(allocation);
        return 4;
    }

    workload_start_ns = monotonic_ns();
    completed = read_exact(
        read_buffer, buffer_bytes, planned_bytes, deadline_ns,
        &sink, &completed_bytes
    );
    workload_end_ns = monotonic_ns();
    if (!completed || workload_end_ns == 0 || workload_end_ns > deadline_ns) {
        printf(
            "{\"status\":\"deadline_abort\",\"planned_bytes\":%" PRIu64
            ",\"bytes_read\":%" PRIu64 ",\"workload_start_ns\":%" PRIu64
            ",\"workload_end_ns\":%" PRIu64 ",\"deadline_ns\":%" PRIu64
            ",\"load_width_bytes\":%zu,\"sink\":%" PRIu64 "}\n",
            planned_bytes, completed_bytes, workload_start_ns, workload_end_ns,
            deadline_ns, LOAD_WIDTH_BYTES, sink
        );
        free(allocation);
        return 5;
    }

    printf(
        "{\"status\":\"done\",\"buffer_bytes\":%zu"
        ",\"planned_bytes\":%" PRIu64 ",\"bytes_read\":%" PRIu64
        ",\"load_width_bytes\":%zu,\"deadline_chunk_bytes\":%u"
        ",\"workload_start_ns\":%" PRIu64 ",\"workload_end_ns\":%" PRIu64
        ",\"active_ns\":%" PRIu64 ",\"deadline_ns\":%" PRIu64
        ",\"deadline_margin_ns\":%" PRIu64 ",\"sink\":%" PRIu64 "}\n",
        buffer_bytes, planned_bytes, completed_bytes, LOAD_WIDTH_BYTES,
        DEADLINE_CHUNK_BYTES, workload_start_ns, workload_end_ns,
        workload_end_ns - workload_start_ns, deadline_ns,
        deadline_ns - workload_end_ns, sink
    );
    free(allocation);
    return 0;
}
