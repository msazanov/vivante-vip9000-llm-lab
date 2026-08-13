/*
 * E049: bounded sequential read stream for sunxi-nsi calibration.
 *
 * Аргументы: <buffer_mib> <duration_ms>. Буфер один раз предварительно
 * заполняется до старта измерения, затем volatile pointer читает каждую
 * cache-line. JSON stdout используется Python-оркестратором; stderr остаётся
 * пустым при штатном завершении.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
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

int main(int argc, char **argv) {
    unsigned long long size_mib;
    unsigned long long duration_ms;
    size_t bytes;
    unsigned char *buffer = NULL;
    volatile const unsigned char *read_buffer;
    volatile uint64_t sink = 0;
    uint64_t passes = 0;
    double started;
    double ended;

    if (argc != 3 || !parse_positive(argv[1], &size_mib) ||
        !parse_positive(argv[2], &duration_ms) || size_mib > 4096 ||
        duration_ms > 60000) {
        fprintf(stderr, "usage: %s <buffer_mib 1..4096> <duration_ms 1..60000>\n", argv[0]);
        return 2;
    }
    if (size_mib > (unsigned long long)(SIZE_MAX / (1024u * 1024u))) {
        fprintf(stderr, "buffer size overflow\n");
        return 2;
    }
    bytes = (size_t)size_mib * 1024u * 1024u;
    if (posix_memalign((void **)&buffer, 64u, bytes) != 0 || buffer == NULL) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
        return 3;
    }

    /* Page-faults и first-touch не должны попадать в measured read window. */
    for (size_t offset = 0; offset < bytes; offset += 64u) {
        buffer[offset] = (unsigned char)((offset >> 6u) ^ 0x5aU);
    }
    read_buffer = buffer;
    started = monotonic_seconds();
    do {
        for (size_t offset = 0; offset < bytes; offset += 64u) {
            sink += (uint64_t)read_buffer[offset];
        }
        ++passes;
        ended = monotonic_seconds();
    } while ((ended - started) * 1000.0 < (double)duration_ms);

    printf("{\"bytes_per_pass\":%zu,\"passes\":%" PRIu64
           ",\"bytes_read\":%" PRIu64 ",\"elapsed_s\":%.9f"
           ",\"sink\":%" PRIu64 "}\n",
           bytes, passes, (uint64_t)bytes * passes, ended - started, sink);
    free(buffer);
    return 0;
}
