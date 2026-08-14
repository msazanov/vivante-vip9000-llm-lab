/* Deterministic CPU/memory controls for validating the E049c PMU envelope. */

#define _GNU_SOURCE

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_marker(char marker) {
    const char *fd_text = getenv("A733_PMU_SYNC_FD");
    char *end = NULL;
    long fd_number;
    ssize_t written;

    if (fd_text == NULL || *fd_text == '\0') {
        return 0;
    }
    errno = 0;
    fd_number = strtol(fd_text, &end, 10);
    if (errno != 0 || end == fd_text || *end != '\0' || fd_number < 0 || fd_number > INT_MAX) {
        return -1;
    }
    do {
        written = write((int)fd_number, &marker, 1);
    } while (written < 0 && errno == EINTR);
    return written == 1 ? 0 : -1;
}

static int parse_positive(const char *text, uint64_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        return -1;
    }
    *value = (uint64_t)parsed;
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --mode cpu|memory [--iterations N] [--bytes N]\n",
            program);
}

static uint64_t run_cpu(uint64_t iterations) {
    volatile uint64_t value = UINT64_C(0x123456789abcdef0);
    for (uint64_t index = 0; index < iterations; ++index) {
        value ^= value << 7;
        value ^= value >> 9;
        value += UINT64_C(0x9e3779b97f4a7c15) + index;
    }
    return value;
}

static uint64_t run_memory(uint64_t bytes, uint64_t iterations) {
    uint8_t *buffer = NULL;
    volatile uint64_t checksum = 0;
    size_t size;

    if (bytes > SIZE_MAX || bytes < 64 || bytes % 64 != 0) {
        return 0;
    }
    size = (size_t)bytes;
    if (posix_memalign((void **)&buffer, 64, size) != 0 || buffer == NULL) {
        return 0;
    }
    for (size_t offset = 0; offset < size; offset += 64) {
        buffer[offset] = (uint8_t)(offset / 64U + 1U);
    }
    for (uint64_t repeat = 0; repeat < iterations; ++repeat) {
        for (size_t offset = 0; offset < size; offset += 64) {
            uint64_t lane = buffer[offset];
            checksum += lane + (uint64_t)offset;
            buffer[offset] = (uint8_t)(lane + repeat + 1U);
        }
    }
    printf("control=memory bytes=%" PRIu64 " iterations=%" PRIu64 " checksum=%" PRIu64 "\n",
           bytes, iterations, checksum);
    free(buffer);
    return checksum;
}

int main(int argc, char **argv) {
    static const struct option options[] = {
        {"mode", required_argument, NULL, 'm'},
        {"iterations", required_argument, NULL, 'i'},
        {"bytes", required_argument, NULL, 'b'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    const char *mode = NULL;
    uint64_t iterations = 10000000;
    uint64_t bytes = 64 * 1024 * 1024;
    int option;
    uint64_t checksum;

    while ((option = getopt_long(argc, argv, "m:i:b:h", options, NULL)) != -1) {
        switch (option) {
        case 'm':
            mode = optarg;
            break;
        case 'i':
            if (parse_positive(optarg, &iterations) != 0) {
                fprintf(stderr, "invalid iterations: %s\n", optarg);
                return 2;
            }
            break;
        case 'b':
            if (parse_positive(optarg, &bytes) != 0) {
                fprintf(stderr, "invalid bytes: %s\n", optarg);
                return 2;
            }
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (mode == NULL || (strcmp(mode, "cpu") != 0 && strcmp(mode, "memory") != 0)) {
        usage(argv[0]);
        return 2;
    }
    if (write_marker('S') != 0) {
        fprintf(stderr, "could not write start marker: %s\n", strerror(errno));
        return 2;
    }
    if (strcmp(mode, "cpu") == 0) {
        checksum = run_cpu(iterations);
        printf("control=cpu iterations=%" PRIu64 " checksum=%" PRIu64 "\n",
               iterations, checksum);
    } else {
        checksum = run_memory(bytes, iterations);
        if (checksum == 0) {
            fprintf(stderr, "memory control allocation/shape failed\n");
            (void)write_marker('E');
            return 2;
        }
    }
    if (write_marker('E') != 0) {
        fprintf(stderr, "could not write end marker: %s\n", strerror(errno));
        return 2;
    }
    return 0;
}
