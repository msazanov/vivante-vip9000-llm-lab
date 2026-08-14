#define _GNU_SOURCE

#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

static pid_t replacement_pid;
static unsigned int replacement_reads;

/* Test-only LD_PRELOAD seam: redirect exactly /proc/1/stat to a fixture. */
static int open_via_syscall(const char *path, int flags, va_list arguments) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        mode = (mode_t)va_arg(arguments, int);
    }
    return (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}

static int raw_read_only(const char *path) {
    return (int)syscall(SYS_openat, AT_FDCWD, path, O_RDONLY | O_CLOEXEC, 0);
}

static int synthesize_owned_stat(const char *path, int flags) {
    char input[4096];
    char output[4096];
    char *closing;
    char *cursor;
    char *end;
    char *start_begin = NULL;
    char *start_end = NULL;
    long long parent_pid = -1;
    unsigned long long start_time = 0;
    ssize_t size;
    int field;
    int source;
    int memory;
    int output_size;
    pid_t stat_pid;
    const char *zero_mode = getenv("E049C_TEST_ZERO_OWNED_START");
    const char *replace_mode = getenv("E049C_TEST_REPLACE_OWNED_START");

    if ((zero_mode == NULL && replace_mode == NULL) ||
        strncmp(path, "/proc/", 6) != 0 || strstr(path + 6, "/stat") == NULL ||
        (flags & O_ACCMODE) != O_RDONLY) {
        return -2;
    }
    stat_pid = (pid_t)strtol(path + 6, &end, 10);
    if (stat_pid <= 0 || strcmp(end, "/stat") != 0) {
        return -2;
    }
    source = raw_read_only(path);
    if (source < 0) {
        return source;
    }
    size = read(source, input, sizeof(input) - 1U);
    if (size <= 0 || size >= (ssize_t)sizeof(input) || lseek(source, 0, SEEK_SET) < 0) {
        close(source);
        return -1;
    }
    input[size] = '\0';
    closing = strrchr(input, ')');
    if (closing == NULL || closing[1] != ' ') {
        close(source);
        return -1;
    }
    cursor = closing + 4;
    for (field = 4; field <= 22; ++field) {
        if (field == 22) {
            start_begin = cursor;
            start_time = strtoull(cursor, &end, 10);
            start_end = end;
        } else {
            long long value = strtoll(cursor, &end, 10);
            if (field == 4) {
                parent_pid = value;
            }
        }
        if (end == cursor) {
            close(source);
            return -1;
        }
        cursor = end;
        while (*cursor == ' ') {
            ++cursor;
        }
    }
    if (parent_pid != (long long)getpid()) {
        return source;
    }
    if (replace_mode != NULL) {
        if (replacement_pid != stat_pid) {
            replacement_pid = stat_pid;
            replacement_reads = 0;
        }
        replacement_reads += 1U;
        if ((replacement_reads & 1U) != 0U) {
            return source;
        }
        start_time += 1U;
    } else {
        start_time = 0;
    }
    output_size = snprintf(
        output, sizeof(output), "%.*s%llu%s",
        (int)(start_begin - input), input, start_time, start_end
    );
    close(source);
    if (output_size <= 0 || output_size >= (int)sizeof(output)) {
        return -1;
    }
    memory = (int)syscall(SYS_memfd_create, "e049c-proc-stat", MFD_CLOEXEC);
    if (memory < 0 || write(memory, output, (size_t)output_size) != output_size ||
        lseek(memory, 0, SEEK_SET) < 0) {
        if (memory >= 0) {
            close(memory);
        }
        return -1;
    }
    return memory;
}

static int redirected_open(const char *path, int flags, va_list arguments) {
    const char *fixture = getenv("E049C_TEST_PROC1_STAT");
    int synthesized = synthesize_owned_stat(path, flags);
    if (synthesized != -2) {
        return synthesized;
    }
    if (fixture != NULL && strcmp(path, "/proc/1/stat") == 0) {
        return open_via_syscall(fixture, flags, arguments);
    }
    return open_via_syscall(path, flags, arguments);
}

int open(const char *path, int flags, ...) {
    va_list arguments;
    int result;
    va_start(arguments, flags);
    result = redirected_open(path, flags, arguments);
    va_end(arguments);
    return result;
}

int open64(const char *path, int flags, ...) {
    va_list arguments;
    int result;
    va_start(arguments, flags);
    result = redirected_open(path, flags, arguments);
    va_end(arguments);
    return result;
}
