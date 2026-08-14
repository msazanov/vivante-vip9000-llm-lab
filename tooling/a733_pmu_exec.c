/*
 * A733/VIP9000 lab PMU envelope.
 *
 * The launcher is deliberately small and uses the Linux perf_event_open(2)
 * ABI directly.  It does not translate any event into DDR bytes: raw PMU
 * values are event counts and their availability/semantics are published in
 * the JSON result.  The process that owns the counters stays privileged while
 * the child is dropped to the requested account before exec(3).
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror tooling/a733_pmu_exec.c -o a733-pmu-exec
 *
 * In exact-boundary mode the child command must write one byte 'S' to the file
 * descriptor named by A733_PMU_SYNC_FD immediately before the measured work,
 * and one byte 'E' immediately after it.  The companion control program in
 * this directory implements that protocol.
 */

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <grp.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PERF_TYPE_RAW
#define PERF_TYPE_RAW 4
#endif

#ifndef PERF_FORMAT_TOTAL_TIME_ENABLED
#define PERF_FORMAT_TOTAL_TIME_ENABLED (1ULL << 0)
#endif

#ifndef PERF_FORMAT_TOTAL_TIME_RUNNING
#define PERF_FORMAT_TOTAL_TIME_RUNNING (1ULL << 1)
#endif

/* Keep this in the shell's portable redirection range; dash rejects >&198. */
#define SYNC_FD 9
#define DEFAULT_USER "orangepi"
#define DEFAULT_GROUP "orangepi"
#define DEFAULT_TIMEOUT_MS 30000
#define DEFAULT_MAX_TEMP_MC 85000
#define MAX_EVENTS 8

/* Linux arm_pmuv3.h architectural/common event encodings. */
struct event_definition {
    const char *name;
    uint64_t config;
    const char *meaning;
};

static const struct event_definition EVENT_DEFINITIONS[MAX_EVENTS] = {
    {"cpu_cycles", 0x11, "ARMv8 PMUv3 CPU_CYCLES event count"},
    {"instructions", 0x08, "ARMv8 PMUv3 INST_RETIRED event count"},
    {"l1d_cache_refill", 0x03, "ARMv8 PMUv3 L1D_CACHE_REFILL event count"},
    {"l2d_cache_refill", 0x17, "ARMv8 PMUv3 L2D_CACHE_REFILL event count"},
    {"l3d_cache_refill", 0x2A, "ARMv8 PMUv3 L3D_CACHE_REFILL event count"},
    {"mem_access", 0x13, "ARMv8 PMUv3 MEM_ACCESS event count"},
    {"bus_access", 0x19, "ARMv8 PMUv3 BUS_ACCESS event count"},
    {"stall_backend", 0x24, "ARMv8 PMUv3 STALL_BACKEND event count"},
};

struct event_state {
    int fd;
    int open_errno;
    int read_errno;
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
};

struct options {
    const char *output_path;
    const char *user_name;
    const char *group_name;
    int no_drop;
    int start_on_ready;
    int sync_timeout_ms;
    int max_temp_mc;
    int thermal_guard;
    int command_index;
};

struct run_state {
    pid_t child_pid;
    int setup_pipe[2];
    int sync_pipe[2];
    int sync_enabled;
    int started;
    int ended;
    int thermal_tripped;
    int timed_out;
    int internal_failure;
    int child_status_valid;
    int child_status;
    int measured_elapsed_valid;
    uint64_t measured_elapsed_ns;
    int64_t max_temp_mc;
    struct event_state events[MAX_EVENTS];
    size_t supported_events;
    char failure_reason[256];
};

static int perf_event_open_local(struct perf_event_attr *attr, pid_t pid, int cpu,
                                 int group_fd, unsigned long flags) {
    return (int)syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void set_failure(struct run_state *run, const char *reason) {
    if (run->failure_reason[0] != '\0') {
        return;
    }
    snprintf(run->failure_reason, sizeof(run->failure_reason), "%s", reason);
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 1 || parsed > INT_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_temp_mc(const char *text, int *value) {
    char *end = NULL;
    double celsius;
    errno = 0;
    celsius = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || celsius < 1.0 || celsius > 200.0) {
        return -1;
    }
    *value = (int)(celsius * 1000.0 + 0.5);
    return 0;
}

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "Usage: %s [options] -- command [args...]\n"
            "\n"
            "Options:\n"
            "  -o, --output PATH          JSON result path (default pmu-result.json)\n"
            "  -u, --user NAME            drop child to NAME (default orangepi)\n"
            "  -g, --group NAME           drop child to GROUP (default orangepi)\n"
            "      --no-drop              keep current uid/gid (host tests only)\n"
            "      --start-on-ready       exact S/E boundary protocol (default)\n"
            "      --start-immediately    enable after child setup gate\n"
            "      --sync-timeout-ms N    S/E marker timeout (default 30000)\n"
            "      --max-temp-c C         stop at thermal threshold (default 85)\n"
            "      --no-thermal-guard     disable target thermal guard\n"
            "  -h, --help                 show this help\n"
            "\n"
            "In --start-on-ready mode the child writes 'S' and 'E' to the fd\n"
            "named by A733_PMU_SYNC_FD. PMU values are event counts, never bytes.\n",
            program);
}

static int parse_options(int argc, char **argv, struct options *options) {
    static const struct option long_options[] = {
        {"output", required_argument, NULL, 'o'},
        {"user", required_argument, NULL, 'u'},
        {"group", required_argument, NULL, 'g'},
        {"no-drop", no_argument, NULL, 1000},
        {"start-on-ready", no_argument, NULL, 1001},
        {"start-immediately", no_argument, NULL, 1002},
        {"sync-timeout-ms", required_argument, NULL, 1003},
        {"max-temp-c", required_argument, NULL, 1004},
        {"no-thermal-guard", no_argument, NULL, 1005},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    int option_index = 0;

    options->output_path = "pmu-result.json";
    options->user_name = DEFAULT_USER;
    options->group_name = DEFAULT_GROUP;
    options->no_drop = 0;
    options->start_on_ready = 1;
    options->sync_timeout_ms = DEFAULT_TIMEOUT_MS;
    options->max_temp_mc = DEFAULT_MAX_TEMP_MC;
    options->thermal_guard = 1;
    options->command_index = -1;

    while ((option = getopt_long(argc, argv, "o:u:g:h", long_options, &option_index)) != -1) {
        switch (option) {
        case 'o':
            options->output_path = optarg;
            break;
        case 'u':
            options->user_name = optarg;
            break;
        case 'g':
            options->group_name = optarg;
            break;
        case 1000:
            options->no_drop = 1;
            break;
        case 1001:
            options->start_on_ready = 1;
            break;
        case 1002:
            options->start_on_ready = 0;
            break;
        case 1003:
            if (parse_positive_int(optarg, &options->sync_timeout_ms) != 0) {
                fprintf(stderr, "invalid --sync-timeout-ms: %s\n", optarg);
                return -1;
            }
            break;
        case 1004:
            if (parse_temp_mc(optarg, &options->max_temp_mc) != 0) {
                fprintf(stderr, "invalid --max-temp-c: %s\n", optarg);
                return -1;
            }
            break;
        case 1005:
            options->thermal_guard = 0;
            break;
        case 'h':
            usage(stdout, argv[0]);
            return 1;
        default:
            usage(stderr, argv[0]);
            return -1;
        }
    }
    if (optind >= argc) {
        fprintf(stderr, "missing command after --\n");
        usage(stderr, argv[0]);
        return -1;
    }
    options->command_index = optind;
    return 0;
}

static int set_fd_cloexec(int fd, int enabled) {
    int flags = fcntl(fd, F_GETFD);
    if (flags < 0) {
        return -1;
    }
    if (enabled) {
        flags |= FD_CLOEXEC;
    } else {
        flags &= ~FD_CLOEXEC;
    }
    return fcntl(fd, F_SETFD, flags);
}

static int set_child_identity(const struct options *options, uid_t *uid_out, gid_t *gid_out) {
    struct passwd *passwd_entry;
    struct group *group_entry;

    if (options->no_drop) {
        *uid_out = getuid();
        *gid_out = getgid();
        return 0;
    }
    passwd_entry = getpwnam(options->user_name);
    if (passwd_entry == NULL) {
        fprintf(stderr, "getpwnam(%s): %s\n", options->user_name, strerror(errno));
        return -1;
    }
    group_entry = getgrnam(options->group_name);
    if (group_entry == NULL) {
        fprintf(stderr, "getgrnam(%s): %s\n", options->group_name, strerror(errno));
        return -1;
    }
    *uid_out = passwd_entry->pw_uid;
    *gid_out = group_entry->gr_gid;
    return 0;
}

static int drop_child_identity(const struct options *options, uid_t uid, gid_t gid) {
    struct passwd *passwd_entry;

    if (options->no_drop) {
        return 0;
    }
    if (geteuid() != 0) {
        if (getuid() != uid || getgid() != gid) {
            errno = EPERM;
            return -1;
        }
        return 0;
    }
    passwd_entry = getpwuid(uid);
    if (passwd_entry == NULL) {
        return -1;
    }
    if (initgroups(passwd_entry->pw_name, gid) != 0 || setgid(gid) != 0 || setuid(uid) != 0) {
        return -1;
    }
    return 0;
}

static int64_t read_max_temp_mc(void) {
    DIR *directory;
    struct dirent *entry;
    int64_t maximum = -1;

    directory = opendir("/sys/class/thermal");
    if (directory == NULL) {
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        char path[PATH_MAX];
        FILE *file;
        long value;

        if (strncmp(entry->d_name, "thermal_zone", 12) != 0) {
            continue;
        }
        if (snprintf(path, sizeof(path), "/sys/class/thermal/%s/temp", entry->d_name) >=
            (int)sizeof(path)) {
            continue;
        }
        file = fopen(path, "r");
        if (file == NULL) {
            continue;
        }
        if (fscanf(file, "%ld", &value) == 1) {
            if (value > maximum) {
                maximum = value;
            }
        }
        fclose(file);
    }
    closedir(directory);
    return maximum;
}

static int check_thermal(struct run_state *run, const struct options *options, pid_t child_pid) {
    int64_t temperature;
    if (!options->thermal_guard) {
        return 0;
    }
    temperature = read_max_temp_mc();
    if (temperature < 0) {
        return 0;
    }
    if (temperature > run->max_temp_mc) {
        run->max_temp_mc = temperature;
    }
    if (temperature > options->max_temp_mc) {
        run->thermal_tripped = 1;
        set_failure(run, "thermal_guard_exceeded");
        (void)kill(child_pid, SIGTERM);
        return -1;
    }
    return 0;
}

static void init_event_states(struct run_state *run) {
    size_t index;
    for (index = 0; index < MAX_EVENTS; ++index) {
        run->events[index].fd = -1;
        run->events[index].open_errno = 0;
        run->events[index].read_errno = 0;
        run->events[index].value = 0;
        run->events[index].time_enabled = 0;
        run->events[index].time_running = 0;
    }
}

static size_t open_events(struct run_state *run) {
    size_t index;
    run->supported_events = 0;
    for (index = 0; index < MAX_EVENTS; ++index) {
        struct perf_event_attr attr;
        int fd;

        memset(&attr, 0, sizeof(attr));
        attr.type = PERF_TYPE_RAW;
        attr.size = sizeof(attr);
        attr.config = EVENT_DEFINITIONS[index].config;
        attr.disabled = 1;
        attr.inherit = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fd = perf_event_open_local(&attr, run->child_pid, -1, -1, 0);
        if (fd < 0) {
            run->events[index].open_errno = errno;
            continue;
        }
        run->events[index].fd = fd;
        run->supported_events += 1;
    }
    return run->supported_events;
}

static int reset_enable_events(struct run_state *run) {
    size_t index;
    int enabled = 0;
    for (index = 0; index < MAX_EVENTS; ++index) {
        int fd = run->events[index].fd;
        if (fd < 0) {
            continue;
        }
        if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) != 0 ||
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
            run->events[index].open_errno = errno;
            close(fd);
            run->events[index].fd = -1;
            if (run->supported_events > 0) {
                run->supported_events -= 1;
            }
            continue;
        }
        enabled += 1;
    }
    return enabled;
}

static void disable_events(struct run_state *run) {
    size_t index;
    for (index = 0; index < MAX_EVENTS; ++index) {
        if (run->events[index].fd >= 0) {
            (void)ioctl(run->events[index].fd, PERF_EVENT_IOC_DISABLE, 0);
        }
    }
}

static void read_events(struct run_state *run) {
    size_t index;
    for (index = 0; index < MAX_EVENTS; ++index) {
        struct {
            uint64_t value;
            uint64_t time_enabled;
            uint64_t time_running;
        } sample;
        ssize_t bytes;

        if (run->events[index].fd < 0) {
            continue;
        }
        memset(&sample, 0, sizeof(sample));
        do {
            bytes = read(run->events[index].fd, &sample, sizeof(sample));
        } while (bytes < 0 && errno == EINTR);
        if (bytes != (ssize_t)sizeof(sample)) {
            run->events[index].read_errno = bytes < 0 ? errno : EIO;
            continue;
        }
        run->events[index].value = sample.value;
        run->events[index].time_enabled = sample.time_enabled;
        run->events[index].time_running = sample.time_running;
    }
}

static void close_events(struct run_state *run) {
    size_t index;
    for (index = 0; index < MAX_EVENTS; ++index) {
        if (run->events[index].fd >= 0) {
            close(run->events[index].fd);
        }
    }
}

static int wait_for_marker(struct run_state *run, const struct options *options,
                           int marker_fd, char wanted) {
    const uint64_t deadline = monotonic_ns() + (uint64_t)options->sync_timeout_ms * 1000000ULL;
    struct pollfd descriptor;
    int child_exited = 0;

    descriptor.fd = marker_fd;
    descriptor.events = POLLIN | POLLHUP | POLLERR;
    for (;;) {
        int status;
        pid_t waited;
        int timeout = 100;
        uint64_t now = monotonic_ns();
        if (now >= deadline) {
            run->timed_out = 1;
            set_failure(run, wanted == 'S' ? "start_marker_timeout" : "end_marker_timeout");
            return -1;
        }
        if (deadline - now < 100000000ULL) {
            timeout = (int)((deadline - now) / 1000000ULL);
            if (timeout < 1) {
                timeout = 1;
            }
        }
        waited = waitpid(run->child_pid, &status, WNOHANG);
        if (waited == run->child_pid) {
            run->child_status_valid = 1;
            run->child_status = status;
            /* A fast child can write E and exit before the parent observes
             * either.  Drain the marker pipe before declaring failure. */
            child_exited = 1;
        }
        if (waited < 0 && errno != EINTR) {
            set_failure(run, "waitpid_failed");
            return -1;
        }
        (void)check_thermal(run, options, run->child_pid);
        if (run->thermal_tripped) {
            return -1;
        }
        if (poll(&descriptor, 1, timeout) < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_failure(run, "sync_poll_failed");
            return -1;
        }
        if (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) {
            char marker;
            ssize_t bytes;
            do {
                bytes = read(marker_fd, &marker, 1);
            } while (bytes < 0 && errno == EINTR);
            if (bytes == 1 && marker == wanted) {
                return 0;
            }
            if (bytes == 0) {
                set_failure(run, "sync_pipe_closed");
                return -1;
            }
            if (bytes == 1 && marker != wanted) {
                char message[128];
                snprintf(message, sizeof(message), "unexpected_sync_marker_%c", marker);
                set_failure(run, message);
                return -1;
            }
        }
        if (child_exited) {
            set_failure(run, wanted == 'S' ? "child_exited_before_start_marker"
                                          : "child_exited_before_end_marker");
            return -1;
        }
    }
}

static int wait_for_child(struct run_state *run, const struct options *options) {
    uint64_t last_thermal = monotonic_ns();
    if (run->child_status_valid) {
        return 0;
    }
    for (;;) {
        int status;
        pid_t waited = waitpid(run->child_pid, &status, WNOHANG);
        if (waited == run->child_pid) {
            run->child_status_valid = 1;
            run->child_status = status;
            return 0;
        }
        if (waited < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_failure(run, "waitpid_failed");
            return -1;
        }
        if (monotonic_ns() - last_thermal >= 100000000ULL) {
            last_thermal = monotonic_ns();
            if (check_thermal(run, options, run->child_pid) != 0) {
                return -1;
            }
        }
        (void)poll(NULL, 0, 20);
    }
}

static void terminate_child(struct run_state *run) {
    if (run->child_pid <= 0 || run->child_status_valid) {
        return;
    }
    (void)kill(run->child_pid, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
        int status;
        pid_t waited = waitpid(run->child_pid, &status, WNOHANG);
        if (waited == run->child_pid) {
            run->child_status_valid = 1;
            run->child_status = status;
            return;
        }
        (void)poll(NULL, 0, 10);
    }
    (void)kill(run->child_pid, SIGKILL);
    if (waitpid(run->child_pid, &run->child_status, 0) == run->child_pid) {
        run->child_status_valid = 1;
    }
}

static void json_string(FILE *stream, const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    fputc('"', stream);
    while (*cursor != '\0') {
        switch (*cursor) {
        case '\\':
            fputs("\\\\", stream);
            break;
        case '"':
            fputs("\\\"", stream);
            break;
        case '\n':
            fputs("\\n", stream);
            break;
        case '\r':
            fputs("\\r", stream);
            break;
        case '\t':
            fputs("\\t", stream);
            break;
        default:
            if (*cursor < 0x20U) {
                fprintf(stream, "\\u%04x", *cursor);
            } else {
                fputc(*cursor, stream);
            }
        }
        cursor += 1;
    }
    fputc('"', stream);
}

static const char *event_support(const struct event_state *event) {
    if (event->fd >= 0 && event->read_errno == 0) {
        return "supported";
    }
    if (event->read_errno != 0) {
        return "read_error";
    }
    return "unavailable";
}

static int command_exit_code(const struct run_state *run) {
    if (!run->child_status_valid) {
        return 125;
    }
    if (WIFEXITED(run->child_status)) {
        return WEXITSTATUS(run->child_status);
    }
    if (WIFSIGNALED(run->child_status)) {
        return 128 + WTERMSIG(run->child_status);
    }
    return 125;
}

static const char *result_status(const struct run_state *run) {
    size_t index;
    size_t readable = 0;
    size_t failures = 0;
    if (run->internal_failure || run->timed_out || run->thermal_tripped ||
        command_exit_code(run) != 0) {
        return "failed";
    }
    for (index = 0; index < MAX_EVENTS; ++index) {
        if (run->events[index].fd >= 0 && run->events[index].read_errno == 0) {
            readable += 1;
        } else {
            failures += 1;
        }
    }
    if (readable == 0) {
        return "counter_unavailable";
    }
    return failures == 0 ? "ok" : "partial";
}

static int write_result(const struct options *options, const struct run_state *run,
                        int argc, char **argv) {
    FILE *stream = fopen(options->output_path, "w");
    size_t index;
    int command_code = command_exit_code(run);

    if (stream == NULL) {
        fprintf(stderr, "fopen(%s): %s\n", options->output_path, strerror(errno));
        return -1;
    }
    fprintf(stream, "{\n");
    fprintf(stream, "  \"schema_version\": \"e049c-arm-pmu/v1\",\n");
    fprintf(stream, "  \"status\": ");
    json_string(stream, result_status(run));
    fprintf(stream, ",\n  \"event_source\": \"armv8_pmuv3_raw_config\",\n");
    fprintf(stream, "  \"counter_semantics\": \"event counts only; no DDR-byte conversion\",\n");
    fprintf(stream, "  \"pid\": %ld,\n", (long)run->child_pid);
    fprintf(stream, "  \"command\": [");
    for (index = (size_t)options->command_index; index < (size_t)argc; ++index) {
        if (index != (size_t)options->command_index) {
            fputs(", ", stream);
        }
        json_string(stream, argv[index]);
    }
    fprintf(stream, "],\n");
    fprintf(stream, "  \"sync\": {\"mode\": ");
    json_string(stream, options->start_on_ready ? "ready_markers" : "immediate_after_setup_gate");
    fprintf(stream, ", \"started\": %s, \"ended\": %s},\n",
            run->started ? "true" : "false", run->ended ? "true" : "false");
    fprintf(stream, "  \"measured_elapsed_ns\": %" PRIu64 ",\n", run->measured_elapsed_ns);
    fprintf(stream, "  \"thermal\": {\"guard_enabled\": %s, \"limit_c\": %.3f, "
                   "\"max_observed_c\": ",
            options->thermal_guard ? "true" : "false", (double)options->max_temp_mc / 1000.0);
    if (run->max_temp_mc >= 0) {
        fprintf(stream, "%.3f", (double)run->max_temp_mc / 1000.0);
    } else {
        fputs("null", stream);
    }
    fprintf(stream, ", \"tripped\": %s},\n", run->thermal_tripped ? "true" : "false");
    fprintf(stream, "  \"exit\": {\"code\": %d, \"raw_wait_status\": %d},\n",
            command_code, run->child_status_valid ? run->child_status : -1);
    fprintf(stream, "  \"failure_reason\": ");
    if (run->failure_reason[0] == '\0') {
        fputs("null", stream);
    } else {
        json_string(stream, run->failure_reason);
    }
    fprintf(stream, ",\n  \"events\": [\n");
    for (index = 0; index < MAX_EVENTS; ++index) {
        const struct event_state *event = &run->events[index];
        const char *support = event_support(event);
        if (index != 0) {
            fputs(",\n", stream);
        }
        fprintf(stream, "    {\"name\": ");
        json_string(stream, EVENT_DEFINITIONS[index].name);
        fprintf(stream, ", \"config\": \"0x%llx\", \"meaning\": ",
                (unsigned long long)EVENT_DEFINITIONS[index].config);
        json_string(stream, EVENT_DEFINITIONS[index].meaning);
        fprintf(stream, ", \"support\": ");
        json_string(stream, support);
        fprintf(stream, ", \"count_semantics\": \"event_count_not_bytes\", "
                       "\"value\": ");
        if (strcmp(support, "supported") == 0) {
            fprintf(stream, "%" PRIu64, event->value);
        } else {
            fputs("null", stream);
        }
        fprintf(stream, ", \"time_enabled_ns\": ");
        if (strcmp(support, "supported") == 0) {
            fprintf(stream, "%" PRIu64, event->time_enabled);
        } else {
            fputs("null", stream);
        }
        fprintf(stream, ", \"time_running_ns\": ");
        if (strcmp(support, "supported") == 0) {
            fprintf(stream, "%" PRIu64, event->time_running);
        } else {
            fputs("null", stream);
        }
        fprintf(stream, ", \"scaled_value\": ");
        if (strcmp(support, "supported") == 0 && event->time_running != 0) {
            long double scaled = (long double)event->value *
                                 (long double)event->time_enabled /
                                 (long double)event->time_running;
            fprintf(stream, "%.0Lf", scaled);
        } else {
            fputs("null", stream);
        }
        fprintf(stream, ", \"errno\": %d, \"error\": ",
                event->read_errno != 0 ? event->read_errno : event->open_errno);
        if (event->read_errno != 0) {
            json_string(stream, strerror(event->read_errno));
        } else if (event->open_errno != 0) {
            json_string(stream, strerror(event->open_errno));
        } else {
            fputs("null", stream);
        }
        fputs("}", stream);
    }
    fprintf(stream, "\n  ]\n}\n");
    if (fclose(stream) != 0) {
        fprintf(stderr, "fclose(%s): %s\n", options->output_path, strerror(errno));
        return -1;
    }
    return 0;
}

static int child_main(const struct options *options, char **command, int setup_read_fd,
                      int sync_write_fd, uid_t uid, gid_t gid) {
    char uid_text[32];
    char gid_text[32];
    char fd_text[32];
    char gate;
    ssize_t bytes;

    if (options->start_on_ready) {
        if (dup2(sync_write_fd, SYNC_FD) < 0 || set_fd_cloexec(SYNC_FD, 0) != 0) {
            dprintf(STDERR_FILENO, "sync fd setup failed: %s\n", strerror(errno));
            _exit(126);
        }
        close(sync_write_fd);
        snprintf(fd_text, sizeof(fd_text), "%d", SYNC_FD);
        if (setenv("A733_PMU_SYNC_FD", fd_text, 1) != 0) {
            dprintf(STDERR_FILENO, "setenv sync fd failed: %s\n", strerror(errno));
            _exit(126);
        }
    }
    if (drop_child_identity(options, uid, gid) != 0) {
        dprintf(STDERR_FILENO, "drop child identity failed: %s\n", strerror(errno));
        _exit(126);
    }
    snprintf(uid_text, sizeof(uid_text), "%lu", (unsigned long)getuid());
    snprintf(gid_text, sizeof(gid_text), "%lu", (unsigned long)getgid());
    if (setenv("A733_PMU_CHILD_UID", uid_text, 1) != 0 ||
        setenv("A733_PMU_CHILD_GID", gid_text, 1) != 0) {
        dprintf(STDERR_FILENO, "setenv child identity failed: %s\n", strerror(errno));
        _exit(126);
    }
    do {
        bytes = read(setup_read_fd, &gate, 1);
    } while (bytes < 0 && errno == EINTR);
    close(setup_read_fd);
    if (bytes != 1 || gate != 'G') {
        dprintf(STDERR_FILENO, "setup gate failed\n");
        _exit(126);
    }
    execvp(command[0], command);
    dprintf(STDERR_FILENO, "execvp(%s): %s\n", command[0], strerror(errno));
    _exit(127);
}

static int release_setup_gate(int fd) {
    const char gate = 'G';
    ssize_t bytes;
    do {
        bytes = write(fd, &gate, 1);
    } while (bytes < 0 && errno == EINTR);
    close(fd);
    return bytes == 1 ? 0 : -1;
}

int main(int argc, char **argv) {
    struct options options;
    struct run_state run;
    uid_t child_uid;
    gid_t child_gid;
    int parse_status;
    int setup_pipe[2];
    int sync_pipe[2] = {-1, -1};
    pid_t child;
    char **command;
    int enabled;
    uint64_t measured_start = 0;

    memset(&run, 0, sizeof(run));
    init_event_states(&run);
    run.max_temp_mc = -1;
    run.failure_reason[0] = '\0';
    parse_status = parse_options(argc, argv, &options);
    if (parse_status != 0) {
        return parse_status > 0 ? 0 : 2;
    }
    if (set_child_identity(&options, &child_uid, &child_gid) != 0) {
        return 2;
    }
    if (pipe(setup_pipe) != 0) {
        fprintf(stderr, "pipe setup: %s\n", strerror(errno));
        return 2;
    }
    if (options.start_on_ready && pipe(sync_pipe) != 0) {
        fprintf(stderr, "pipe sync: %s\n", strerror(errno));
        close(setup_pipe[0]);
        close(setup_pipe[1]);
        return 2;
    }
    command = &argv[options.command_index];
    child = fork();
    if (child < 0) {
        fprintf(stderr, "fork: %s\n", strerror(errno));
        close(setup_pipe[0]);
        close(setup_pipe[1]);
        if (sync_pipe[0] >= 0) {
            close(sync_pipe[0]);
            close(sync_pipe[1]);
        }
        return 2;
    }
    run.child_pid = child;
    run.setup_pipe[0] = setup_pipe[0];
    run.setup_pipe[1] = setup_pipe[1];
    run.sync_pipe[0] = sync_pipe[0];
    run.sync_pipe[1] = sync_pipe[1];
    run.sync_enabled = options.start_on_ready;

    if (child == 0) {
        close(setup_pipe[1]);
        if (options.start_on_ready) {
            close(sync_pipe[0]);
            (void)child_main(&options, command, setup_pipe[0], sync_pipe[1], child_uid, child_gid);
        } else {
            (void)child_main(&options, command, setup_pipe[0], -1, child_uid, child_gid);
        }
        _exit(126);
    }

    close(setup_pipe[0]);
    if (options.start_on_ready) {
        close(sync_pipe[1]);
    }
    (void)set_fd_cloexec(setup_pipe[1], 1);
    (void)open_events(&run);
    if (release_setup_gate(setup_pipe[1]) != 0) {
        run.internal_failure = 1;
        set_failure(&run, "parent_setup_gate_failed");
        terminate_child(&run);
        close_events(&run);
        (void)write_result(&options, &run, argc, argv);
        return 2;
    }
    if (options.start_on_ready) {
        if (wait_for_marker(&run, &options, sync_pipe[0], 'S') != 0) {
            terminate_child(&run);
            close(sync_pipe[0]);
            close_events(&run);
            (void)write_result(&options, &run, argc, argv);
            return command_exit_code(&run);
        }
    }
    enabled = reset_enable_events(&run);
    if (enabled == 0) {
        set_failure(&run, "no_perf_events_enabled");
    }
    run.started = 1;
    measured_start = monotonic_ns();
    if (options.start_on_ready) {
        if (wait_for_marker(&run, &options, sync_pipe[0], 'E') != 0) {
            terminate_child(&run);
        } else {
            run.ended = 1;
            disable_events(&run);
            run.measured_elapsed_ns = monotonic_ns() - measured_start;
            (void)wait_for_child(&run, &options);
        }
        close(sync_pipe[0]);
    } else {
        if (wait_for_child(&run, &options) != 0) {
            terminate_child(&run);
        }
        run.ended = run.child_status_valid;
        disable_events(&run);
        run.measured_elapsed_ns = monotonic_ns() - measured_start;
    }
    if (run.measured_elapsed_ns == 0 && measured_start != 0) {
        run.measured_elapsed_ns = monotonic_ns() - measured_start;
    }
    read_events(&run);
    close_events(&run);
    if (run.thermal_tripped) {
        run.internal_failure = 1;
    }
    if (write_result(&options, &run, argc, argv) != 0) {
        return 2;
    }
    return command_exit_code(&run);
}
