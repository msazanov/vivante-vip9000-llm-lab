/*
 * A733 process-scoped ARM PMU launcher, evidence schema v2.
 *
 * Exact mode is a bidirectional protocol:
 *   child writes S on fd 9 -> parent RESET+ENABLEs one PMU group atomically
 *   -> parent writes A on fd 8 -> workload runs -> child writes E on fd 9.
 *
 * Raw PMU values are event counts, never DDR bytes.  A run is usable only
 * when every event in its explicit group has a sufficient running ratio.
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
#include <math.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PERF_TYPE_RAW
#define PERF_TYPE_RAW 4
#endif

#ifndef PERF_IOC_FLAG_GROUP
#define PERF_IOC_FLAG_GROUP 1UL
#endif

#ifndef PERF_FLAG_FD_CLOEXEC
#define PERF_FLAG_FD_CLOEXEC (1UL << 3)
#endif

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

#define MARKER_FD 9
#define ACK_FD 8
#define DEFAULT_USER "orangepi"
#define DEFAULT_GROUP "orangepi"
#define DEFAULT_TIMEOUT_MS 30000
#define DEFAULT_MAX_TEMP_MC 85000
#define DEFAULT_MIN_RUNNING_RATIO 0.95
#define SOFTWARE_GROUP_SIZE_LIMIT 4
#define MAX_GROUP_EVENTS SOFTWARE_GROUP_SIZE_LIMIT
#define EVENT_DEFINITION_COUNT 8
#define MAX_ADOPTED_CHILDREN 4096U
#define CLEANUP_TERM_ATTEMPTS 50
#define CLEANUP_KILL_ATTEMPTS 100
#define CLEANUP_POLL_MS 10
#define CLEANUP_EMPTY_CONFIRMATIONS 2

static volatile sig_atomic_t parent_cancel_signal = 0;

struct event_definition {
    const char *name;
    uint64_t config;
    const char *meaning;
};

static const struct event_definition EVENT_DEFINITIONS[EVENT_DEFINITION_COUNT] = {
    {"cpu_cycles", 0x11, "ARMv8 PMUv3 CPU_CYCLES event count"},
    {"instructions", 0x08, "ARMv8 PMUv3 INST_RETIRED event count"},
    {"l1d_cache_refill", 0x03, "ARMv8 PMUv3 L1D_CACHE_REFILL event count"},
    {"l2d_cache_refill", 0x17, "ARMv8 PMUv3 L2D_CACHE_REFILL event count"},
    {"l3d_cache_refill", 0x2A, "ARMv8 PMUv3 L3D_CACHE_REFILL event count"},
    {"mem_access", 0x13, "ARMv8 PMUv3 MEM_ACCESS event count"},
    {"bus_access", 0x19, "ARMv8 PMUv3 BUS_ACCESS event count"},
    {"stall_backend", 0x24, "ARMv8 PMUv3 STALL_BACKEND event count"},
};

struct event_group_definition {
    const char *name;
    size_t event_count;
    size_t event_indices[MAX_GROUP_EVENTS];
};

static const struct event_group_definition EVENT_GROUPS[] = {
    {"core", 3, {0, 1, 7}},
    {"cache", 3, {2, 3, 4}},
    {"memory", 2, {5, 6, 0}},
};

#define EVENT_GROUP_COUNT (sizeof(EVENT_GROUPS) / sizeof(EVENT_GROUPS[0]))

struct event_state {
    const struct event_definition *definition;
    int fd;
    int opened;
    int open_errno;
    int read_errno;
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
    double running_ratio;
    int sample_valid;
};

struct options {
    const char *output_path;
    const char *child_stdout_path;
    const char *child_stderr_path;
    const char *user_name;
    const char *group_name;
    const char *thermal_root;
    const struct event_group_definition *event_group;
    int no_drop;
    int start_on_ready;
    int sync_timeout_ms;
    int max_temp_mc;
    int thermal_guard;
    double min_running_ratio;
    int command_index;
};

struct run_state {
    pid_t child_pid;
    pid_t child_pgid;
    pid_t child_sid;
    int child_session_established;
    int marker_read_fd;
    int ack_write_fd;
    int sync_enabled;
    int started;
    int acknowledged;
    int ended;
    int sync_failed;
    int thermal_checked;
    int thermal_readable;
    int thermal_unreadable;
    int thermal_tripped;
    int timed_out;
    int internal_failure;
    int read_failed;
    int group_open_failed;
    int group_enabled;
    int child_status_valid;
    int child_status;
    int measured_elapsed_valid;
    uint64_t measured_elapsed_ns;
    int64_t max_temp_mc;
    struct event_state events[MAX_GROUP_EVENTS];
    size_t event_count;
    size_t supported_events;
    int group_leader_fd;
    int sample_valid;
    char failure_reason[256];
};

struct proc_identity {
    pid_t pid;
    pid_t parent_pid;
    pid_t process_group;
    pid_t session;
    uint64_t start_time;
    int pidfd;
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
    if (run->failure_reason[0] == '\0') {
        (void)snprintf(run->failure_reason, sizeof(run->failure_reason), "%s", reason);
    }
}

static void parent_signal_handler(int signal_number) {
    if (parent_cancel_signal == 0) {
        parent_cancel_signal = signal_number;
    }
}

static int install_parent_signal_handlers(void) {
    const int cancellation_signals[] = {SIGTERM, SIGHUP, SIGINT};
    struct sigaction action;
    struct sigaction ignored;
    size_t index;
    memset(&action, 0, sizeof(action));
    sigemptyset(&action.sa_mask);
    action.sa_handler = parent_signal_handler;
    memset(&ignored, 0, sizeof(ignored));
    sigemptyset(&ignored.sa_mask);
    ignored.sa_handler = SIG_IGN;
    for (index = 0; index < sizeof(cancellation_signals) /
                                sizeof(cancellation_signals[0]); ++index) {
        if (sigaction(cancellation_signals[index], &action, NULL) != 0) {
            return -1;
        }
    }
    return sigaction(SIGPIPE, &ignored, NULL);
}

static int reset_child_signal_handlers(void) {
    const int signals[] = {SIGTERM, SIGHUP, SIGINT, SIGPIPE};
    struct sigaction action;
    size_t index;
    memset(&action, 0, sizeof(action));
    sigemptyset(&action.sa_mask);
    action.sa_handler = SIG_DFL;
    for (index = 0; index < sizeof(signals) / sizeof(signals[0]); ++index) {
        if (sigaction(signals[index], &action, NULL) != 0) {
            return -1;
        }
    }
    return 0;
}

static int enable_child_subreaper(int *previous_state) {
    int current = 0;
    if (previous_state == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (prctl(PR_GET_CHILD_SUBREAPER, &current, 0, 0, 0) != 0 ||
        (current != 0 && current != 1)) {
        return -1;
    }
    *previous_state = current;
    if (current == 0 && prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0) {
        return -1;
    }
    return 0;
}

static int restore_child_subreaper(int previous_state) {
    if (previous_state != 0 && previous_state != 1) {
        errno = EINVAL;
        return -1;
    }
    return prctl(PR_SET_CHILD_SUBREAPER, previous_state, 0, 0, 0);
}

static int observe_parent_cancellation(struct run_state *run) {
    if (parent_cancel_signal == 0) {
        return 0;
    }
    run->internal_failure = 1;
    set_failure(run, "parent_cancelled");
    return -1;
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
    if (errno != 0 || end == text || *end != '\0' || !isfinite(celsius) ||
        celsius < 1.0 || celsius > 200.0) {
        return -1;
    }
    *value = (int)(celsius * 1000.0 + 0.5);
    return 0;
}

static int parse_ratio(const char *text, double *value) {
    char *end = NULL;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) ||
        parsed <= 0.0 || parsed > 1.0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static const struct event_group_definition *find_event_group(const char *name) {
    size_t index;
    for (index = 0; index < EVENT_GROUP_COUNT; ++index) {
        if (strcmp(EVENT_GROUPS[index].name, name) == 0) {
            return &EVENT_GROUPS[index];
        }
    }
    return NULL;
}

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "Usage: %s [options] -- command [args...]\n"
            "\n"
            "Options:\n"
            "  -o, --output PATH          create JSON result safely (must not exist)\n"
            "      --child-stdout PATH    create child stdout safely (paired)\n"
            "      --child-stderr PATH    create child stderr safely (paired)\n"
            "  -u, --user NAME            drop child to NAME (default orangepi)\n"
            "  -g, --group NAME           drop child to GROUP (default orangepi)\n"
            "      --event-group NAME     core, cache, or memory (default core)\n"
            "      --min-running-ratio R  validity threshold (default 0.95)\n"
            "      --no-drop              keep current uid/gid (host tests only)\n"
            "      --start-on-ready       S -> atomic enable -> ACK -> work -> E\n"
            "      --start-immediately    enable after child setup gate\n"
            "      --sync-timeout-ms N    marker timeout (default 30000)\n"
            "      --max-temp-c C         stop at thermal threshold (default 85)\n"
            "      --thermal-root PATH    thermal sysfs root (testability)\n"
            "      --no-thermal-guard     disable target thermal guard\n"
            "  -h, --help                 show this help\n"
            "\n"
            "Exact mode uses fixed child fds: marker=9, ACK=8. PMU counts are\n"
            "event counts, never DDR bytes. Each run contains one PMU group\n"
            "bounded by a conservative software limit, not hardware capacity.\n",
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
        {"event-group", required_argument, NULL, 1006},
        {"min-running-ratio", required_argument, NULL, 1007},
        {"thermal-root", required_argument, NULL, 1008},
        {"child-stdout", required_argument, NULL, 1009},
        {"child-stderr", required_argument, NULL, 1010},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    int option_index = 0;

    options->output_path = "pmu-result.json";
    options->child_stdout_path = NULL;
    options->child_stderr_path = NULL;
    options->user_name = DEFAULT_USER;
    options->group_name = DEFAULT_GROUP;
    options->thermal_root = "/sys/class/thermal";
    options->event_group = &EVENT_GROUPS[0];
    options->no_drop = 0;
    options->start_on_ready = 1;
    options->sync_timeout_ms = DEFAULT_TIMEOUT_MS;
    options->max_temp_mc = DEFAULT_MAX_TEMP_MC;
    options->thermal_guard = 1;
    options->min_running_ratio = DEFAULT_MIN_RUNNING_RATIO;
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
        case 1006:
            options->event_group = find_event_group(optarg);
            if (options->event_group == NULL) {
                fprintf(stderr, "invalid --event-group: %s\n", optarg);
                return -1;
            }
            break;
        case 1007:
            if (parse_ratio(optarg, &options->min_running_ratio) != 0) {
                fprintf(stderr, "invalid --min-running-ratio: %s\n", optarg);
                return -1;
            }
            break;
        case 1008:
            options->thermal_root = optarg;
            break;
        case 1009:
            options->child_stdout_path = optarg;
            break;
        case 1010:
            options->child_stderr_path = optarg;
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
    if ((options->child_stdout_path == NULL) !=
        (options->child_stderr_path == NULL)) {
        fprintf(stderr, "--child-stdout and --child-stderr must be supplied together\n");
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
    flags = enabled ? (flags | FD_CLOEXEC) : (flags & ~FD_CLOEXEC);
    return fcntl(fd, F_SETFD, flags);
}

static int install_sync_fds(int marker_write_fd, int ack_read_fd) {
    int marker_copy = -1;
    int ack_copy = -1;

    /*
     * Copy both pipes above the fixed range before touching fd 8 or 9.  The
     * original pipe ends can themselves be 8/9 when the caller inherited
     * extra descriptors, so sequential dup2 calls would destroy one source.
     */
    ack_copy = fcntl(ack_read_fd, F_DUPFD_CLOEXEC, 10);
    if (ack_copy < 0) {
        return -1;
    }
    marker_copy = fcntl(marker_write_fd, F_DUPFD_CLOEXEC, 10);
    if (marker_copy < 0) {
        close(ack_copy);
        return -1;
    }
    if (ack_read_fd != ACK_FD) {
        close(ack_read_fd);
    }
    if (marker_write_fd != MARKER_FD) {
        close(marker_write_fd);
    }
    if (dup2(ack_copy, ACK_FD) < 0 || dup2(marker_copy, MARKER_FD) < 0) {
        close(ack_copy);
        close(marker_copy);
        return -1;
    }
    close(ack_copy);
    close(marker_copy);
    return set_fd_cloexec(ACK_FD, 0) == 0 && set_fd_cloexec(MARKER_FD, 0) == 0
               ? 0
               : -1;
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
    group_entry = getgrnam(options->group_name);
    if (passwd_entry == NULL || group_entry == NULL) {
        fprintf(stderr, "cannot resolve child uid/gid\n");
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

static int64_t read_max_temp_mc(const char *thermal_root, int *readable) {
    DIR *directory;
    struct dirent *entry;
    int64_t maximum = -1;
    int samples = 0;

    *readable = 0;
    directory = opendir(thermal_root);
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
        if (snprintf(path, sizeof(path), "%s/%s/temp", thermal_root, entry->d_name) >=
            (int)sizeof(path)) {
            continue;
        }
        file = fopen(path, "r");
        if (file == NULL) {
            continue;
        }
        if (fscanf(file, "%ld", &value) == 1) {
            samples += 1;
            if (value > maximum) {
                maximum = value;
            }
        }
        fclose(file);
    }
    closedir(directory);
    if (samples > 0) {
        *readable = 1;
    }
    return maximum;
}

static int check_thermal(struct run_state *run, const struct options *options) {
    int readable = 0;
    int64_t temperature;
    if (!options->thermal_guard) {
        return 0;
    }
    run->thermal_checked = 1;
    temperature = read_max_temp_mc(options->thermal_root, &readable);
    run->thermal_readable = readable;
    if (!readable || temperature < 0) {
        run->thermal_unreadable = 1;
        run->internal_failure = 1;
        set_failure(run, "thermal_unreadable");
        return -1;
    }
    if (temperature > run->max_temp_mc) {
        run->max_temp_mc = temperature;
    }
    if (temperature > options->max_temp_mc) {
        run->thermal_tripped = 1;
        run->internal_failure = 1;
        set_failure(run, "thermal_guard_exceeded");
        return -1;
    }
    return 0;
}

static void init_event_states(struct run_state *run,
                              const struct event_group_definition *group) {
    size_t index;
    run->event_count = group->event_count;
    run->group_leader_fd = -1;
    run->supported_events = 0;
    for (index = 0; index < group->event_count; ++index) {
        struct event_state *event = &run->events[index];
        memset(event, 0, sizeof(*event));
        event->definition = &EVENT_DEFINITIONS[group->event_indices[index]];
        event->fd = -1;
    }
}

static void close_event_fds(struct run_state *run) {
    size_t index;
    for (index = 0; index < run->event_count; ++index) {
        if (run->events[index].fd >= 0) {
            close(run->events[index].fd);
            run->events[index].fd = -1;
        }
    }
    run->group_leader_fd = -1;
}

static size_t open_event_group(struct run_state *run) {
    size_t index;
    int leader = -1;
    int first_errno = 0;

    for (index = 0; index < run->event_count; ++index) {
        struct perf_event_attr attr;
        struct event_state *event = &run->events[index];
        int fd;
        memset(&attr, 0, sizeof(attr));
        attr.type = PERF_TYPE_RAW;
        attr.size = sizeof(attr);
        attr.config = event->definition->config;
        attr.disabled = index == 0 ? 1U : 0U;
        attr.inherit = 1;
        attr.inherit_stat = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fd = perf_event_open_local(&attr, run->child_pid, -1, leader,
                                   PERF_FLAG_FD_CLOEXEC);
        if (fd < 0) {
            first_errno = errno;
            event->open_errno = errno;
            if (index > 0) {
                run->group_open_failed = 1;
                run->internal_failure = 1;
                set_failure(run, "event_group_open_failed");
            }
            break;
        }
        event->fd = fd;
        event->opened = 1;
        run->supported_events += 1;
        if (index == 0) {
            leader = fd;
            run->group_leader_fd = fd;
        }
    }

    if (run->supported_events != run->event_count) {
        for (index = 0; index < run->event_count; ++index) {
            if (!run->events[index].opened && run->events[index].open_errno == 0) {
                run->events[index].open_errno = first_errno != 0 ? first_errno : ECANCELED;
            }
        }
        if (run->supported_events > 0) {
            close_event_fds(run);
            for (index = 0; index < run->event_count; ++index) {
                run->events[index].opened = 0;
            }
            run->supported_events = 0;
        }
    }
    return run->supported_events;
}

static int reset_enable_group(struct run_state *run) {
    if (run->group_leader_fd < 0) {
        return 0;
    }
    if (ioctl(run->group_leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(run->group_leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        run->internal_failure = 1;
        set_failure(run, "event_group_enable_failed");
        return -1;
    }
    run->group_enabled = 1;
    return 1;
}

static void disable_group(struct run_state *run) {
    if (run->group_enabled && run->group_leader_fd >= 0) {
        if (ioctl(run->group_leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
            run->internal_failure = 1;
            set_failure(run, "event_group_disable_failed");
        }
    }
    run->group_enabled = 0;
}

static void read_events(struct run_state *run, const struct options *options) {
    size_t index;
    int all_valid = run->supported_events == run->event_count && run->event_count > 0;
    for (index = 0; index < run->event_count; ++index) {
        struct event_state *event = &run->events[index];
        struct {
            uint64_t value;
            uint64_t time_enabled;
            uint64_t time_running;
        } sample;
        ssize_t bytes;
        if (!event->opened || event->fd < 0) {
            all_valid = 0;
            continue;
        }
        memset(&sample, 0, sizeof(sample));
        do {
            bytes = read(event->fd, &sample, sizeof(sample));
        } while (bytes < 0 && errno == EINTR);
        if (bytes != (ssize_t)sizeof(sample)) {
            event->read_errno = bytes < 0 ? errno : EIO;
            run->read_failed = 1;
            run->internal_failure = 1;
            set_failure(run, "event_read_failed");
            all_valid = 0;
            continue;
        }
        event->value = sample.value;
        event->time_enabled = sample.time_enabled;
        event->time_running = sample.time_running;
        event->running_ratio = sample.time_enabled == 0
                                   ? 0.0
                                   : (double)sample.time_running / (double)sample.time_enabled;
        event->sample_valid = sample.time_enabled > 0 && sample.time_running > 0 &&
                              event->running_ratio >= options->min_running_ratio;
        if (!event->sample_valid) {
            all_valid = 0;
            set_failure(run, "event_sample_invalid");
        }
    }
    run->sample_valid = all_valid;
}

static int wait_for_marker(struct run_state *run, const struct options *options,
                           char wanted) {
    const uint64_t deadline = monotonic_ns() +
                              (uint64_t)options->sync_timeout_ms * 1000000ULL;
    struct pollfd descriptor;
    int child_exited = 0;
    descriptor.fd = run->marker_read_fd;
    descriptor.events = POLLIN | POLLHUP | POLLERR;

    for (;;) {
        int status;
        pid_t waited;
        int timeout = 100;
        uint64_t now = monotonic_ns();
        if (observe_parent_cancellation(run) != 0) {
            return -1;
        }
        if (now >= deadline) {
            run->timed_out = 1;
            run->sync_failed = 1;
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
            child_exited = 1;
        } else if (waited < 0 && errno != EINTR) {
            run->sync_failed = 1;
            set_failure(run, "waitpid_failed");
            return -1;
        }
        if (check_thermal(run, options) != 0) {
            return -1;
        }
        if (poll(&descriptor, 1, timeout) < 0) {
            if (errno == EINTR) {
                if (observe_parent_cancellation(run) != 0) {
                    return -1;
                }
                continue;
            }
            run->sync_failed = 1;
            set_failure(run, "sync_poll_failed");
            return -1;
        }
        if (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) {
            char marker;
            ssize_t bytes;
            do {
                bytes = read(run->marker_read_fd, &marker, 1);
            } while (bytes < 0 && errno == EINTR);
            if (bytes == 1 && marker == wanted) {
                return 0;
            }
            run->sync_failed = 1;
            if (bytes == 0) {
                set_failure(run, "sync_pipe_closed");
            } else if (bytes == 1) {
                set_failure(run, "unexpected_sync_marker");
            } else {
                set_failure(run, "sync_read_failed");
            }
            return -1;
        }
        if (child_exited) {
            run->sync_failed = 1;
            set_failure(run, wanted == 'S' ? "child_exited_before_start_marker"
                                          : "child_exited_before_end_marker");
            return -1;
        }
    }
}

static int send_ack(struct run_state *run) {
    static const char ack[] = "A\n";
    size_t offset = 0;
    while (offset < sizeof(ack) - 1U) {
        ssize_t written = write(run->ack_write_fd, ack + offset,
                                sizeof(ack) - 1U - offset);
        if (written < 0 && errno == EINTR) {
            if (observe_parent_cancellation(run) != 0) {
                return -1;
            }
            continue;
        }
        if (written <= 0) {
            run->sync_failed = 1;
            set_failure(run, "ack_write_failed");
            return -1;
        }
        offset += (size_t)written;
    }
    run->acknowledged = 1;
    return 0;
}

static int wait_for_child(struct run_state *run, const struct options *options) {
    uint64_t last_thermal = monotonic_ns();
    if (run->child_status_valid) {
        return 0;
    }
    for (;;) {
        int status;
        if (observe_parent_cancellation(run) != 0) {
            return -1;
        }
        pid_t waited = waitpid(run->child_pid, &status, WNOHANG);
        if (waited == run->child_pid) {
            run->child_status_valid = 1;
            run->child_status = status;
            return 0;
        }
        if (waited < 0) {
            if (errno == EINTR) {
                if (observe_parent_cancellation(run) != 0) {
                    return -1;
                }
                continue;
            }
            run->internal_failure = 1;
            set_failure(run, "waitpid_failed");
            return -1;
        }
        if (monotonic_ns() - last_thermal >= 100000000ULL) {
            last_thermal = monotonic_ns();
            if (check_thermal(run, options) != 0) {
                return -1;
            }
        }
        (void)poll(NULL, 0, 20);
    }
}

static int parse_proc_directory_pid(const char *name, pid_t *pid) {
    char *end = NULL;
    long parsed;
    if (name == NULL || name[0] < '1' || name[0] > '9') {
        return 0;
    }
    errno = 0;
    parsed = strtol(name, &end, 10);
    if (errno != 0 || end == name || *end != '\0' || parsed <= 0 ||
        parsed > INT_MAX) {
        return 0;
    }
    *pid = (pid_t)parsed;
    return 1;
}

/*
 * Return 1 for a stable parse, 0 when the task raced away, and -1 for an
 * unreadable or malformed proc record.  Field 22 (start_time) is the PID-reuse
 * identity paired with a pidfd before any signal is sent.
 */
static int read_proc_identity(pid_t pid, struct proc_identity *identity) {
    char path[64];
    char line[4096];
    char *cursor;
    char *end;
    char *closing;
    long parsed_pid;
    long long value = 0;
    size_t used = 0;
    int fd;
    int field;

    if (pid <= 0 || identity == NULL ||
        snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid) >=
            (int)sizeof(path)) {
        errno = EINVAL;
        return -1;
    }
    fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        return errno == ENOENT || errno == ESRCH ? 0 : -1;
    }
    for (;;) {
        ssize_t bytes = read(fd, line + used, sizeof(line) - 1U - used);
        if (bytes < 0 && errno == EINTR) {
            continue;
        }
        if (bytes < 0) {
            int saved_errno = errno;
            close(fd);
            errno = saved_errno;
            return saved_errno == ENOENT || saved_errno == ESRCH ? 0 : -1;
        }
        if (bytes == 0) {
            break;
        }
        used += (size_t)bytes;
        if (used == sizeof(line) - 1U) {
            close(fd);
            errno = EOVERFLOW;
            return -1;
        }
    }
    if (close(fd) != 0) {
        return -1;
    }
    if (used == 0) {
        return 0;
    }
    line[used] = '\0';

    errno = 0;
    parsed_pid = strtol(line, &end, 10);
    if (errno != 0 || end == line || *end != ' ' || end[1] != '(' ||
        parsed_pid != (long)pid) {
        errno = EPROTO;
        return -1;
    }
    closing = strrchr(end + 2, ')');
    if (closing == NULL || closing[1] != ' ' || closing[2] == '\0' ||
        closing[2] == ' ' || closing[3] != ' ') {
        errno = EPROTO;
        return -1;
    }
    cursor = closing + 4;
    memset(identity, 0, sizeof(*identity));
    identity->pid = pid;
    identity->pidfd = -1;
    for (field = 4; field <= 22; ++field) {
        errno = 0;
        if (field == 22) {
            unsigned long long start_time = strtoull(cursor, &end, 10);
            if (errno == 0) {
                identity->start_time = (uint64_t)start_time;
            }
        } else {
            value = strtoll(cursor, &end, 10);
        }
        if (errno != 0 || end == cursor ||
            (field < 22 && *end != ' ') ||
            (field == 22 && *end != ' ' && *end != '\n' && *end != '\0')) {
            errno = EPROTO;
            return -1;
        }
        if (field >= 4 && field <= 6 && (value < 0 || value > INT_MAX)) {
            errno = EPROTO;
            return -1;
        }
        if (field == 4) identity->parent_pid = (pid_t)value;
        if (field == 5) identity->process_group = (pid_t)value;
        if (field == 6) identity->session = (pid_t)value;
        cursor = end;
        while (*cursor == ' ') {
            ++cursor;
        }
    }
    return 1;
}

static int pidfd_open_local(pid_t pid) {
#ifdef SYS_pidfd_open
    return (int)syscall(SYS_pidfd_open, pid, 0U);
#else
    (void)pid;
    errno = ENOSYS;
    return -1;
#endif
}

static int pidfd_send_signal_local(int pidfd, int signal_number) {
#ifdef SYS_pidfd_send_signal
    return (int)syscall(SYS_pidfd_send_signal, pidfd, signal_number, NULL, 0U);
#else
    (void)pidfd;
    (void)signal_number;
    errno = ENOSYS;
    return -1;
#endif
}

static void close_identity_list(struct proc_identity *children, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        if (children[index].pidfd >= 0) {
            close(children[index].pidfd);
            children[index].pidfd = -1;
        }
    }
}

static int enumerate_adopted_children(struct proc_identity *children,
                                      size_t capacity, size_t *count) {
    DIR *directory;
    struct dirent *entry;
    const pid_t wrapper_pid = getpid();
    size_t found = 0;
    int scan_errno = 0;

    *count = 0;
    directory = opendir("/proc");
    if (directory == NULL) {
        return -1;
    }
    errno = 0;
    while ((entry = readdir(directory)) != NULL) {
        struct proc_identity first;
        struct proc_identity confirmed;
        pid_t pid;
        int read_status;
        int pidfd;
        if (!parse_proc_directory_pid(entry->d_name, &pid)) {
            errno = 0;
            continue;
        }
        read_status = read_proc_identity(pid, &first);
        if (read_status == 0) {
            errno = 0;
            continue;
        }
        if (read_status < 0) {
            scan_errno = errno != 0 ? errno : EPROTO;
            break;
        }
        if (first.parent_pid != wrapper_pid) {
            errno = 0;
            continue;
        }
        if (first.start_time == 0) {
            scan_errno = EPROTO;
            break;
        }
        if (found == capacity) {
            scan_errno = EOVERFLOW;
            break;
        }
        pidfd = pidfd_open_local(pid);
        if (pidfd < 0) {
            if (errno == ESRCH || errno == ENOENT) {
                errno = 0;
                continue;
            }
            scan_errno = errno != 0 ? errno : ENOSYS;
            break;
        }
        read_status = read_proc_identity(pid, &confirmed);
        if (read_status == 0) {
            close(pidfd);
            errno = 0;
            continue;
        }
        if (read_status < 0) {
            scan_errno = errno != 0 ? errno : EPROTO;
            close(pidfd);
            break;
        }
        if (confirmed.parent_pid != wrapper_pid || confirmed.start_time == 0 ||
            confirmed.start_time != first.start_time) {
            close(pidfd);
            errno = 0;
            continue;
        }
        confirmed.pidfd = pidfd;
        children[found++] = confirmed;
        errno = 0;
    }
    if (entry == NULL && errno != 0 && scan_errno == 0) {
        scan_errno = errno;
    }
    if (closedir(directory) != 0 && scan_errno == 0) {
        scan_errno = errno != 0 ? errno : EIO;
    }
    if (scan_errno != 0) {
        close_identity_list(children, found);
        errno = scan_errno;
        return -1;
    }
    *count = found;
    return 0;
}

static int reap_adopted_children_nonblocking(struct run_state *run) {
    for (;;) {
        int status;
        pid_t waited = waitpid(-1, &status, WNOHANG);
        if (waited > 0) {
            if (waited == run->child_pid) {
                run->child_status_valid = 1;
                run->child_status = status;
            }
            continue;
        }
        if (waited == 0 || errno == ECHILD) {
            return 0;
        }
        if (errno == EINTR) {
            continue;
        }
        return -1;
    }
}

/* Return 0 when quiescent, 1 when the bounded phase expires, and -1 on error. */
static int cleanup_adopted_children_phase(struct run_state *run,
                                          int signal_number, int attempts) {
    struct proc_identity *children;
    int empty_scans = 0;
    int attempt;
    children = calloc(MAX_ADOPTED_CHILDREN, sizeof(*children));
    if (children == NULL) {
        return -1;
    }
    for (attempt = 0; attempt < attempts; ++attempt) {
        size_t count = 0;
        size_t index;
        if (reap_adopted_children_nonblocking(run) != 0 ||
            enumerate_adopted_children(children, MAX_ADOPTED_CHILDREN, &count) != 0) {
            free(children);
            return -1;
        }
        if (count == 0) {
            ++empty_scans;
            if (empty_scans >= CLEANUP_EMPTY_CONFIRMATIONS) {
                free(children);
                return 0;
            }
        } else {
            empty_scans = 0;
            for (index = 0; index < count; ++index) {
                if (pidfd_send_signal_local(children[index].pidfd,
                                            signal_number) != 0 &&
                    errno != ESRCH) {
                    close_identity_list(children, count);
                    free(children);
                    return -1;
                }
            }
        }
        close_identity_list(children, count);
        (void)poll(NULL, 0, CLEANUP_POLL_MS);
    }
    free(children);
    return 1;
}

static int terminate_process_tree(struct run_state *run) {
    int phase_status;
    if (run->child_pid <= 0) {
        return 0;
    }
    phase_status = cleanup_adopted_children_phase(run, SIGTERM,
                                                   CLEANUP_TERM_ATTEMPTS);
    if (phase_status > 0) {
        phase_status = cleanup_adopted_children_phase(run, SIGKILL,
                                                       CLEANUP_KILL_ATTEMPTS);
    }
    if (phase_status != 0 || !run->child_status_valid) {
        run->internal_failure = 1;
        set_failure(run, "process_tree_cleanup_failed");
        return -1;
    }
    return 0;
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
    if (run->internal_failure || run->sync_failed || run->timed_out ||
        run->thermal_unreadable || run->thermal_tripped || run->read_failed ||
        run->group_open_failed || command_exit_code(run) != 0) {
        return "failed";
    }
    if (run->supported_events == 0) {
        return "counter_unavailable";
    }
    if (!run->sample_valid) {
        return "failed";
    }
    return "ok";
}

static const char *event_support(const struct event_state *event) {
    if (event->opened && event->read_errno == 0) {
        return "supported";
    }
    if (event->read_errno != 0) {
        return "read_error";
    }
    return "unavailable";
}

static void json_string(FILE *stream, const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    fputc('"', stream);
    while (*cursor != '\0') {
        switch (*cursor) {
        case '\\': fputs("\\\\", stream); break;
        case '"': fputs("\\\"", stream); break;
        case '\n': fputs("\\n", stream); break;
        case '\r': fputs("\\r", stream); break;
        case '\t': fputs("\\t", stream); break;
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

static void discard_new_output(const char *path, int *fd);

static int reserve_output_safely(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    struct stat status;
    int injected_failure = 0;
    if (fd < 0) {
        fprintf(stderr, "safe open(%s): %s\n", path, strerror(errno));
        return -1;
    }
#ifdef E049C_TESTING
    {
        const char *requested = getenv("E049C_TEST_FAIL_OUTPUT_VALIDATION");
        injected_failure = requested != NULL && strcmp(requested, path) == 0;
    }
#endif
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_nlink != 1 ||
        injected_failure) {
        fprintf(stderr, "unsafe output target: %s\n", path);
        discard_new_output(path, &fd);
        return -1;
    }
    return fd;
}

static int move_above_fixed_fds(int fd) {
    int replacement;
    if (fd > MARKER_FD) {
        return fd;
    }
    replacement = fcntl(fd, F_DUPFD_CLOEXEC, MARKER_FD + 1);
    if (replacement < 0) {
        return -1;
    }
    close(fd);
    return replacement;
}

static void discard_new_output(const char *path, int *fd) {
    struct stat opened;
    struct stat current;
    int same_file = 0;
    if (*fd < 0) {
        return;
    }
    if (fstat(*fd, &opened) == 0 && lstat(path, &current) == 0 &&
        S_ISREG(current.st_mode) && current.st_nlink == 1 &&
        opened.st_dev == current.st_dev && opened.st_ino == current.st_ino) {
        same_file = 1;
    }
    if (same_file && unlink(path) != 0) {
        fprintf(stderr, "unlink(%s): %s\n", path, strerror(errno));
    }
    close(*fd);
    *fd = -1;
}

static int redirect_child_streams(int child_stdout_fd, int child_stderr_fd) {
    if (child_stdout_fd < 0 && child_stderr_fd < 0) {
        return 0;
    }
    if (child_stdout_fd <= MARKER_FD || child_stderr_fd <= MARKER_FD) {
        errno = EINVAL;
        return -1;
    }
    if (dup2(child_stdout_fd, STDOUT_FILENO) < 0 ||
        dup2(child_stderr_fd, STDERR_FILENO) < 0) {
        return -1;
    }
    close(child_stdout_fd);
    close(child_stderr_fd);
    return 0;
}

static int write_result(const struct options *options, const struct run_state *run,
                        int argc, char **argv, int output_fd) {
    FILE *stream = fdopen(output_fd, "w");
    size_t index;
    if (stream == NULL) {
        fprintf(stderr, "fdopen(%s): %s\n", options->output_path, strerror(errno));
        close(output_fd);
        return -1;
    }
    fprintf(stream, "{\n  \"schema_version\": \"e049c-arm-pmu/v2\",\n");
    fprintf(stream, "  \"status\": ");
    json_string(stream, result_status(run));
    fprintf(stream, ",\n  \"sample_valid\": %s,\n",
            run->sample_valid && strcmp(result_status(run), "ok") == 0 ? "true" : "false");
    fprintf(stream, "  \"event_source\": \"armv8_pmuv3_raw_config\",\n");
    fprintf(stream, "  \"counter_semantics\": \"event counts only; no DDR-byte conversion\",\n");
    fprintf(stream, "  \"event_group\": ");
    json_string(stream, options->event_group->name);
    fprintf(stream, ",\n  \"software_group_size_limit\": %d,\n",
            SOFTWARE_GROUP_SIZE_LIMIT);
    fprintf(stream, "  \"event_group_size\": %zu,\n", run->event_count);
    fprintf(stream, "  \"software_group_size_limit_semantics\": ");
    json_string(stream,
                "conservative launcher policy; not measured hardware PMU capacity");
    fprintf(stream, ",\n");
    fprintf(stream, "  \"min_running_ratio\": %.6f,\n", options->min_running_ratio);
    fprintf(stream, "  \"pid\": %ld,\n  \"process_group\": %ld,\n",
            (long)run->child_pid, (long)run->child_pgid);
    fprintf(stream, "  \"command\": [");
    for (index = (size_t)options->command_index; index < (size_t)argc; ++index) {
        if (index != (size_t)options->command_index) {
            fputs(", ", stream);
        }
        json_string(stream, argv[index]);
    }
    fprintf(stream, "],\n  \"sync\": {\"mode\": ");
    json_string(stream, options->start_on_ready ? "start_ack_end"
                                                : "immediate_after_setup_gate");
    fprintf(stream, ", \"started\": %s, \"acknowledged\": %s, \"ended\": %s},\n",
            run->started ? "true" : "false", run->acknowledged ? "true" : "false",
            run->ended ? "true" : "false");
    fprintf(stream, "  \"measured_elapsed_ns\": %" PRIu64 ",\n",
            run->measured_elapsed_ns);
    fprintf(stream, "  \"thermal\": {\"guard_enabled\": %s, \"checked\": %s, "
                    "\"readable\": %s, \"limit_c\": %.3f, \"max_observed_c\": ",
            options->thermal_guard ? "true" : "false",
            run->thermal_checked ? "true" : "false",
            run->thermal_readable ? "true" : "false",
            (double)options->max_temp_mc / 1000.0);
    if (run->max_temp_mc >= 0) {
        fprintf(stream, "%.3f", (double)run->max_temp_mc / 1000.0);
    } else {
        fputs("null", stream);
    }
    fprintf(stream, ", \"tripped\": %s},\n", run->thermal_tripped ? "true" : "false");
    fprintf(stream, "  \"exit\": {\"code\": %d, \"raw_wait_status\": %d},\n",
            command_exit_code(run), run->child_status_valid ? run->child_status : -1);
    fprintf(stream, "  \"failure_reason\": ");
    if (run->failure_reason[0] == '\0') {
        fputs("null", stream);
    } else {
        json_string(stream, run->failure_reason);
    }
    fprintf(stream, ",\n  \"events\": [\n");
    for (index = 0; index < run->event_count; ++index) {
        const struct event_state *event = &run->events[index];
        const char *support = event_support(event);
        int error_number = event->read_errno != 0 ? event->read_errno : event->open_errno;
        if (index != 0) {
            fputs(",\n", stream);
        }
        fprintf(stream, "    {\"name\": ");
        json_string(stream, event->definition->name);
        fprintf(stream, ", \"config\": \"0x%llx\", \"meaning\": ",
                (unsigned long long)event->definition->config);
        json_string(stream, event->definition->meaning);
        fprintf(stream, ", \"support\": ");
        json_string(stream, support);
        fprintf(stream, ", \"count_semantics\": \"event_count_not_bytes\", "
                        "\"sample_valid\": %s, \"value\": ",
                event->sample_valid ? "true" : "false");
        if (strcmp(support, "supported") == 0) fprintf(stream, "%" PRIu64, event->value);
        else fputs("null", stream);
        fprintf(stream, ", \"time_enabled_ns\": ");
        if (strcmp(support, "supported") == 0) fprintf(stream, "%" PRIu64, event->time_enabled);
        else fputs("null", stream);
        fprintf(stream, ", \"time_running_ns\": ");
        if (strcmp(support, "supported") == 0) fprintf(stream, "%" PRIu64, event->time_running);
        else fputs("null", stream);
        fprintf(stream, ", \"running_ratio\": ");
        if (strcmp(support, "supported") == 0 && event->time_enabled > 0)
            fprintf(stream, "%.9f", event->running_ratio);
        else fputs("null", stream);
        fprintf(stream, ", \"errno\": %d, \"error\": ", error_number);
        if (error_number != 0) json_string(stream, strerror(error_number));
        else fputs("null", stream);
        fputs("}", stream);
    }
    fprintf(stream, "\n  ]\n}\n");
    if (fclose(stream) != 0) {
        fprintf(stderr, "fclose(%s): %s\n", options->output_path, strerror(errno));
        return -1;
    }
    return 0;
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

static int confirm_child_session(int fd, struct run_state *run) {
    char marker;
    ssize_t bytes;
    do {
        bytes = read(fd, &marker, 1);
    } while (bytes < 0 && errno == EINTR && parent_cancel_signal == 0);
    close(fd);
    if (bytes != 1 || marker != 'I' || parent_cancel_signal != 0) {
        run->internal_failure = 1;
        set_failure(run, parent_cancel_signal != 0 ? "parent_cancelled"
                                                   : "child_session_unconfirmed");
        return -1;
    }
    run->child_sid = run->child_pid;
    run->child_session_established = 1;
    return 0;
}

static int child_main(const struct options *options, char **command, int setup_read_fd,
                      int identity_write_fd, int marker_write_fd, int ack_read_fd,
                      int child_stdout_fd,
                      int child_stderr_fd, uid_t uid, gid_t gid,
                      pid_t expected_parent_pid) {
    const char identity_marker = 'I';
    char gate;
    ssize_t bytes;
    if (reset_child_signal_handlers() != 0 ||
        prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
        dprintf(STDERR_FILENO, "child lifecycle setup failed: %s\n", strerror(errno));
        return 126;
    }
    if (getppid() != expected_parent_pid) {
        (void)raise(SIGTERM);
        return 126;
    }
    if (setsid() < 0) {
        dprintf(STDERR_FILENO, "setsid failed: %s\n", strerror(errno));
        return 126;
    }
    do {
        bytes = write(identity_write_fd, &identity_marker, 1);
    } while (bytes < 0 && errno == EINTR);
    close(identity_write_fd);
    if (bytes != 1) {
        dprintf(STDERR_FILENO, "child identity confirmation failed: %s\n",
                strerror(errno));
        return 126;
    }
    if (redirect_child_streams(child_stdout_fd, child_stderr_fd) != 0) {
        dprintf(STDERR_FILENO, "child stream redirect failed: %s\n", strerror(errno));
        return 126;
    }
    if (options->start_on_ready) {
        if (install_sync_fds(marker_write_fd, ack_read_fd) != 0) {
            dprintf(STDERR_FILENO, "fixed sync fd setup failed: %s\n", strerror(errno));
            return 126;
        }
    }
    if (drop_child_identity(options, uid, gid) != 0) {
        dprintf(STDERR_FILENO, "drop child identity failed: %s\n", strerror(errno));
        return 126;
    }
    do {
        bytes = read(setup_read_fd, &gate, 1);
    } while (bytes < 0 && errno == EINTR);
    close(setup_read_fd);
    if (bytes != 1 || gate != 'G') {
        dprintf(STDERR_FILENO, "setup gate failed\n");
        return 126;
    }
    execvp(command[0], command);
    dprintf(STDERR_FILENO, "execvp(%s): %s\n", command[0], strerror(errno));
    return 127;
}

int main(int argc, char **argv) {
    struct options options;
    struct run_state run;
    uid_t child_uid;
    gid_t child_gid;
    int parse_status;
    int setup_pipe[2];
    int identity_pipe[2] = {-1, -1};
    int marker_pipe[2] = {-1, -1};
    int ack_pipe[2] = {-1, -1};
    pid_t child;
    char **command;
    uint64_t measured_start = 0;
    int failed = 0;
    int result_write_status;
    int final_code;
    int output_fd;
    int child_stdout_fd = -1;
    int child_stderr_fd = -1;
    int subreaper_previous = -1;
    pid_t parent_pid;

    memset(&run, 0, sizeof(run));
    run.max_temp_mc = -1;
    run.group_leader_fd = -1;
    run.marker_read_fd = -1;
    run.ack_write_fd = -1;
    parse_status = parse_options(argc, argv, &options);
    if (parse_status != 0) {
        return parse_status > 0 ? 0 : 2;
    }
    if (install_parent_signal_handlers() != 0) {
        fprintf(stderr, "cannot install parent signal handlers: %s\n", strerror(errno));
        return 2;
    }
    parent_pid = getpid();
    init_event_states(&run, options.event_group);
    if (set_child_identity(&options, &child_uid, &child_gid) != 0) {
        return 2;
    }
    output_fd = reserve_output_safely(options.output_path);
    if (output_fd < 0) {
        return 2;
    }
    if (options.child_stdout_path != NULL) {
        child_stdout_fd = reserve_output_safely(options.child_stdout_path);
        if (child_stdout_fd < 0) {
            discard_new_output(options.output_path, &output_fd);
            return 2;
        }
        {
            int moved = move_above_fixed_fds(child_stdout_fd);
            if (moved < 0) {
                discard_new_output(options.child_stdout_path, &child_stdout_fd);
                discard_new_output(options.output_path, &output_fd);
                return 2;
            }
            child_stdout_fd = moved;
        }
        child_stderr_fd = reserve_output_safely(options.child_stderr_path);
        if (child_stderr_fd < 0) {
            discard_new_output(options.child_stdout_path, &child_stdout_fd);
            discard_new_output(options.output_path, &output_fd);
            return 2;
        }
        {
            int moved = move_above_fixed_fds(child_stderr_fd);
            if (moved < 0) {
                discard_new_output(options.child_stderr_path, &child_stderr_fd);
                discard_new_output(options.child_stdout_path, &child_stdout_fd);
                discard_new_output(options.output_path, &output_fd);
                return 2;
            }
            child_stderr_fd = moved;
        }
    }
    if (pipe(setup_pipe) != 0 || pipe(identity_pipe) != 0 ||
        (options.start_on_ready && (pipe(marker_pipe) != 0 || pipe(ack_pipe) != 0))) {
        fprintf(stderr, "pipe setup failed: %s\n", strerror(errno));
        if (options.child_stderr_path != NULL) {
            discard_new_output(options.child_stderr_path, &child_stderr_fd);
            discard_new_output(options.child_stdout_path, &child_stdout_fd);
        }
        discard_new_output(options.output_path, &output_fd);
        return 2;
    }
    if (enable_child_subreaper(&subreaper_previous) != 0) {
        fprintf(stderr, "cannot enable child subreaper: %s\n", strerror(errno));
        if (options.child_stderr_path != NULL) {
            discard_new_output(options.child_stderr_path, &child_stderr_fd);
            discard_new_output(options.child_stdout_path, &child_stdout_fd);
        }
        discard_new_output(options.output_path, &output_fd);
        return 2;
    }
    command = &argv[options.command_index];
    child = fork();
    if (child < 0) {
        fprintf(stderr, "fork: %s\n", strerror(errno));
        if (restore_child_subreaper(subreaper_previous) != 0) {
            fprintf(stderr, "cannot restore child subreaper: %s\n", strerror(errno));
        }
        if (options.child_stderr_path != NULL) {
            discard_new_output(options.child_stderr_path, &child_stderr_fd);
            discard_new_output(options.child_stdout_path, &child_stdout_fd);
        }
        discard_new_output(options.output_path, &output_fd);
        return 2;
    }
    run.child_pid = child;
    run.child_pgid = child;
    run.sync_enabled = options.start_on_ready;

    if (child == 0) {
        int status;
        close(output_fd);
        close(setup_pipe[1]);
        close(identity_pipe[0]);
        if (options.start_on_ready) {
            close(marker_pipe[0]);
            close(ack_pipe[1]);
            status = child_main(&options, command, setup_pipe[0], identity_pipe[1],
                                marker_pipe[1], ack_pipe[0], child_stdout_fd, child_stderr_fd,
                                child_uid, child_gid, parent_pid);
        } else {
            status = child_main(&options, command, setup_pipe[0], identity_pipe[1], -1, -1,
                                child_stdout_fd, child_stderr_fd, child_uid, child_gid,
                                parent_pid);
        }
        _exit(status);
    }

    if (child_stdout_fd >= 0) close(child_stdout_fd);
    if (child_stderr_fd >= 0) close(child_stderr_fd);
    close(setup_pipe[0]);
    close(identity_pipe[1]);
    if (options.start_on_ready) {
        close(marker_pipe[1]);
        close(ack_pipe[0]);
        run.marker_read_fd = marker_pipe[0];
        run.ack_write_fd = ack_pipe[1];
    }
    (void)set_fd_cloexec(setup_pipe[1], 1);
    if (confirm_child_session(identity_pipe[0], &run) != 0) {
        failed = 1;
    }
    (void)open_event_group(&run);

    if (!failed && check_thermal(&run, &options) != 0) {
        failed = 1;
    } else if (!failed && release_setup_gate(setup_pipe[1]) != 0) {
        run.internal_failure = 1;
        set_failure(&run, "parent_setup_gate_failed");
        failed = 1;
    } else if (failed) {
        close(setup_pipe[1]);
    }

    if (!failed && options.start_on_ready) {
        if (wait_for_marker(&run, &options, 'S') != 0) {
            failed = 1;
        } else {
            run.started = 1;
            if (reset_enable_group(&run) < 0) {
                failed = 1;
            } else {
                measured_start = monotonic_ns();
                if (send_ack(&run) != 0) {
                    failed = 1;
                } else if (wait_for_marker(&run, &options, 'E') != 0) {
                    failed = 1;
                } else {
                    run.ended = 1;
                    disable_group(&run);
                    run.measured_elapsed_ns = monotonic_ns() - measured_start;
                    run.measured_elapsed_valid = 1;
                    if (wait_for_child(&run, &options) != 0) {
                        failed = 1;
                    } else if (command_exit_code(&run) != 0) {
                        run.internal_failure = 1;
                        set_failure(&run, "child_exit_nonzero");
                        failed = 1;
                    }
                }
            }
        }
    } else if (!failed) {
        if (reset_enable_group(&run) < 0) {
            failed = 1;
        } else {
            run.started = 1;
            measured_start = monotonic_ns();
            if (wait_for_child(&run, &options) != 0) {
                failed = 1;
            } else {
                run.ended = 1;
                if (command_exit_code(&run) != 0) {
                    run.internal_failure = 1;
                    set_failure(&run, "child_exit_nonzero");
                    failed = 1;
                }
            }
            disable_group(&run);
            run.measured_elapsed_ns = monotonic_ns() - measured_start;
            run.measured_elapsed_valid = 1;
        }
    }

    if (observe_parent_cancellation(&run) != 0) {
        failed = 1;
    }

    if (failed) {
        disable_group(&run);
        if (measured_start != 0 && run.measured_elapsed_ns == 0) {
            run.measured_elapsed_ns = monotonic_ns() - measured_start;
        }
    }
    if (terminate_process_tree(&run) != 0) {
        failed = 1;
    }
    if (restore_child_subreaper(subreaper_previous) != 0) {
        run.internal_failure = 1;
        set_failure(&run, "subreaper_restore_failed");
        failed = 1;
    }
    if (run.marker_read_fd >= 0) close(run.marker_read_fd);
    if (run.ack_write_fd >= 0) close(run.ack_write_fd);
    read_events(&run, &options);
    if (!run.started || (options.start_on_ready &&
                         (!run.acknowledged || !run.ended || run.sync_failed)) ||
        command_exit_code(&run) != 0 || failed) {
        run.sample_valid = 0;
    }
    final_code = command_exit_code(&run);
    result_write_status = write_result(&options, &run, argc, argv, output_fd);
    close_event_fds(&run);
    if (result_write_status != 0) {
        return parent_cancel_signal != 0 ? 128 + parent_cancel_signal : 2;
    }
    if (parent_cancel_signal != 0) {
        return 128 + parent_cancel_signal;
    }
    if (strcmp(result_status(&run), "failed") == 0) {
        return final_code != 0 ? final_code : 2;
    }
    return final_code;
}
