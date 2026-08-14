#!/bin/sh
# Mark the exact measured interval for a733_pmu_exec.c, then preserve the
# workload exit status. The launcher supplies A733_PMU_SYNC_FD to the child.
set -u

if [ -z "${A733_PMU_SYNC_FD:-}" ]; then
    echo "A733_PMU_SYNC_FD is required" >&2
    exit 125
fi

# The descriptor is inherited across the uid drop.  `eval` is intentional:
# the launcher supplies only a decimal fd and dash cannot parse >&"$var".
eval "printf 'S' >&$A733_PMU_SYNC_FD"
"$@"
status=$?
eval "printf 'E' >&$A733_PMU_SYNC_FD"
exit "$status"
