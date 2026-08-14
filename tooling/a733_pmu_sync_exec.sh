#!/bin/sh
# Fixed-descriptor S -> ACK -> workload -> E adapter for a733_pmu_exec.c.
set -u

if ! printf 'S' >&9; then
    echo "could not write start marker on fd 9" >&2
    exit 125
fi

if ! IFS= read -r ack <&8; then
    echo "could not read acknowledgement on fd 8" >&2
    exit 125
fi
if [ "$ack" != "A" ]; then
    echo "unexpected acknowledgement on fd 8" >&2
    exit 125
fi

"$@"
status=$?
if ! printf 'E' >&9; then
    echo "could not write end marker on fd 9" >&2
    exit 125
fi
exit "$status"
