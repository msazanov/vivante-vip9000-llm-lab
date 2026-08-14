#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 null|candidate OUTPUT_ROOT" >&2
    exit 2
fi

gate=$1
output_root=$2
case "$gate" in
    null) comparison=null ;;
    candidate) comparison=candidate ;;
    *) echo "gate must be null or candidate" >&2; exit 2 ;;
esac

lab=/home/orangepi/vip9000-lab
runner=$lab/build/e054-q1-runner/q1_cpu_operator_runner
fixture=/tmp/e054-blk0-ffn-gate-seed1843-v1
thermal=$lab/thermal_exec_guard.py
profiler=$lab/profile_command.py

[[ -x "$runner" ]]
[[ -f "$fixture/fixture.json" ]]
[[ -f "$thermal" && -f "$profiler" ]]
[[ ! -e "$output_root" ]]
mkdir -p "$output_root"

run_variant() {
    local pair=$1
    local variant=$2
    local run_id
    run_id=$(printf 'e054-%s-p%02d-%s' "$gate" "$pair" "$variant")
    local run_dir=$output_root/$run_id
    mkdir -p "$run_dir"

    set +e
    GGML_Q1_A76_DISPATCH="$variant" \
        python3 "$thermal" \
            --limit-mc 85000 \
            --interval-ms 20 \
            --trace "$run_dir/thermal.jsonl" \
            -- \
        python3 "$profiler" \
            --output-dir "$run_dir/profile/$run_id" \
            --run-id "$run_id" \
            --interval-ms 20 \
            --label "E054 $gate pair $pair $variant" \
            --phase-file phases.jsonl \
            -- \
        taskset -c 6-7 "$runner" \
            --run-id "$run_id" \
            --weights "$fixture/weights.q1_0.bin" \
            --activation "$fixture/activation.f32.bin" \
            --ne0 5120 \
            --ne1 17408 \
            --threads 2 \
            --warmup 1 \
            --iterations 50 \
            --output-jsonl "$run_dir/runner.jsonl" \
            --output-f32 "$run_dir/output.f32.bin" \
            --output-q8 "$run_dir/activation.q8_0.bin" \
        > "$run_dir/wrapper.stdout.txt" \
        2> "$run_dir/wrapper.stderr.txt"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$run_dir/exitcode"
    if [[ $rc -ne 0 ]]; then
        echo "$run_id failed with $rc" >&2
        return "$rc"
    fi
}

for pair in 1 2 3 4 5; do
    if (( pair % 2 == 1 )); then
        run_variant "$pair" stock
        sleep 1
        run_variant "$pair" "$comparison"
    else
        run_variant "$pair" "$comparison"
        sleep 1
        run_variant "$pair" stock
    fi
    sleep 1
done
