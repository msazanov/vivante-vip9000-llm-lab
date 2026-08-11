#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=""
NETWORK=""
WEIGHTS=""
ACTIVATION=""
GOLDEN=""
REMOTE="orangepi@192.168.31.117"
REMOTE_DIR="/home/orangepi/vip9000-lab/e022-fused/manual"
RUNNER="/home/orangepi/vip9000-lab/bin/profiled_viplite_runner"
ITERATIONS=100
DTYPE="i32"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --network) NETWORK=$2; shift 2 ;;
        --weights) WEIGHTS=$2; shift 2 ;;
        --activation) ACTIVATION=$2; shift 2 ;;
        --golden) GOLDEN=$2; shift 2 ;;
        --remote) REMOTE=$2; shift 2 ;;
        --remote-dir) REMOTE_DIR=$2; shift 2 ;;
        --runner) RUNNER=$2; shift 2 ;;
        --iterations) ITERATIONS=$2; shift 2 ;;
        --dtype) DTYPE=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$OUTPUT_DIR" ] || [ -z "$NETWORK" ] || [ -z "$WEIGHTS" ] ||
   [ -z "$ACTIVATION" ] || [ -z "$GOLDEN" ]; then
    echo "usage: $0 --output-dir DIR --network NBG --weights BIN --activation BIN --golden JSON [--dtype i32|f32]" >&2
    exit 2
fi
case "$DTYPE" in i32|f32) ;; *) echo "--dtype must be i32 or f32" >&2; exit 2 ;; esac
case "$ITERATIONS" in ''|*[!0-9]*) echo "--iterations must be an integer" >&2; exit 2 ;; esac
mkdir -p "$OUTPUT_DIR"
for path in "$NETWORK" "$WEIGHTS" "$ACTIVATION" "$GOLDEN"; do
    [ -f "$path" ] || { echo "missing input: $path" >&2; exit 2; }
done

ssh -F /dev/null -o BatchMode=yes "$REMOTE" "mkdir -p '$REMOTE_DIR'"
scp -F /dev/null -o BatchMode=yes "$NETWORK" "$WEIGHTS" "$ACTIVATION" \
    "$REMOTE:$REMOTE_DIR/" >/dev/null
ssh -F /dev/null -o BatchMode=yes "$REMOTE" "set -e; mkdir -p '$REMOTE_DIR'; for d in /sys/class/thermal/thermal_zone*; do printf '%s type=' \"\$d\"; cat \"\$d/type\"; printf ' temp='; awk '{printf \"%.3f C\\n\", \$1/1000}' \"\$d/temp\"; done" \
    >"$OUTPUT_DIR/thermal_before.txt"
ssh -F /dev/null -o BatchMode=yes "$REMOTE" "set -e; cd '$REMOTE_DIR'; '$RUNNER' --iterations '$ITERATIONS' '$(basename "$NETWORK")' '$(basename "$WEIGHTS")' '$(basename "$ACTIVATION")' output.bin > profile.log" \
    >/dev/null
scp -F /dev/null -o BatchMode=yes "$REMOTE:$REMOTE_DIR/profile.log" "$OUTPUT_DIR/profile.log" >/dev/null
scp -F /dev/null -o BatchMode=yes "$REMOTE:$REMOTE_DIR/output.bin" "$OUTPUT_DIR/output.bin" >/dev/null
ssh -F /dev/null -o BatchMode=yes "$REMOTE" "set -e; for d in /sys/class/thermal/thermal_zone*; do printf '%s type=' \"\$d\"; cat \"\$d/type\"; printf ' temp='; awk '{printf \"%.3f C\\n\", \$1/1000}' \"\$d/temp\"; done" \
    >"$OUTPUT_DIR/thermal_after.txt"

python3 - "$OUTPUT_DIR/summary.json" "$GOLDEN" "$OUTPUT_DIR/output.bin" \
    "$OUTPUT_DIR/profile.log" "$OUTPUT_DIR/thermal_before.txt" "$OUTPUT_DIR/thermal_after.txt" "$DTYPE" <<'PY'
import json
import re
import struct
import sys
from pathlib import Path

summary_path, golden_path, output_path, profile_path, before_path, after_path, dtype = sys.argv[1:]
expected = json.loads(Path(golden_path).read_text(encoding="utf-8"))
data = Path(output_path).read_bytes()
if dtype == "i32":
    if len(data) % 4:
        raise SystemExit("INT32 output is not 4-byte aligned")
    raw = list(struct.unpack("<" + "i" * (len(data) // 4), data))
    selected = raw[::4]
    error = None
    exact = selected == expected
else:
    if len(data) % 4:
        raise SystemExit("FLOAT32 output is not 4-byte aligned")
    raw = list(struct.unpack("<" + "f" * (len(data) // 4), data))
    selected = raw[::4]
    error = max((abs(a - b) for a, b in zip(selected, expected)), default=0.0)
    exact = error <= 1e-5

steady = []
for line in Path(profile_path).read_text(encoding="utf-8").splitlines():
    if "kind=steady" not in line:
        continue
    steady.append({k: float(v) for k, v in re.findall(
        r"(h2d_us|run_us|d2h_us|device_us|cycles)=([0-9.]+)", line)})

def mean(name):
    values = [item[name] for item in steady if name in item]
    return sum(values) / len(values) if values else None

summary = {
    "schema": "vip9000-e022-fused-q1-target/v1",
    "dtype": dtype,
    "golden_status": "exact" if exact else "mismatch",
    "max_abs": error,
    "output_values_checked": len(expected),
    "repeat_equal_bad": sum("repeat_equal=0" in line for line in Path(profile_path).read_text(encoding="utf-8").splitlines()),
    "steady_iterations": len(steady),
    "h2d_mean_us": mean("h2d_us"),
    "run_mean_us": mean("run_us"),
    "d2h_mean_us": mean("d2h_us"),
    "device_mean_us": mean("device_us"),
    "cycles_mean": mean("cycles"),
    "thermal_before_raw": Path(before_path).read_text(encoding="utf-8"),
    "thermal_after_raw": Path(after_path).read_text(encoding="utf-8"),
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if not exact or summary["repeat_equal_bad"]:
    raise SystemExit(1)
PY
