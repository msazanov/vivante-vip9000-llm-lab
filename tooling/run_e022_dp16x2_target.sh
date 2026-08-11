#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NBG=""
OUTPUT_DIR=""
TARGET="orangepi@192.168.31.117"
MODE="base"
REMOTE_DIR="/home/orangepi/vip9000-lab/e022-dp16x2"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --nbg) NBG=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$NBG" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --nbg FILE --output-dir DIR [--target USER@HOST]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
ssh -F /dev/null -o BatchMode=yes "$TARGET" "mkdir -p $REMOTE_DIR"
scp -F /dev/null -o BatchMode=yes "$NBG" "$TARGET:$REMOTE_DIR/network.nb"

for case_name in base selector_all_zero selector_all_five reverse_split; do
    case_dir="$OUTPUT_DIR/$case_name"
    mkdir -p "$case_dir"
    "$ROOT_DIR/tooling/generate_e022_dp16x2_fixture.py" \
        --output-dir "$case_dir" --case "$case_name"
    scp -F /dev/null -o BatchMode=yes \
        "$case_dir/a_hi.bin" "$case_dir/a_lo.bin" "$case_dir/b.bin" \
        "$TARGET:$REMOTE_DIR/"
    ssh -F /dev/null -o BatchMode=yes "$TARGET" \
        "cd $REMOTE_DIR && /home/orangepi/vip9000-lab/bin/profiled_viplite_runner --iterations 100 network.nb a_hi.bin a_lo.bin b.bin output.bin" \
        >"$case_dir/profile.log"
    scp -F /dev/null -o BatchMode=yes "$TARGET:$REMOTE_DIR/output.bin" "$case_dir/output.bin"
    python3 - "$case_dir/golden.json" "$case_dir/output.bin" "$MODE" <<'PY'
import json
import struct
import sys
from pathlib import Path

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["golden"]
raw = Path(sys.argv[2]).read_bytes()
if len(raw) != 32:
    raise SystemExit(f"unexpected output size: {len(raw)}")
values = list(struct.unpack("<8i", raw))
if values[:2] == expected or values[2:4] == expected or values[4:6] == expected or values[6:8] == expected:
    print(json.dumps({"mode": sys.argv[3], "output": values, "reference": expected, "match": True}))
else:
    raise SystemExit(f"pair mismatch: output={values}, expected={expected}")
PY
done
echo "dp16x2_b_target=reference_lanes_exact,mode=$MODE"
