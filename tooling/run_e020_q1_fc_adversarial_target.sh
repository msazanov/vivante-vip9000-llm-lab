#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NBG=""
OUTPUT_DIR=""
TARGET="orangepi@192.168.31.117"
REMOTE_DIR="/home/orangepi/vip9000-lab/e020-q1-fc-adversarial"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --nbg) NBG=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
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

for pattern in zeros ones alternating mixed; do
    CASE_DIR="$OUTPUT_DIR/$pattern"
    mkdir -p "$CASE_DIR"
    "$ROOT_DIR/tooling/generate_e020_q1_fc_fixture.py" \
        --output-dir "$CASE_DIR" --m 16 --k 32 --pattern "$pattern"
    scp -F /dev/null -o BatchMode=yes \
        "$CASE_DIR/packed_q1.bin" "$CASE_DIR/activation_i8.bin" \
        "$CASE_DIR/golden_output_i8.bin" "$TARGET:$REMOTE_DIR/"
    ssh -F /dev/null -o BatchMode=yes "$TARGET" \
        "cd $REMOTE_DIR && /home/orangepi/vip9000-lab/bin/profiled_viplite_runner --iterations 10 network.nb packed_q1.bin activation_i8.bin output_i8.bin" \
        >"$CASE_DIR/profile.log"
    scp -F /dev/null -o BatchMode=yes \
        "$TARGET:$REMOTE_DIR/output_i8.bin" "$CASE_DIR/output_i8.bin"
    cmp "$CASE_DIR/golden_output_i8.bin" "$CASE_DIR/output_i8.bin"
done

echo "golden=bit_exact,patterns=4,iterations_each=10"
