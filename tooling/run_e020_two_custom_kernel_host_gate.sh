#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=""
IMAGE="ubuntu-npu:v2.0.10.2"
ELEMENTS="16"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --elements) ELEMENTS=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --output-dir DIR [--image IMAGE] [--elements N]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

NBG_NAME="e020_two_custom_kernel_u8_${ELEMENTS}.nb"
LOG_PATH="$OUTPUT_DIR/host_export.log"
set +e
docker run --rm \
    -e E020_ELEMENTS="$ELEMENTS" -e E020_NBG_NAME="$NBG_NAME" \
    -v "$ROOT_DIR:/work:ro" -v "$OUTPUT_DIR:/out" \
    "$IMAGE" sh -lc '
        set -eu
        SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
        COMPILER="$SDK/vsimulator/bin/vcCompiler"
        CONFIG="$SDK/common/cfg/VIP9000NANODI_PID0X1000003B.config"
        INCLUDE="$SDK/vsimulator/include"
        SOURCE=/work/experiments/E020-two-custom-kernel/two_stage_invert.vx
        "$COMPILER" -VX -f"$CONFIG" -Ke020_invert_stage_a \
            -I"$INCLUDE" -I"$INCLUDE/CL" -o/out/e020_stage_a "$SOURCE"
        "$COMPILER" -VX -f"$CONFIG" -Ke020_invert_stage_b \
            -I"$INCLUDE" -I"$INCLUDE/CL" -o/out/e020_stage_b "$SOURCE"
        gcc -std=c11 -O2 -Wall -Wextra -Werror \
            -I"$INCLUDE" -L"$SDK/vsimulator/lib" -L"$SDK/common/lib" \
            -Wl,-rpath,"$SDK/vsimulator/lib:$SDK/common/lib" \
            -o /tmp/e020_builder \
            /work/experiments/E020-two-custom-kernel/two_custom_kernel_nbg_builder.c \
            -lOpenVX -lm -ldl -lpthread
        VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B \
        VSIMULATOR_SHADER_CORE_COUNT=1 \
        LD_LIBRARY_PATH="$SDK/vsimulator/lib:$SDK/common/lib" \
            /tmp/e020_builder /out/e020_stage_a.vxgcSL \
            /out/e020_stage_b.vxgcSL "/out/$E020_NBG_NAME" "$E020_ELEMENTS"
    ' >"$LOG_PATH" 2>&1
HOST_EXIT=$?
set -e
if [ "$HOST_EXIT" -ne 0 ]; then
    tail -100 "$LOG_PATH" >&2 || true
    exit "$HOST_EXIT"
fi
NBG_BYTES=$(stat -c %s "$OUTPUT_DIR/$NBG_NAME")
python3 - "$OUTPUT_DIR/summary.json" "$ELEMENTS" "$NBG_NAME" "$NBG_BYTES" <<'PY'
import json
import sys
from pathlib import Path

summary = {
    "schema": "vip9000-e020-two-custom-kernel-host/v1",
    "custom_kernel_count": 2,
    "graph_node_count": 2,
    "graph_inputs": 1,
    "graph_outputs": 1,
    "intermediate": "virtual_uint8_tensor",
    "elements": int(sys.argv[2]),
    "nbg_file": sys.argv[3],
    "nbg_bytes": int(sys.argv[4]),
}
Path(sys.argv[1]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cat "$OUTPUT_DIR/summary.json"
