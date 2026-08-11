#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=""
IMAGE="ubuntu-npu:v2.0.10.2"
ROWS=4
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --rows) ROWS=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --output-dir DIR [--image IMAGE] [--rows multiple-of-4]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
NBG_NAME="e022_q1_fused_dp16x1_rows4_m${ROWS}_k5120_i32.nb"
LOG_PATH="$OUTPUT_DIR/host_export.log"
set +e
docker run --rm \
    -e E022_FUSED_NBG_NAME="$NBG_NAME" -e E022_FUSED_ROWS="$ROWS" \
    -v "$ROOT_DIR:/work:ro" -v "$OUTPUT_DIR:/out" \
    "$IMAGE" sh -lc '
        set -eu
        SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
        COMPILER="$SDK/vsimulator/bin/vcCompiler"
        CONFIG="$SDK/common/cfg/VIP9000NANODI_PID0X1000003B.config"
        INCLUDE="$SDK/vsimulator/include"
        "$COMPILER" -VX -f"$CONFIG" -Ke022_q1_fused_dp16x1_rows4_k5120 \
            -I"$INCLUDE" -I"$INCLUDE/CL" \
            -o/out/e022_q1_fused_dp16x1_rows4_k5120 \
            /work/experiments/E022-fused-q1/q1_fused_dp16x1_rows4_k5120.vx
        gcc -std=c11 -O2 -Wall -Wextra -Werror \
            -I"$INCLUDE" -L"$SDK/vsimulator/lib" -L"$SDK/common/lib" \
            -Wl,-rpath,"$SDK/vsimulator/lib:$SDK/common/lib" \
            -o /tmp/e022_q1_fused_rows4_k5120_builder \
            /work/experiments/E022-fused-q1/q1_fused_dp16x1_rows4_k5120_nbg_builder.c \
            -lOpenVX -lm -ldl -lpthread
        VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B \
        VSIMULATOR_SHADER_CORE_COUNT=1 \
        LD_LIBRARY_PATH="$SDK/vsimulator/lib:$SDK/common/lib" \
            /tmp/e022_q1_fused_rows4_k5120_builder \
            /out/e022_q1_fused_dp16x1_rows4_k5120.vxgcSL \
            "/out/$E022_FUSED_NBG_NAME" "$E022_FUSED_ROWS"
    ' >"$LOG_PATH" 2>&1
HOST_EXIT=$?
set -e
if [ "$HOST_EXIT" -ne 0 ]; then
    tail -160 "$LOG_PATH" >&2 || true
    exit "$HOST_EXIT"
fi
NBG_BYTES=$(stat -c %s "$OUTPUT_DIR/$NBG_NAME")
python3 - "$OUTPUT_DIR/summary.json" "$NBG_NAME" "$NBG_BYTES" "$ROWS" <<'PY'
import json
import sys
from pathlib import Path

rows = int(sys.argv[4])
if rows <= 0 or rows % 4:
    raise SystemExit("rows must be a positive multiple of 4")
summary = {
    "schema": "vip9000-e022-fused-q1-rows4-k5120-host/v1",
    "custom_kernel_count": 1,
    "graph_inputs": 2,
    "graph_outputs": 1,
    "rows": rows,
    "rows_per_work_item": 4,
    "k": 5120,
    "q1_row_bytes": 720,
    "q8_activation_bytes": 5440,
    "q1_expanded": False,
    "q8_expanded": False,
    "output_type": "INT32",
    "output_lanes_per_row": 4,
    "nbg_file": sys.argv[2],
    "nbg_bytes": int(sys.argv[3]),
}
Path(sys.argv[1]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
PY
cat "$OUTPUT_DIR/summary.json"
