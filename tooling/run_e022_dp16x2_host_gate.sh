#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=""
IMAGE="ubuntu-npu:v2.0.10.2"
MODE="base"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --output-dir DIR [--image IMAGE]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
NBG_NAME="e022_dp16x2_b_probe_${MODE}.nb"
LOG_PATH="$OUTPUT_DIR/host_export.log"
set +e
docker run --rm \
    -e E022_NBG_NAME="$NBG_NAME" -e E022_MODE="$MODE" \
    -v "$ROOT_DIR:/work:ro" -v "$OUTPUT_DIR:/out" \
    "$IMAGE" sh -lc '
        set -eu
        SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
        COMPILER="$SDK/vsimulator/bin/vcCompiler"
        CONFIG="$SDK/common/cfg/VIP9000NANODI_PID0X1000003B.config"
        INCLUDE="$SDK/vsimulator/include"
        SOURCE=/work/experiments/E022-fused-q1/dp16x2_b_probe.vx
        "$COMPILER" -VX -f"$CONFIG" -Ke022_dp16x2_b_probe \
            -I"$INCLUDE" -I"$INCLUDE/CL" -o/out/e022_dp16x2_b_probe "$SOURCE"
        gcc -std=c11 -O2 -Wall -Wextra -Werror \
            -I"$INCLUDE" -L"$SDK/vsimulator/lib" -L"$SDK/common/lib" \
            -Wl,-rpath,"$SDK/vsimulator/lib:$SDK/common/lib" \
            -o /tmp/e022_dp16x2_builder \
            /work/experiments/E022-fused-q1/dp16x2_b_probe_nbg_builder.c \
            -lOpenVX -lm -ldl -lpthread
        VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B \
        VSIMULATOR_SHADER_CORE_COUNT=1 \
        LD_LIBRARY_PATH="$SDK/vsimulator/lib:$SDK/common/lib" \
            /tmp/e022_dp16x2_builder /out/e022_dp16x2_b_probe.vxgcSL \
            "/out/$E022_NBG_NAME" "$E022_MODE"
    ' >"$LOG_PATH" 2>&1
HOST_EXIT=$?
set -e
if [ "$HOST_EXIT" -ne 0 ]; then
    tail -120 "$LOG_PATH" >&2 || true
    exit "$HOST_EXIT"
fi
NBG_BYTES=$(stat -c %s "$OUTPUT_DIR/$NBG_NAME")
python3 - "$OUTPUT_DIR/summary.json" "$NBG_NAME" "$NBG_BYTES" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

summary = {
    "schema": "vip9000-e022-dp16x2-probe-host/v1",
    "custom_kernel_count": 1,
    "graph_inputs": 3,
    "graph_outputs": 1,
    "output_values": ["pair_00_x", "pair_00_y", "pair_01_x", "pair_01_y", "pair_11_x", "pair_11_y", "pair_15_x", "pair_15_y"],
    "mode": sys.argv[4],
    "nbg_file": sys.argv[2],
    "nbg_bytes": int(sys.argv[3]),
}
Path(sys.argv[1]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cat "$OUTPUT_DIR/summary.json"
