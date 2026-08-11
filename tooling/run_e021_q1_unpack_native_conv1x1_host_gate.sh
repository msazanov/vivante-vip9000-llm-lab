#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=""
IMAGE="ubuntu-npu:v2.0.10.2"
M="16"
K="32"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --m) M=$2; shift 2 ;;
        --k) K=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --output-dir DIR [--image IMAGE]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
NBG_NAME="e021_q1_unpack_native_conv1x1_m${M}_k${K}_i8.nb"
LOG_PATH="$OUTPUT_DIR/host_export.log"
set +e
docker run --rm \
    -e E020_NBG_NAME="$NBG_NAME" -e E020_M="$M" -e E020_K="$K" \
    -v "$ROOT_DIR:/work:ro" -v "$OUTPUT_DIR:/out" \
    "$IMAGE" sh -lc '
        set -eu
        SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
        COMPILER="$SDK/vsimulator/bin/vcCompiler"
        CONFIG="$SDK/common/cfg/VIP9000NANODI_PID0X1000003B.config"
        INCLUDE="$SDK/vsimulator/include"
        SOURCE=/work/experiments/E021-q1-unpack-native-matmul/q1_unpack_i8.vx
        "$COMPILER" -VX -f"$CONFIG" -Ke020_q1_unpack_i8 -DE020_K="$E020_K" \
            -I"$INCLUDE" -I"$INCLUDE/CL" -o/out/e020_q1_unpack "$SOURCE"
        gcc -std=c11 -O2 -Wall -Wextra -Werror \
            -I"$INCLUDE" -L"$SDK/vsimulator/lib" -L"$SDK/common/lib" \
            -Wl,-rpath,"$SDK/vsimulator/lib:$SDK/common/lib" \
            -o /tmp/e021_q1_conv_builder \
            /work/experiments/E021-q1-unpack-native-matmul/q1_unpack_native_conv1x1_nbg_builder.c \
            -lOpenVX -lm -ldl -lpthread
        cd "$INCLUDE/CL"
        VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B \
        VSIMULATOR_SHADER_CORE_COUNT=1 \
        VX_SHADER_SOURCE_PATH="$INCLUDE/CL" \
        LD_LIBRARY_PATH="$SDK/vsimulator/lib:$SDK/common/lib" \
            /tmp/e021_q1_conv_builder /out/e020_q1_unpack.vxgcSL "/out/$E020_NBG_NAME" \
            "$E020_M" "$E020_K"
    ' >"$LOG_PATH" 2>&1
HOST_EXIT=$?
set -e
if [ "$HOST_EXIT" -ne 0 ]; then
    tail -120 "$LOG_PATH" >&2 || true
    exit "$HOST_EXIT"
fi
NBG_BYTES=$(stat -c %s "$OUTPUT_DIR/$NBG_NAME")
python3 - "$OUTPUT_DIR/summary.json" "$NBG_NAME" "$NBG_BYTES" "$M" "$K" <<'PY'
import json
import sys
from pathlib import Path

summary = {
    "schema": "vip9000-e021-q1-unpack-native-conv1x1-host/v1",
    "shape": {"m": int(sys.argv[4]), "k": int(sys.argv[5]), "n": 1},
    "graph_inputs": 2,
    "graph_outputs": 1,
    "custom_kernel_count": 1,
    "native_op": "vxConvolutionLayer_1x1",
    "intermediate": "virtual_int8_weights",
    "external_packed_weight_bytes": int(sys.argv[4]) * int(sys.argv[5]) // 8,
    "external_expanded_weight_bytes": 0,
    "nbg_file": sys.argv[2],
    "nbg_bytes": int(sys.argv[3]),
}
Path(sys.argv[1]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cat "$OUTPUT_DIR/summary.json"
