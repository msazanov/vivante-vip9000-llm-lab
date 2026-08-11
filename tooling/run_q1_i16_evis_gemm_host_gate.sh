#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=""
IMAGE="ubuntu-npu:v2.0.10.2"
M="1"
K="32"
N="1"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --m) M=$2; shift 2 ;;
        --k) K=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "usage: $0 --output-dir DIR [--image IMAGE] [--m M --k K --n N]" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

SOURCE_REL="experiments/E019-q1-i16-evis-gemm/matrixmul_i16_coordfix.vx"
SHADER_NAME="matrixmul_i16.vxgcSL"
NBG_NAME="q1_i16_evis_gemm_m${M}_k${K}_n${N}.nb"
LOG_PATH="$OUTPUT_DIR/host_export.log"

for forbidden_marker in vxTensorMatrixMultiplyNode vxMatrixMultiplyNode vxTransposeNode MatrixMultiply; do
    if rg -Fq "$forbidden_marker" "$ROOT_DIR/$SOURCE_REL"; then
        echo "forbidden marker found in shader: $forbidden_marker" >&2
        exit 3
    fi
done

set +e
docker run --rm \
    -e SOURCE_REL="$SOURCE_REL" \
    -e E019_M="$M" -e E019_K="$K" -e E019_N="$N" \
    -e E019_NBG_NAME="$NBG_NAME" \
    -v "$ROOT_DIR:/work:ro" \
    -v "$OUTPUT_DIR:/out" \
    "$IMAGE" \
    sh -lc '
        set -eu
        SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
        COMPILER="$SDK/vsimulator/bin/vcCompiler"
        CONFIG="$SDK/common/cfg/VIP9000NANODI_PID0X1000003B.config"
        INCLUDE="$SDK/vsimulator/include"
        LIBS="$SDK/vsimulator/lib:$SDK/common/lib"
        "$COMPILER" -VX -f"$CONFIG" -Kgemm_I16I16toI16 \
            -I"$INCLUDE" -I"$INCLUDE/CL" -o/out/matrixmul_i16 \
            "/work/$SOURCE_REL"
        gcc -std=c11 -O2 \
            -I/work/experiments/E019-q1-i16-evis-gemm \
            -I"$INCLUDE" \
            -L"$SDK/vsimulator/lib" -L"$SDK/common/lib" \
            -Wl,-rpath,"$SDK/vsimulator/lib:$SDK/common/lib" \
            -o /tmp/q1_i16_evis_gemm_nbg_builder \
            /work/experiments/E019-q1-i16-evis-gemm/q1_i16_evis_gemm_nbg_builder.c \
            /work/experiments/E019-q1-i16-evis-gemm/q1_i16_evis_gemm_contract.c \
            -lOpenVX -lm -ldl -lpthread
        VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B \
        VSIMULATOR_SHADER_CORE_COUNT=1 \
        LD_LIBRARY_PATH="$LIBS" \
            /tmp/q1_i16_evis_gemm_nbg_builder \
            /out/matrixmul_i16.vxgcSL "/out/$E019_NBG_NAME" \
            "$E019_M" "$E019_K" "$E019_N"
' >"$LOG_PATH" 2>&1
HOST_EXPORT_EXIT_CODE=$?
set -e

if [ "$HOST_EXPORT_EXIT_CODE" -ne 0 ]; then
    echo "host export failed with exit code $HOST_EXPORT_EXIT_CODE" >&2
    tail -80 "$LOG_PATH" >&2 || true
    exit "$HOST_EXPORT_EXIT_CODE"
fi

SHADER_BYTES=$(stat -c %s "$OUTPUT_DIR/$SHADER_NAME")
NBG_BYTES=$(stat -c %s "$OUTPUT_DIR/$NBG_NAME")
python3 - "$OUTPUT_DIR/summary.json" "$M" "$K" "$N" "$NBG_NAME" "$SHADER_BYTES" "$NBG_BYTES" "$HOST_EXPORT_EXIT_CODE" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = {
    "schema": "vip9000-q1-i16-evis-gemm-host/v1",
    "gate": {"m": int(sys.argv[2]), "k": int(sys.argv[3]), "n": int(sys.argv[4])},
    "kernel": "com.vivantecorp.extension.evis.gemm_I16I16toI16",
    "shader_entry": "gemm_I16I16toI16",
    "graph_inputs": 2,
    "graph_outputs": 1,
    "host_export_exit_code": int(sys.argv[8]),
    "source_variant": "coordfix-signed-output-v2",
    "shader_source_file": "matrixmul_i16_coordfix.vx",
    "shader_binary_bytes": int(sys.argv[6]),
    "nbg_bytes": int(sys.argv[7]),
    "nbg_file": sys.argv[5],
    "forbidden_markers_found": [],
    "registration_experiment": {
        "compiler_entry": "gemm_I16I16toI16",
        "vxAddKernelInProgram": "com.vivantecorp.extension.evis.gemm_I16I16toI16",
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cat "$OUTPUT_DIR/summary.json"
