#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --output FILE [--image IMAGE] [--docker-bin PATH]" >&2
}

docker_bin="docker"
image="acuitylite:ready"
output=""
while (($#)); do
    case "$1" in
        --docker-bin)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            docker_bin="$2"
            shift 2
            ;;
        --image)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            image="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

[[ -n "$output" ]] || { usage; exit 2; }
[[ -n "$docker_bin" ]] || { echo "docker binary must be nonempty" >&2; exit 2; }
if [[ -z "$image" || "$image" == -* || ! "$image" =~ ^[A-Za-z0-9._/:@-]+$ ]]; then
    echo "invalid Docker image reference: $image" >&2
    exit 2
fi
if [[ -e "$output" || -L "$output" ]]; then
    echo "refusing to overwrite existing output file: $output" >&2
    exit 2
fi
output_parent=$(dirname -- "$output")
mkdir -p -- "$output_parent"
tmp_output=$(mktemp -- "${output}.tmp.XXXXXX")
cleanup() {
    if [[ -n "${tmp_output:-}" ]]; then
        rm -f -- "$tmp_output"
    fi
}
trap cleanup EXIT

if "$docker_bin" run \
        --rm \
        --network none \
        --read-only \
        --tmpfs /tmp:rw,noexec,nosuid,size=256m \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --pids-limit 128 \
        --memory 1g \
        "$image" \
        bash -lc '
set -euo pipefail
archive=/usr/local/lib/python3.10/dist-packages/acuitylib/vsi_sdk.tar.gz
sdk_root=/tmp/vsi-sdk
printf "[acuity-package]\n"
python3 - <<"PY"
from importlib.metadata import metadata, version
for name in ("acuitylite", "acuitylib"):
    try:
        print(f"distribution={name}")
        print(f"version={version(name)}")
        details = metadata(name)
        print(f"summary={details.get('Summary', '')}")
        break
    except Exception:
        pass
PY
printf "\n[embedded-sdk-archive]\n"
sha256sum "$archive"
mkdir -p "$sdk_root"
tar -xzf "$archive" -C "$sdk_root"
printf "\n[license-target-marker]\n"
license_file="$sdk_root/vsi_sdk/licence.txt"
license_sha="$(sha256sum "$license_file")"
printf "sha256=%s\n" "${license_sha%% *}"
license_marker="$(python3 - "$license_file" <<"PY"
import re
import sys
from pathlib import Path

value = Path(sys.argv[1]).read_bytes().decode("utf-8", errors="replace").strip()
if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
    print(value)
else:
    print("<redacted>")
PY
)"
printf "marker=%s\n" "$license_marker"
printf "\n\n[prebuilt-sdk-version]\n"
cat "$sdk_root/vsi_sdk/prebuilt-sdk/x86_64_linux/VERSION"
printf "\n[sdk-libraries]\n"
find "$sdk_root/vsi_sdk/prebuilt-sdk/x86_64_linux/lib" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
printf "\n[llm-relevant-headers]\n"
find "$sdk_root/vsi_sdk/prebuilt-sdk/x86_64_linux/include" -type f \
    \( -iname "*matrixmul*" -o -iname "*fullconnect*" -o -iname "*rmsnorm*" -o -iname "*layernorm*" -o -iname "*rope*" -o -path "*/custom/*" \) \
    -printf "%P\n" | sort
printf "\n[vxc-evis-markers]\n"
strings "$sdk_root/vsi_sdk/prebuilt-sdk/x86_64_linux/lib/libtim-vx.so" \
    | grep -E "VXC_512Bits|VXC_DP|EVIS" \
    | sort -u \
    | head -n 80 || true
' > "$tmp_output"; then
    if [[ -e "$output" || -L "$output" ]]; then
        echo "refusing to overwrite existing output file: $output" >&2
        exit 2
    fi
    mv -- "$tmp_output" "$output"
    tmp_output=""
else
    docker_status=$?
    exit "$docker_status"
fi

printf '%s\n' "$output"
