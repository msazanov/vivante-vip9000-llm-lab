#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --output-dir PATH" >&2
}

output_dir=""
while (($#)); do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output_dir="$2"
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

[[ -n "$output_dir" ]] || { usage; exit 2; }
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
    echo "refusing to overwrite existing output directory: $output_dir" >&2
    exit 2
fi
output_parent=$(dirname -- "$output_dir")
mkdir -p -- "$output_parent"
tmp_dir=$(mktemp -d -- "${output_dir}.tmp.XXXXXX")
cleanup() {
    if [[ -n "${tmp_dir:-}" ]]; then
        rm -rf -- "$tmp_dir"
    fi
}
trap cleanup EXIT

system_file="$tmp_dir/system.txt"
hash_file="$tmp_dir/runtime-files.sha256"

required_section() {
    local name="$1"
    shift
    printf '[%s]\n' "$name"
    "$@" 2>&1
    printf '\n'
}

optional_section() {
    local name="$1"
    shift
    printf '[%s]\n' "$name"
    if "$@" 2>&1; then
        :
    else
        printf '[unavailable]\n'
    fi
    printf '\n'
}

device_tree_compatible() {
    local path="/proc/device-tree/compatible"
    if [[ -r "$path" ]]; then
        tr '\000' '\n' < "$path"
    else
        return 1
    fi
}

npu_devfreq() {
    local found=0
    local entry
    for entry in /sys/class/devfreq/*npu*; do
        [[ -e "$entry" ]] || continue
        found=1
        printf '%s\n' "$entry"
        for field in cur_freq governor available_frequencies available_governors; do
            if [[ -r "$entry/$field" ]]; then
                printf '%s=' "$field"
                tr '\n' ' ' < "$entry/$field"
                printf '\n'
            fi
        done
    done
    [[ $found -eq 1 ]]
}

cpu_frequency() {
    local found=0
    local cpu
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        [[ -d "$cpu/cpufreq" ]] || continue
        found=1
        printf '%s ' "${cpu##*/}"
        for field in scaling_governor scaling_cur_freq scaling_max_freq; do
            if [[ -r "$cpu/cpufreq/$field" ]]; then
                printf '%s=' "$field"
                tr '\n' ' ' < "$cpu/cpufreq/$field"
            fi
        done
        printf '\n'
    done
    [[ $found -eq 1 ]]
}

thermal_zones() {
    local found=0
    local zone
    for zone in /sys/class/thermal/thermal_zone*; do
        [[ -d "$zone" ]] || continue
        found=1
        printf '%s ' "${zone##*/}"
        if [[ -r "$zone/type" ]]; then
            printf 'type='; tr '\n' ' ' < "$zone/type"
        fi
        if [[ -r "$zone/temp" ]]; then
            printf 'temp_millic='; tr '\n' ' ' < "$zone/temp"
        fi
        printf '\n'
    done
    [[ $found -eq 1 ]]
}

vip_devices() {
    local found=0
    local path
    for path in /dev/vipcore /sys/module/vipcore /sys/class/devfreq/*npu*; do
        [[ -e "$path" ]] || continue
        found=1
        ls -ld "$path"
    done
    [[ $found -eq 1 ]]
}

runtime_api() {
    local found=0
    local path
    for path in /usr/include/vip_lite.h /usr/include/vip_lite_common.h /usr/lib/libNBGlinker.so /usr/lib/libVIPhal.so; do
        [[ -e "$path" ]] || continue
        found=1
        file "$path"
    done
    [[ $found -eq 1 ]]
}

{
    printf '[generated-utc]\n'
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    printf '\n'
    required_section uname uname -a
    required_section os-release cat /etc/os-release
    required_section lscpu lscpu
    required_section memory free -h
    optional_section device-tree-compatible device_tree_compatible
    optional_section loaded-modules lsmod
    optional_section vip-devices vip_devices
    optional_section npu-devfreq npu_devfreq
    optional_section cpu-frequency cpu_frequency
    optional_section thermal-zones thermal_zones
    optional_section runtime-api runtime_api
} > "$system_file"

: > "$hash_file"
for candidate in \
    /usr/include/vip_lite.h \
    /usr/include/vip_lite_common.h \
    /usr/lib/libNBGlinker.so \
    /usr/lib/libVIPhal.so \
    /opt/labse-npu/persistent_viplite_runner \
    /var/lib/labse-npu/models/labse_int8.nb; do
    if [[ -r "$candidate" && -f "$candidate" ]]; then
        sha256sum "$candidate" >> "$hash_file"
    fi
done

if [[ -e "$output_dir" || -L "$output_dir" ]]; then
    echo "refusing to overwrite existing output directory: $output_dir" >&2
    exit 2
fi
mv -- "$tmp_dir" "$output_dir"
tmp_dir=""

printf '%s\n' "$output_dir"
