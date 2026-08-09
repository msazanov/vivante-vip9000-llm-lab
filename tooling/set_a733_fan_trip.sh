#!/usr/bin/env bash
set -euo pipefail

mode="apply"
if (($#)); then
    case "$1" in
        --verify)
            mode="verify"
            shift
            ;;
        --check-ready)
            mode="check-ready"
            shift
            ;;
        --wait-ready)
            mode="wait-ready"
            shift
            ;;
        -h|--help)
            echo "usage: $0 [--verify|--check-ready|--wait-ready]" >&2
            exit 0
            ;;
        *)
            echo "usage: $0 [--verify|--check-ready|--wait-ready]" >&2
            exit 2
            ;;
    esac
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--verify|--check-ready|--wait-ready]" >&2; exit 2; }

fail() {
    echo "set_a733_fan_trip: $*" >&2
    exit 1
}

# Test hooks are permitted only when the helper is run from the staging tree.
# The installed production path rejects every A733_*_TEST_* environment name,
# including future hooks, before inspecting the test-only sysfs override.
case "$0" in
    /usr/local/sbin/*)
        test_hook_names=$(env | awk -F= '$1 ~ /^A733_.*_TEST_/ { print $1 }')
        [[ -z "$test_hook_names" ]] ||
            fail "test-only environment hooks are forbidden from the production path: $test_hook_names"
        ;;
esac

# The alternate root is deliberately test-only. Production runs always use
# the real kernel sysfs mounted at /sys.
if [[ -n "${A733_FAN_TRIP_TEST_SYSFS_ROOT:-}" ]]; then
    sysfs_root="$A733_FAN_TRIP_TEST_SYSFS_ROOT"
else
    sysfs_root="/sys"
fi

thermal_root="$sysfs_root/class/thermal"
target_temp="30000"

if [[ "$mode" == "wait-ready" ]]; then
    for attempt in 1 2 3 4 5; do
        if "$0" --check-ready; then
            echo "set_a733_fan_trip: sysfs ready after attempt $attempt"
            exit 0
        fi
        [[ "$attempt" -eq 5 ]] || sleep 1
    done
    fail "thermal sysfs was not ready after 5 attempts"
fi

read_value() {
    local path="$1"
    local value
    [[ -r "$path" ]] || return 1
    value=$(<"$path") || return 1
    value="${value//$'\r'/}"
    value="${value//[[:space:]]/}"
    printf '%s' "$value"
}

[[ -d "$thermal_root" ]] || fail "thermal sysfs root is unavailable: $thermal_root"

cpub_zone=""
cpul_zone=""
zone_count=0
for zone in "$thermal_root"/thermal_zone*; do
    [[ -d "$zone" ]] || continue
    zone_type=$(read_value "$zone/type") || fail "cannot read zone type: $zone/type"
    case "$zone_type" in
        cpub_thermal_zone)
            [[ -z "$cpub_zone" ]] || fail "multiple cpub_thermal_zone entries"
            cpub_zone="$zone"
            zone_count=$((zone_count + 1))
            ;;
        cpul_thermal_zone)
            [[ -z "$cpul_zone" ]] || fail "multiple cpul_thermal_zone entries"
            cpul_zone="$zone"
            zone_count=$((zone_count + 1))
            ;;
    esac
done

[[ $zone_count -eq 2 ]] || fail "expected exactly one cpub_thermal_zone and one cpul_thermal_zone"

fan_trip_paths=()
original_values=()

resolve_fan_trip() {
    local zone="$1"
    local trip_type trip_index trip_temp original_value
    local cdev cdev_name cdev_index cdev_trip cdev_type bound
    local fan_bound found=0

    for trip_type in "$zone"/trip_point_*_type; do
        [[ -f "$trip_type" ]] || continue
        trip_index="${trip_type##*/trip_point_}"
        trip_index="${trip_index%_type}"
        [[ "$trip_index" =~ ^[0-9]+$ ]] || continue
        [[ "$(read_value "$trip_type")" == "passive" ]] || continue

        fan_bound=0
        for cdev in "$zone"/cdev[0-9]*; do
            [[ -e "$cdev" || -L "$cdev" ]] || continue
            cdev_name="${cdev##*/}"
            [[ "$cdev_name" =~ ^cdev([0-9]+)$ ]] || continue
            cdev_index="${BASH_REMATCH[1]}"
            cdev_trip="$zone/cdev${cdev_index}_trip_point"
            [[ -r "$cdev_trip" ]] || continue
            bound=$(read_value "$cdev_trip") || continue
            [[ "$bound" == "$trip_index" ]] || continue
            cdev_type="$cdev/type"
            [[ "$(read_value "$cdev_type" 2>/dev/null || true)" == "pwm-fan" ]] || continue
            fan_bound=1
            break
        done

        if [[ $fan_bound -eq 1 ]]; then
            found=$((found + 1))
            trip_temp="$zone/trip_point_${trip_index}_temp"
            [[ -f "$trip_temp" && -r "$trip_temp" && -w "$trip_temp" ]] ||
                fail "fan trip is not readable and writable: $trip_temp"
            if ! original_value=$(read_value "$trip_temp"); then
                fail "cannot read original fan trip: $trip_temp"
            fi
            [[ "$original_value" =~ ^[0-9]+$ ]] ||
                fail "original fan trip is not numeric: $trip_temp"
            fan_trip_paths+=("$trip_temp")
            original_values+=("$original_value")
        fi
    done

    [[ $found -eq 1 ]] || fail "expected exactly one passive pwm-fan trip in $zone"
}

# Resolve both zones and capture both originals before changing either one.
resolve_fan_trip "$cpub_zone"
resolve_fan_trip "$cpul_zone"

if [[ "$mode" == "verify" ]]; then
    for index in "${!fan_trip_paths[@]}"; do
        if [[ "${A733_FAN_TRIP_TEST_READBACK_ERROR_INDEX:-}" == "$index" ]]; then
            fail "injected test readback error at index $index"
        fi
        if ! current_value=$(read_value "${fan_trip_paths[$index]}"); then
            fail "verification readback failed for ${fan_trip_paths[$index]}"
        fi
        [[ "$current_value" == "$target_temp" ]] ||
            fail "verification mismatch for ${fan_trip_paths[$index]}"
    done
    echo "set_a733_fan_trip: verified ${#fan_trip_paths[@]} pwm-fan trips at ${target_temp} mC"
    exit 0
fi

if [[ "$mode" == "check-ready" ]]; then
    echo "set_a733_fan_trip: sysfs zones and pwm-fan bindings are ready"
    exit 0
fi

rollback() {
    local rollback_status=0
    local index
    for index in "${!fan_trip_paths[@]}"; do
        if [[ "${A733_FAN_TRIP_TEST_FAIL_ROLLBACK_INDEX:-}" == "$index" ]]; then
            echo "set_a733_fan_trip: injected rollback failure at index $index" >&2
            rollback_status=1
            continue
        fi
        if ! printf '%s\n' "${original_values[$index]}" > "${fan_trip_paths[$index]}"; then
            echo "set_a733_fan_trip: rollback failed for ${fan_trip_paths[$index]}" >&2
            rollback_status=1
        fi
    done
    return "$rollback_status"
}

rollback_and_exit() {
    local reason="$1"
    echo "set_a733_fan_trip: $reason" >&2
    if ! rollback; then
        echo "set_a733_fan_trip: WARNING: rollback was incomplete" >&2
    fi
    exit 1
}

for index in "${!fan_trip_paths[@]}"; do
    if [[ "${A733_FAN_TRIP_TEST_FAIL_WRITE_INDEX:-}" == "$index" ]]; then
        rollback_and_exit "injected test write failure at index $index"
    fi
    if ! printf '%s\n' "$target_temp" > "${fan_trip_paths[$index]}"; then
        rollback_and_exit "write failed for ${fan_trip_paths[$index]}"
    fi

    if [[ "${A733_FAN_TRIP_TEST_READBACK_ERROR_INDEX:-}" == "$index" ]]; then
        rollback_and_exit "injected test readback error at index $index"
    elif [[ "${A733_FAN_TRIP_TEST_FAIL_READBACK_INDEX:-}" == "$index" ]]; then
        readback="injected-readback-failure"
    elif ! readback=$(read_value "${fan_trip_paths[$index]}"); then
        rollback_and_exit "readback failed for ${fan_trip_paths[$index]}"
    fi
    if [[ "$readback" != "$target_temp" ]]; then
        rollback_and_exit "readback mismatch for ${fan_trip_paths[$index]}"
    fi
done

echo "set_a733_fan_trip: configured ${#fan_trip_paths[@]} pwm-fan trips to ${target_temp} mC"
