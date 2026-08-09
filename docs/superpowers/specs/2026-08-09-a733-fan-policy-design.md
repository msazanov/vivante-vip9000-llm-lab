# A733 fan trip policy design

## Goal

Keep the physically connected PWM fan active from 30 °C upward while retaining
the kernel thermal governor, the existing fan curve, the 90 °C CPU-frequency
cooling trip, and the 110 °C critical trip.

## Verified target facts

- `cpub_thermal_zone` and `cpul_thermal_zone` each expose a passive trip 0
  bound to the `pwm-fan` cooling device.
- Both fan trips originally read 55000 millidegrees Celsius.
- The runtime value 30000 was accepted by both zones on 2026-08-09.
- The physical fan was confirmed by the operator. At verification it reported
  cooling state 4/4 and PWM 255.

These facts apply to the current BSP. Thermal-zone numbers are not a stable
interface and must not be embedded in the persistent policy.

## Design

Install a root-owned oneshot helper and systemd unit. The helper discovers
zones by their `type` files, requires exactly one `cpub_thermal_zone` and one
`cpul_thermal_zone`, and resolves the trip that is both passive and bound to a
cooling device whose type is `pwm-fan`. It completes all discovery and
validation before writing either zone.

The helper writes 30000 millidegrees Celsius and verifies the readback. On any
write or verification failure it attempts to restore every original value and
returns non-zero. It never changes governors, PWM values, fan enable mode,
CPU-frequency trips, critical trips, GPU trips, or NPU trips.

The systemd unit runs after module loading and before `multi-user.target`, then
remains in the successful exited state. The helper accepts an alternate sysfs
root only through a test-only environment variable so tests never write real
sysfs.

## Safety and evidence

Thirty degrees is an operator-approved fan activation point, not a hardware
safe-temperature claim. The policy makes the fan start earlier; it does not
replace the separate 85 °C benchmark abort ceiling or kernel protection.
Installation is verified from the live sysfs values and `systemctl` state.

