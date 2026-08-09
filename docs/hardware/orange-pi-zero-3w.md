# Target hardware inventory: `orangepizero3w`

**Date:** 2026-08-07  
**Scope:** read-only target inventory; no NPU inference was run for this document.

## Question

What board identity, compute state, and VIPLite entry points are actually available on the SSH target labelled `orangepizero3w`?

## Current answer

**Unknown/contradictory —** the SSH hostname/profile says `orangepizero3w`, but the device tree reports compatible strings `xunlong,orangepi-4-pro` and `arm,sun60iw2p1`. The identity conflict is retained rather than resolved.  
**Verified on target —** the observed operating system is Debian 12 with kernel `6.6.98-sun60iw2`; the target exposes eight CPUs, roughly 11 GiB usable RAM, a 5.8 GiB swap area, `/dev/vipcore`, and an NPU devfreq node.  
**Verified on target —** target files and a persistent runner are present, but their presence and linkage do not establish inference correctness or performance.

## Evidence

- **Verified on target —** CPU inventory was six Cortex-A55 cores up to 1.794 GHz and two Cortex-A76 cores up to 2.002 GHz. CPU governors were `ondemand`; idle temperatures were approximately 57–59 °C in most reported zones.
- **Verified on target —** `/sys/class/devfreq/3600000.npu` reported available frequencies `492 852 1008` MHz, current frequency `1008` MHz, and governor `performance`.
- **Verified on target —** `/dev/vipcore` existed with world read/write permissions and the `vipcore` kernel module was loaded.
- **Verified on target —** `/usr/include/vip_lite.h` and `/usr/include/vip_lite_common.h` reported VIPLite API 2.0.
- **Verified on target —** `/usr/lib/libNBGlinker.so` was 174,560 bytes with SHA-256 `de93bb4a7d86af67bc4ca597d12228b9b21b062601b18f33ca30393a2af4a842`; `/usr/lib/libVIPhal.so` was 39,248 bytes with SHA-256 `648444a26a1aeaec0182c0ba348b7a453134b8ccdf8282a211ce4c0f5e5e51a0`.
- **Verified on target —** `libNBGlinker.so` exported persistent network, buffer, and profiling APIs.
- **Unknown/contradictory —** no target userspace OpenVX or TIM-VX libraries were found. An installed OpenCL ICD is not evidence that Vivante OpenCL is available or usable.
- **Verified on target —** `/opt/labse-npu/persistent_viplite_runner` was 72,232 bytes with SHA-256 `13650732606ed3a4dca55dba3a2288bd24d50c35fb4ca8f8dc8b164dfedff055`; it linked to `libNBGlinker.so` and `libVIPhal.so` and exposed persistent-serve and one-shot modes.
- **Verified on target —** `/var/lib/labse-npu/models/labse_int8.nb` was approximately 55 MiB with SHA-256 `c2ae38e4d12c2fb71d5129a539815ad0a6bba179ae6e2ac5634d6b9b762cea88`.
- **Verified on target —** the inspected Python service performed CPU tokenization/embedding, called a persistent VIPLite body, and used a CPU pool/L2 path; its service unit was disabled and inactive.
- **Hypothesis —** persistent serving and resident buffers are the appropriate shape for a future decode experiment because one-shot process or graph setup could obscure steady-state measurements. This is a measurement-design hypothesis, not a speed result.

## Confidence

**Verified on target — high** for the captured kernel, sysfs, device-node, header, file-size, and SHA-256 observations.  
**Unknown/contradictory — medium** for board identity and any capability inferred from a missing library or from exported symbols alone.

## Unresolved

- **Unknown/contradictory —** the hostname/profile and device-tree identity still disagree.
- **Unknown/contradictory —** no correctness, throughput, thermal-duration, or memory-pressure result is established for the existing runner/model pair.
- **Unknown/contradictory —** the relationship between `libNBGlinker.so` profiling exports and the standard profiler’s telemetry is not yet measured.

## Approved PWM-fan policy

**Verified on target —** `cpub_thermal_zone` and `cpul_thermal_zone` each
exposed a passive trip bound to the `pwm-fan` cooling device. The operator
approved `30000` millidegrees Celsius as the fan activation point while
retaining the kernel thermal governor, the existing fan curve, the 90 °C
CPU-frequency cooling trip, and the 110 °C critical trip.

The persistent installation uses
`tooling/set_a733_fan_trip.sh` and
`tooling/systemd/a733-fan-trip-30c.service`. At boot the root-owned oneshot
discovers zones by `type`, validates both fan bindings, completes all
writeability and numeric-original-value preflight, then writes and verifies
the two fan-bound passive trips. Preflight refuses partial configuration;
rollback after a write or readback failure is best-effort and emits a warning
if any original value cannot be restored.

The service and helper are persistent filesystem state. The trip values in
`/sys/class/thermal` are runtime state and must be checked after each boot;
the service reapplies them. This policy makes the fan start earlier; it does
not prove a hardware-safe temperature, replace kernel protection, or alter
CPU-frequency, GPU, NPU, or critical-trip settings.

**Verified on target —** on 2026-08-09 the reviewed helper and unit were
installed with root ownership, passed target `systemd-analyze verify`, and the
unit was enabled and started. Binding-aware verification found both fan trips
at 30 °C; three one-second samples reported cooling state 4/4 and PWM 255 while
CPU-zone temperatures were 45.2–47.6 °C. CPU-frequency and critical trips
remained 90 °C and 110 °C, and GPU/NPU trips were unchanged. A cold reboot was
not performed, so post-boot reapplication remains an explicit follow-up. See
[the sanitized installation evidence](../evidence/a733-fan-policy-2026-08-09.md).

## Next experiment

**Hypothesis —** run the existing persistent runner under the repository profiler with a pinned CPU reference, fixed warm-up/repetition counts, and a deterministic quality check. Record initialization, warm invocation, synchronization, RSS, NPU frequency, and thermal samples separately; do not interpret the run as a performance result until quality data is present.

## Read-only reproduction commands

```sh
ssh orangepizero3w 'cat /etc/os-release; uname -r; tr "\0" "\n" </proc/device-tree/compatible'
ssh orangepizero3w 'awk "/^model name|^CPU MHz/{print}" /proc/cpuinfo; free -h; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
ssh orangepizero3w 'cat /sys/class/devfreq/3600000.npu/available_frequencies; cat /sys/class/devfreq/3600000.npu/cur_freq; cat /sys/class/devfreq/3600000.npu/governor'
ssh orangepizero3w 'stat -c "%A %a %n" /dev/vipcore; lsmod | grep "^vipcore\b"'
ssh orangepizero3w 'grep -n "2\.0" /usr/include/vip_lite.h /usr/include/vip_lite_common.h'
ssh orangepizero3w 'sha256sum /usr/lib/libNBGlinker.so /usr/lib/libVIPhal.so /opt/labse-npu/persistent_viplite_runner /var/lib/labse-npu/models/labse_int8.nb'
ssh orangepizero3w 'readelf -Ws /usr/lib/libNBGlinker.so | grep -Ei "persistent|network|buffer|profil"'
```
