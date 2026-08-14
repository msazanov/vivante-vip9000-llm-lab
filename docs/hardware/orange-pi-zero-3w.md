# Orange Pi Zero 3W target observations

This is a sanitized, read-only target fingerprint. It records what was seen
on the SSH target labelled `orangepizero3w`; it does not resolve the board
identity conflict or claim that every installed file is redistributable.

## Verified on target

- Debian 12 with Linux `6.6.98-sun60iw2`.
- Eight logical CPUs: six A55-class cores and two A76-class cores.
- Approximately 11 GiB usable RAM and a 5.8 GiB swap area at the capture.
- `/dev/vipcore` and a loaded `vipcore` module.
- VIPLite API 2.0 headers and runtime ABI `0x00020003`.
- NPU device frequency node exposing 492, 852, and 1008 MHz; this is the NPU
  clock, not the memory-controller clock.
- Read-only clock/register evidence reports a 2040 MHz `PLL_DDR` parent and a
  510 MHz memory-controller clock.
- The target file inventory contains hashes for runtime libraries and a
  persistent runner; hashes are provenance only and do not publish binaries.

## Unknown or contradictory

- The hostname/profile says `orangepizero3w`, while device-tree compatible
  strings include `xunlong,orangepi-4-pro` and `arm,sun60iw2p1`.
- The installed DRAM vendor, physical data rate, training result, and effective
  sustained bandwidth are not independently established.
- A Linux DDR devfreq control is not bound on the captured image.
- Runtime symbols and a working NBG prove an execution path exists, but they do
  not establish correctness for Bonsai, an efficient graph partition, or
  full-model throughput.

## Reproduction boundary

The original target commands were read-only inventory commands. A future run
must collect the same identity, clock, thermal, cooling, memory, and runtime
fields before interpreting a result. Keep a separate model SHA, repository
commit, runtime identity, and raw-result pointer for every run.

## Evidence pointers

- [Canonical A733 fact sheet](a733.md)
- [DDR frequency audit](../evidence/a733-ddr-frequency-audit-2026-08-10.md)
- [VIP9000 capability probe](../evidence/vip9000-next-capability-probe-2026-08-09.md)
- [Target phase profile](../evidence/vip9000-phase-profile-2026-08-10.md)
- [Fan and thermal policy](../evidence/a733-fan-policy-2026-08-09.md)
