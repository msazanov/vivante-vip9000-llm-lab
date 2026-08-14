# VIP9000 capability map

This map separates public/compiler hints from exact target observations. A
capability is not accepted for Bonsai until an exact golden and end-to-end
measurement exist.

## Target identity

**Verified on target:** VIP product/CID `0x1000003b`, VIPLite ABI
`0x00020003`, runtime marker `2.0.3.2-AW-2024-08-30`, `/dev/vipcore`, and a
runtime-visible count of one device/logical core. The logical count must not be
read as the number of physical NN engines. The target NPU devfreq readback
exposes 492, 852, and 1008 MHz.

**Verified in supplied tooling:** host SDK materials contain OpenVX, TIM-VX,
VSC/CLC, OpenCL, VXC, and EVIS markers. These markers establish inventory, not
target ABI compatibility or code-generation success.

## Capability classes

| Capability | Class | Current interpretation |
|---|---|---|
| Precompiled NBG loading and resident execution | Verified on target | A known UINT8 NBG completed repeated VIPLite runs. |
| Explicit map/flush/invalidate buffer lifecycle | Verified in supplied tooling and target runner | Measure visibility cost; do not call it zero-copy without an A/B. |
| Custom EVIS/VXC kernel inside an NBG | Verified on target | E003/E019/E020 paths execute and can pass an independent golden. |
| Native NN FC/Conv for the required Q1 carrier | Unknown | The next fused NBG gate must prove selected native execution and no DDR spill. |
| Packed Q1 coefficient mode | Unknown | Current direct kernel unpacks register-local signs; no native packed mode is assumed. |
| Dynamic shapes and KV-cache mutation | Unknown | Must be tested for the exact runtime and graph path. |
| Eight-core concurrent utilization | Unknown | Product marketing and runtime logical-core count do not prove it. |
| OpenCL availability on the target | Unknown | An ICD or host SDK string is not target execution evidence. |

## Runtime design consequences

Compile and load graphs outside the token loop. Keep networks and buffers
resident when comparing steady state. Record host run and device profiler time
separately, and include map/cache/synchronization overhead in end-to-end claims.
Public weights, binaries, NBGs, custom kernels/source, SDK or kernel patches,
and NPU tools may be published when redistributable and listed in the public
artifact manifest with SHA-256, byte size, origin, source commit,
build/runtime/toolchain provenance, and destination. Personal/sensitive data is
prohibited, including tokens, passwords, logins, private keys, identifiers, and
credentials. Existing payloads are not automatically tracked; use the manifest
and the Git LFS/release-assets guidance for an intentional large-file release.

## Required next proof

The next useful capability experiment is a small fused graph:

1. read canonical packed Q1 and Q8 inputs;
2. unpack only a tile in EVIS registers or proven on-chip memory;
3. feed a native UINT8/I8/FP16-compatible FC or Conv operation;
4. compare against an independent CPU golden;
5. measure H2D, run, device, D2H, total time, and memory traffic;
6. reject any path that expands the full weights or spills the intermediate to
   DDR without a measured end-to-end benefit.

## Evidence pointers

- [Target capability probe](../evidence/vip9000-next-capability-probe-2026-08-09.md)
- [Target VIPLite phase profile](../evidence/vip9000-phase-profile-2026-08-10.md)
- [Packed Q1 EVIS result](../evidence/q1-vip9000-evis-bonsai-2026-08-11.md)
- [MatrixMultiply repair and limits](../evidence/viplite-matrixmultiply-repair-2026-08-11.md)
- [Supplied toolchain inventory](../toolchain/inventory.md)
