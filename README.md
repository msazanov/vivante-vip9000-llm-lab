# Vivante VIP9000 LLM Lab

This repository is an evidence-driven lab for local LLM inference on an
Allwinner A733 with a Vivante VIP9000-class NPU.

## Objective

The engineering objective is to exceed **1 tok/s steady-state decode for the
full Bonsai model** while preserving deterministic quality, safe thermals,
reproducible provenance, and a CPU fallback. The current qualified full-model
best is **E035 at exactly 0.972497 tok/s**. See the [qualified full-model best
record](docs/experiments/current-best.md#qualified-full-model-best). The
objective is not met yet.

The target SoC has **6x Cortex-A55 cores plus 2x Cortex-A76 cores**. The SoC
family exposes a **32-bit memory interface** and supports **LPDDR5-4800**;
`32-bit x 4800 MT/s = 19.2 GB/s` is a **theoretical SoC ceiling only**. It is
not a measurement of this board's effective data rate or sustained bandwidth.
The current controller readback is **510 MHz**. The effective LPDDR data rate,
sustainable bandwidth under this workload, and firmware training state remain
unknown.

Any frequency or voltage experiment must remain within the board's supported
firmware/BSP and thermal safety envelope. Do not bypass secure firmware,
kernel safety controls, or the approved overclock boundary with ad-hoc register
writes.

## Read this first

- [Objective and success criteria](docs/objective.md)
- [A733 hardware facts and evidence classes](docs/hardware/a733.md)
- [Target-board observations](docs/hardware/orange-pi-zero-3w.md)
- [Backend bottleneck and implementation plan](docs/architecture/backend-plan.md)
- [VIP9000 capabilities](docs/npu/vip9000-capabilities.md)
- [Q1 format and partitioning policy](docs/npu/q1-and-partitioning.md)
- [Profiling contract](docs/profiling/profiling-contract.md)
- [Provenance and promotion gates](docs/profiling/provenance-and-gates.md)
- [Current best and interpretation boundaries](docs/experiments/current-best.md)
- [Machine-readable experiment registry](docs/experiments/registry.json)
- [Registry schema](docs/experiments/schema.json)
- [Remote branch inventory](docs/experiments/branch-inventory.md)
- [Migration map](docs/experiments/migration-map.md)

## Current interpretation

The optimized CPU Q1 path is the reference because it is the only full-model
path with a qualified deterministic result. The programmable EVIS path can
read packed Q1 weights without expanding a full INT8 tensor, but the measured
large Bonsai tile is slower than the CPU full layer. The next NPU gate is a
single fused graph in which EVIS performs only local unpacking and a native NN
operation performs the dense dot product while intermediate data stays on-chip.

E049d-v2 reports **1.11698354 tok/s**, but only for a three-token steady marker
window. It excludes model load, prompt evaluation, and end-to-end request
timing; it is diagnostic and is not an optimization or current-best claim.
E044's short `1.018629 tok/s` screen is also not a qualified full-model result:
its sustained attempts reset the board before throughput and quality output.

## Evidence boundary

Canonical documents summarize facts and point to immutable evidence. Historical
Russian-language reports and raw external traces remain legacy evidence and are
not rewritten by this branch. Public weights, binaries, NBGs, custom
kernels/source, SDK or kernel patches, and NPU tools may be published when
redistributable and recorded in the [public artifact manifest](docs/experiments/public-artifact-manifest.json)
with SHA-256, byte size, origin, source commit, build/runtime/toolchain
provenance, and destination. Personal/sensitive data is prohibited, including
tokens, passwords, logins, private keys, identifiers, and credentials. This
policy does not automatically track existing payloads or copy raw evidence.

Use the [profiling contract](docs/profiling/profiling-contract.md) for every
new run and the [registry](docs/experiments/registry.json) for status and
promotion decisions.
