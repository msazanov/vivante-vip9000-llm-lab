# Legacy-to-canonical migration map

The canonical layer is an index and interpretation boundary. It does not move
raw payloads or silently rewrite historical reports. Use the registry row and
the exact branch/commit pointer to inspect the original evidence.

| Legacy evidence | Canonical destination | Migration rule |
|---|---|---|
| Foundation README and agent instructions | `README.md`, `AGENTS.md`, `docs/objective.md` | Keep only current orientation in the root; retain raw history under its original refs. |
| Target-board inventory | `docs/hardware/a733.md`, `docs/hardware/orange-pi-zero-3w.md` | Summarize with evidence classes; keep read-only hashes and contradictions. |
| DDR frequency and PLL reports | Hardware safety section and E005/E025 registry rows | Preserve the 510 MHz controller readback distinction and secure-firmware boundary. |
| CPU baseline and optimization reports | `docs/architecture/backend-plan.md` and registry rows E026-E040 | Keep full-model claims separate from screens and rejected candidates. |
| VIPLite and packed-Q1 evidence | `docs/npu/vip9000-capabilities.md` and `docs/npu/q1-and-partitioning.md` | Point to exact evidence; public NBGs, SDK/kernel patches, weights, binaries, custom kernels/source, and NPU tools may be published only through the artifact manifest and privacy gate. |
| Profiling and result schemas | `docs/profiling/profiling-contract.md`, `docs/profiling/provenance-and-gates.md` | Use English canonical terminology; legacy language remains immutable. |
| E035 full-gate JSON | `docs/experiments/current-best.md` and registry E035 | Preserve exact `0.972497 tok/s` and token-quality SHA. |
| E044 screen/reset report | Registry E044 and current-best boundary | Keep screen `1.018629 tok/s` bounded; never promote after reset-before-quality. |
| E049d-v2 PMU summary | Registry E049d-v2 and current-best boundary | Keep `1.11698354 tok/s` diagnostic-only and non-end-to-end. |
| E049 NSI calibration v4 | Registry E049-nsi and `codex/e049-nsi-calibration@c8b7e2696c8f5df89d752dd41e55a5855354cca0` | Keep v4 fits bounded as inconclusive/low; they do not establish MB/s or memory saturation. |
| E055 active worktree | Registry E055 and branch inventory | Point-in-time snapshot only; the live branch may advance. Do not modify or import its ongoing raw work. |

## Pointer format

Every registry row carries a branch, full commit SHA, and one or more evidence
objects. A branch pointer uses this form:

```text
codex/e049c-arm-pmu@2b33f6fc03878fdd0f992d758efcecb71d3ae0d1
  experiments/E049d-steady-pmu-v2/data/summary.json
```

This format makes an audit reproducible even when a branch later advances. A
canonical summary must not replace the pointer with a copied raw file.

## Artifact and privacy boundary

Public weights, binaries, NBGs, custom kernels/source, SDK or kernel patches,
and NPU tools may be published when redistributable and recorded in
`public-artifact-manifest.json` with SHA-256, byte size, origin, source commit,
build/runtime/toolchain provenance, and destination. Personal/sensitive data is
prohibited, including tokens, passwords, logins, private keys, identifiers, and
credentials. Existing payloads and raw target evidence are not automatically
tracked or moved; large intentional releases use the `.gitattributes` Git LFS
patterns or release assets after review.
