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
| VIPLite and packed-Q1 evidence | `docs/npu/vip9000-capabilities.md` and `docs/npu/q1-and-partitioning.md` | Point to exact evidence; do not copy NBGs, SDK files, or weights. |
| Profiling and result schemas | `docs/profiling/profiling-contract.md`, `docs/profiling/provenance-and-gates.md` | Use English canonical terminology; legacy language remains immutable. |
| E035 full-gate JSON | `docs/experiments/current-best.md` and registry E035 | Preserve exact `0.972497 tok/s` and token-quality SHA. |
| E044 screen/reset report | Registry E044 and current-best boundary | Keep screen `1.018629 tok/s` bounded; never promote after reset-before-quality. |
| E049d-v2 PMU summary | Registry E049d-v2 and current-best boundary | Keep `1.11698354 tok/s` diagnostic-only and non-end-to-end. |
| E055 active worktree | Registry E055 and branch inventory | Pointer only; do not modify or import its ongoing raw work. |

## Pointer format

Every registry row carries a branch, full commit SHA, and one or more evidence
objects. A branch pointer uses this form:

```text
codex/e049c-arm-pmu@2b33f6fc03878fdd0f992d758efcecb71d3ae0d1
  experiments/E049d-steady-pmu-v2/data/summary.json
```

This format makes an audit reproducible even when a branch later advances. A
canonical summary must not replace the pointer with a copied raw file.

## What remains outside the canonical layer

Vendor SDK archives, runtime libraries, firmware, model weights, NBG files,
credentials, private reports, and external raw target bundles remain external
or on the authoritative experiment branch. Their hashes and sanitized metadata
may be cited; payload duplication is prohibited.
