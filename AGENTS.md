# Agent guide

This repository is a hardware research lab, not a collection of informal
benchmarks. The primary objective is a qualified full Bonsai decode result
above 1 tok/s on the A733 target while preserving deterministic quality and
safe, reproducible operation. Begin with [README.md](README.md), then read the
[objective](docs/objective.md), [hardware facts](docs/hardware/a733.md),
[profiling contract](docs/profiling/profiling-contract.md), and
[experiment registry](docs/experiments/registry.json).

## Non-negotiable rules

1. Classify every statement as `Verified on target`, `Verified upstream`,
   `Verified in supplied tooling`, `Prior local observation`, `Hypothesis`, or
   `Unknown`. Keep the class beside the claim when ambiguity matters.
2. Prefer primary target files, exact SDK documentation, official vendor or
   upstream sources, and reproducible measurements. Do not turn a marketing
   TOPS number or a similar board's specification into a target fact.
3. Preserve raw evidence and exact provenance. Do not overwrite failed runs,
   rewrite legacy reports, or copy proprietary binaries, model weights,
   firmware, NBGs, credentials, or private SDK material into Git.
4. Measure end to end. Include setup, graph load, allocation/import, packing,
   cache maintenance, copies, synchronization, execution, memory, clocks,
   thermals, failure state, and quality. Device time alone is never a model
   throughput result.
5. Keep prompt/prefill throughput, TTFT, and steady decode throughput separate.
   A marker window, microbenchmark, or screen cannot become a full-model claim.
6. Use the registry status taxonomy exactly: `accepted`, `rejected`, `failed`,
   `diagnostic`, `unqualified`, and `planned`. Faster without required quality
   evidence is `unqualified`; a failed quality threshold is `rejected`.
7. Keep all memory statements precise. `LPDDR5-4800` and `19.2 GB/s` are the
   theoretical 32-bit SoC ceiling, not measured board bandwidth. The current
   controller readback is 510 MHz; effective data rate and sustained bandwidth
   are unknown.
8. Frequency and voltage work must stay within secure firmware/BSP and thermal
   safety limits. Never bypass safety controls with an unreviewed raw register
   write.

## Working procedure

- Use Beads for durable task tracking when working across sessions.
- Create or update a design before changing architecture; use TDD for tooling
  and behavior changes.
- Work in an isolated branch/worktree. Preserve all existing experiment
  branches, especially the ongoing `codex/e055-q1-hot-cold` work.
- Run focused tests first, then the relevant full suite. Do not hide baseline
  failures or claim a green suite without fresh command output.
- Commit and push only the intended canonical files. Record the exact branch,
  commit, test command, and result in the handoff.

## Canonical layout

| Path | Purpose |
|---|---|
| `docs/objective.md` | Objective, KPI hierarchy, and success criteria |
| `docs/hardware/a733.md` | Evidence-classed SoC and memory facts |
| `docs/hardware/orange-pi-zero-3w.md` | Target-board identity and read-only observations |
| `docs/architecture/backend-plan.md` | Bottleneck diagnosis and backend sequence |
| `docs/npu/vip9000-capabilities.md` | Capability evidence and limits |
| `docs/npu/q1-and-partitioning.md` | Q1 representation and CPU/EVIS/NN partitioning |
| `docs/profiling/profiling-contract.md` | Run fields, telemetry, phases, and statistics |
| `docs/profiling/provenance-and-gates.md` | Provenance, quality, safety, and promotion gates |
| `docs/experiments/registry.json` | Append-only machine-readable experiment index |
| `docs/experiments/current-best.md` | Human interpretation of current best |
| `docs/experiments/branch-inventory.*` | Exact remote branch refs |
| `docs/experiments/migration-map.md` | Legacy evidence to canonical pointers |
| `tooling/check_repository_index.py` | Offline deterministic canonical checks |

Raw historical evidence under `docs/evidence/`, existing experiment branches,
and external run bundles are immutable inputs. A canonical document may point
to them by path and exact branch/commit; it must not silently import their
payloads or change their language.
