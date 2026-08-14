# Remote branch inventory

This is a point-in-time inventory of `msazanov/vivante-vip9000-llm-lab` remote
heads captured on 2026-08-14. The machine-readable source is
[branch-inventory.json](branch-inventory.json); every commit is a full SHA.
The canonical branch was forked from
`codex/profiling-foundation@c071476773ad0f7fc499b6a39270a98bc1e25878`.

| Branch | Head | Role |
|---|---|---|
| `main` | `426d1467563da211f9621cc2692ec25cf065880d` | Original E001/E002 research track |
| `codex/profiling-foundation` | `c071476773ad0f7fc499b6a39270a98bc1e25878` | Requested foundation anchor |
| `codex/e022-fused-q1` | `d5ac6d0a4570b0a5cab761a2f5afd046baaa67a4` | Fused Q1 NPU evidence |
| `codex/e023-ggml-q1-seam` | `7a79466f7f65a67d64c0624eabf60d63652960b8` | CPU seam, E035, E044, E045 history |
| `codex/e047-hard-profiling` | `203f3db5454ab99a4b39b2376d43a2ccfdca77f7` | Common benchmark contract |
| `codex/e048-per-op-trace` | `f5bd54eebbfc12c7399a64fd0f64ab56fa9172f5` | Per-operation trace |
| `codex/e049-nsi-calibration` | `c8b7e2696c8f5df89d752dd41e55a5855354cca0` | NPU identity calibration |
| `codex/e049b-trace-analysis` | `06a810c2f3330f51ec0a8d97de041c4fdde8a452` | E049b trace analysis |
| `codex/e049c-arm-pmu` | `2b33f6fc03878fdd0f992d758efcecb71d3ae0d1` | E049c and E049d-v2 |
| `codex/e054-a76-q1-multiversion` | `9ac72e4e7486d85b427ae93d73dcb2ed5ddee8c0` | A76 Q1 hypothesis |
| `codex/e055-q1-hot-cold` | `51d1c1cfd3d2e963344e79dc11719c278b234569` | Ongoing E055; do not modify |

All listed branches remain separate. The canonical layer uses branch/commit
pointers and does not merge histories or rewrite legacy experiment payloads.
In particular, E055 continues in its own branch and worktree; this repository
index contains only its exact pointer and planned status.
