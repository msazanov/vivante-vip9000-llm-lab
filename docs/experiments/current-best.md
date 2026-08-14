# Current best and bounded diagnostics

## Qualified full-model best

**E035 is the only current best:** `0.972497 tok/s` for full Bonsai decode,
with exact deterministic token agreement. The authoritative result is
`experiments/E035-poll-screen/results/e035_poll.json` at
`codex/e023-ggml-q1-seam@d012490294a24a17f4ec97707f35911c53e4a2e4`.

The row is `accepted`, `claim_class=full_model`, and
`is_current_best=true` in the [registry](registry.json). The shorter E035
screen value `0.991160 tok/s` is not the headline result; the full gate uses
the three measured samples and the exact quality SHA.

## What is not the current best

### E044: PRFM screen

E044 reached `1.018629 tok/s` in a short `n=8/r=1` ranking screen. Three
sustained full attempts reset the board before throughput and quality output.
The row is `unqualified`, has no full-gate decode value, and is never current
best. Its exact evidence pointer is
`experiments/E044-cluster-prfm/README.md` at
`codex/e023-ggml-q1-seam@3193c0e8bc1b00dfd4cf70c65a193a3482e67ca7`.

### E049d-v2: PMU marker

E049d-v2 reports `1.11698354 tok/s` for a median three-token steady marker
window. It excludes model load, prompt evaluation, request setup, output
handling, and end-to-end timing. It is `diagnostic`, `end_to_end=false`, and
`optimization_claim=false`; it does not qualify or optimize the full model.
The exact summary pointer is
`experiments/E049d-steady-pmu-v2/data/summary.json` at
`codex/e049c-arm-pmu@2b33f6fc03878fdd0f992d758efcecb71d3ae0d1`.

## Promotion rule

Only an exact full-model run with a pinned CPU reference, deterministic quality,
separate prompt/TTFT/decode measurements, complete safety evidence, and an
immutable provenance pointer can replace E035. Operator tiles, PMU windows,
short screens, and diagnostics remain useful for choosing the next experiment,
but they are not interchangeable with the full-model KPI.

See the [objective](../objective.md),
[profiling contract](../profiling/profiling-contract.md), and
[provenance gates](../profiling/provenance-and-gates.md).
