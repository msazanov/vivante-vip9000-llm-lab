# Objective and decision policy

## Primary objective

Exceed **1 tok/s steady-state decode for the full Bonsai model** on the Orange
Pi Zero 3W / Allwinner A733, with deterministic output agreement, controlled
thermals, reproducible software identity, and a CPU fallback for unsupported or
unprofitable work.

The objective is deliberately end to end. A kernel, marker window, device
profiler value, or short screen can guide engineering, but none is a full-model
result unless model loading, prompt/decode boundaries, quality, and safety gates
are declared and measured.

## KPI hierarchy

1. Primary: steady-state full-model decode tokens per second.
2. Secondary: prompt/prefill tokens per second and time to first token.
3. Safety and feasibility: peak RSS, swap delta, CPU/NPU/GPU/DDR thermal zones,
   cooling state, frequency state, failure rate, and power when observable.
4. Correctness: deterministic token agreement for generation; independent CPU
   golden comparison for operators; a pinned task or perplexity check for model
   format changes.

Never combine these units. A ShuffleNet inference per second, a Q1 operator
latency, and a Bonsai token rate are different measurements.

## Current reference

The current qualified full-model best is E035, `0.972497 tok/s`, with exact
deterministic output agreement against the pinned reference. Its authoritative
evidence is retained on `codex/e023-ggml-q1-seam` and is listed in the
[registry](experiments/registry.json) and [current-best record](experiments/current-best.md).

E049d-v2's `1.11698354 tok/s` is a diagnostic three-token marker window. It
starts after prompt evaluation and excludes model load, prompt processing,
request setup, and end-to-end output. It cannot qualify or optimize the model.

E044's short `1.018629 tok/s` PRFM screen is a ranking observation only. All
sustained attempts reset the board before a throughput and quality gate, so it
is not accepted and is not the current best.

## Base verification caveat

The foundation anchor has three pre-existing failures in
`tests/test_fused_q1_fc_source.py`: the expected
`experiments/E009-fused-q1-native-fc/q1_unpack_u8_evis.vx` source is absent on
that base. The canonical-layer verification records these failures separately;
they are not hidden or attributed to the registry/checker changes.

## Target constraints

The SoC family has 6x Cortex-A55 plus 2x Cortex-A76 cores and a 32-bit memory
interface. LPDDR5-4800 implies a **19.2 GB/s theoretical SoC ceiling**; this
number must never be presented as measured sustained bandwidth. Read-only target
clock evidence reports a 510 MHz memory-controller rate, while effective data
rate, training state, and sustained bandwidth remain unknown. See the
[hardware classification](hardware/a733.md).

The board's supported firmware/BSP and thermal safety boundary controls any
frequency experiment. A raw PLL or voltage write is not a valid optimization
unless the secure firmware path, readback, watchdog/recovery, thermal guard,
quality gate, and rollback are independently approved and recorded.

## Decision rule

Promote a candidate only when it has:

- an exact repository/runtime/model/workload identity;
- separate prefill, TTFT, and decode measurements with repetitions and
  dispersion statistics;
- a pinned compatible CPU reference;
- an independent quality result with a declared threshold;
- complete raw evidence or a safe pointer to an immutable external bundle;
- no unexplained thermal, memory, reset, or fallback behavior.

Otherwise retain the run with the most accurate status: `unqualified`,
`rejected`, `failed`, `diagnostic`, or `planned`.
