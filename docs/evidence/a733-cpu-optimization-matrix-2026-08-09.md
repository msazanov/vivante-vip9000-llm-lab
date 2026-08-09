# A733 CPU optimization matrix — 2026-08-09

## Evidence boundary

This is a source-audited test plan for the pinned PrismML llama.cpp commit
`38c66ad0241da4f9fcce541cda8edc219086cec5`. It does not claim measured gains.
The first canonical A76 observation used strict CPUs 6–7, two threads,
batch/ubatch 512, mmap, and the default repack path. Its median pp512/tg128
throughput was 1.582870/0.649836 tok/s.

Every candidate keeps the exact model, runtime, workload, seed, and affinity
evidence pinned. Thread, microbatch, and polling changes are intended to be
quality preserving. Flash attention, quantized KV, or compiler variants must
also pass deterministic output comparison before promotion.

## What the pinned source actually dispatches

- `ggml/src/ggml-cpu/arch/arm/quants.c:149` contains the standard Q1_0 ARM
  implementation. Its DOTPROD path uses `ggml_vdotq_s32` around lines 249–258;
  the fallback NEON path begins around line 266.
- `ggml/src/ggml-cpu/repack.cpp:4958` registers Q1_0 repack traits. The A733
  DOTPROD-without-I8MM branch around lines 5122–5131 selects the Q1_0 `4x4`
  layout.
- Specialized ARM Q1 GEMV is in
  `ggml/src/ggml-cpu/arch/arm/repack.cpp:1844`; GEMM begins around line 5302.
  The `4x8` route requires I8MM and is not the expected A733 route.
- Runtime ARM capabilities distinguish DOTPROD (`HWCAP_ASIMDDP`) from I8MM in
  `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp:25`.
- KleidiAI configuration in `ggml/src/ggml-cpu/CMakeLists.txt:622` provides
  Q4/Q8 kernels, not a Q1_0 fast path for this model.

Consequently, the accepted native build already exercises the relevant
quality-preserving Q1_0 DOTPROD repack. A generic backend toggle is not an NPU
substitute, and enabling KleidiAI is not assumed to accelerate this artifact.

## Ranked A/B sequence

| Priority | Variable | Pinned candidates | Primary metric | Quality risk |
|---:|---|---|---|---|
| 1 | CPU topology / threads | A76 `t1/t2`; A55 `t4/t6`; all-core `t6/t8` | decode first, then prefill | none expected |
| 2 | Prompt microbatch | pp512 with `ubatch=64/128/256/512`, batch 512 | prefill tok/s and peak RSS | none expected |
| 3 | Polling | `poll=0/1/25/50/100` on the winning partition | decode, TTFT, thermals | none expected |
| 4 | Repack control | `llama-completion --repack/--no-repack` | decode and deterministic output | none expected, verify |
| 5 | Flash attention | `off/on/auto`, only after ctx 1024/2048 baseline | long-context decode/TTFT | numerical path; verify |
| 6 | Quantized KV | q8 K/V separately, flash enabled | long-context memory/tok/s | possible quality loss |
| 7 | mmap / direct I/O | mmap on/off, direct I/O separately | cold load only | none expected |
| 8 | Compiler tuning | reference versus a separately named A76-tuned build | end-to-end tok/s | portability/SIGILL risk |

Do not change multiple rows at once. Use at least one warmup and five measured
repetitions, retain individual samples, and compare against the immediately
preceding pinned reference.

For every row retain both the requested affinity and the effective affinity
read from `/proc/<pid>/status` for the model, profiler, and thermal guard. A
partitioned run must state which core is reserved for collection. An all-core
run has no reserved core, so it requires a paired control with and without the
profiler/guard before collector overhead can be separated from model speed.
Do not mix an unprofiled all-core number into the canonical profiled matrix.

## Source constraints that shape the matrix

- `tools/llama-bench/llama-bench.cpp:411` defines defaults including ubatch
  512, mmap enabled, direct I/O disabled, and polling 50.
- Prompt execution consumes `n_batch` around lines 2087–2114, while generation
  runs one token at a time around lines 2116–2132. Batch/ubatch tuning therefore
  targets prefill much more than steady-state decode.
- Threadpool affinity and polling are applied around lines 2298–2325.
- Repack is a common runtime option in `common/arg.cpp:2037`, but this pinned
  `llama-bench` does not expose a repack switch. Its model defaults enable extra
  buffers/repack through `tools/llama-bench/llama-bench.cpp:1175` and
  `src/llama-model.cpp:2267`. A repack A/B therefore uses completion or a
  separately named harness, not an invented benchmark flag.
- CPU flash-attention selection is in `src/llama-graph.cpp:2120`; optimized
  split/tiled CPU paths are constrained in `ggml/src/ggml-cpu/ops.cpp:8898`.
  Short tg128 is unlikely to be the best place to evaluate it.
- Quantized V validation in `src/llama-context.cpp:3823` requires flash
  attention.
- mmap/direct-I/O paths in `src/llama-model-loader.cpp:556` and
  `src/llama-mmap.cpp:186` mainly change loading and page-in behavior, not a
  warmed decode kernel.
- The current portable build uses `armv8.2-a+dotprod`. The architecture flag
  logic is in `ggml/src/ggml-cpu/CMakeLists.txt:169`. Do not replace the
  heterogeneous-board reference with an implicit `-mcpu=native` build.

## Interpretation rule

The A733 has only 12 GiB and limited memory bandwidth. More threads can lose to
fewer threads once Q1 weight streaming saturates memory, even while clocks and
temperatures remain healthy. Prefer the smallest thread set whose five-repeat
decode median is genuinely higher and whose CV, RSS, swap, and thermal evidence
remain acceptable. A speed result without TTFT and deterministic output
evidence remains `unqualified`.

`llama-bench` establishes repeatable prefill/decode throughput, not quality or
application TTFT. The winning throughput configuration therefore needs a
separate pinned `llama-completion` run with deterministic token comparison and
explicit model-load, prompt-processing, first-token, steady-decode, and total
wall-clock phases before it can become `qualified`.

## Per-run control evidence

Record the following beside every matrix result:

- ambient/configuration note, extra-cooler presence, fan policy, and pre-run
  temperature/fan state after cooldown;
- all thermal zones by stable `type`, with start/min/median/p90/max/end values;
- fan current/max state and PWM when exposed; cpufreq policy limits, observed
  frequency, cooling state, and GPU/NPU devfreq observations;
- guard threshold, sample count, termination reason, peak RSS/HWM, minimum
  available memory, and swap delta;
- exact kernel-journal interval and counts for thermal/cpufreq throttling, OOM,
  and accelerator-driver faults;
- prompt/decode samples plus min/p10/median/p90/max/sample-CV, rather than only
  an aggregate average.

The 85 C guard is an emergency termination threshold. It is not the fan trip
temperature and it is not evidence that the CPU avoided frequency throttling;
frequency/cooling and kernel-event evidence must establish that separately.
