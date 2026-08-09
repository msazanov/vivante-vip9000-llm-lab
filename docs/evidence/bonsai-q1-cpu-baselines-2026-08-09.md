# Bonsai 27B Q1_0 CPU baselines — 2026-08-09

## Scope

These runs establish CPU throughput and thermal references before Vivante
VIP9000 offload. They do not qualify model quality. The pinned artifact is
`Bonsai-27B-Q1_0.gguf`, 3,803,452,480 bytes, SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
It is GGML `Q1_0` type 41 with group size 128, not packed `TQ1_0`.

The native runtime is PrismML llama.cpp build 9594 at commit
`38c66ad0241da4f9fcce541cda8edc219086cec5`. The board has six A55 cores in
policy 0 and two A76 cores in policy 6. The physical fan uses the validated
30 °C trip policy, and an additional external cooler was installed before the
canonical throughput sweep.

## Retained pre-baseline attempts

All model executions remain in the model-card table, including invalidated
attempts:

| Run | Exit | Outcome |
|---|---:|---|
| `bonsai27b-q1-cpu-a76-smoke-001` | 143 | `llama-cli` entered an EOF/prompt-output loop; compressed stdout retained; use `llama-completion` |
| `bonsai27b-q1-cpu-a76-smoke-002` | 0 | bounded eight-token `llama-completion` smoke; stopped inside `<think>`, so no quality or TTFT qualification |
| `bonsai27b-q1-cpu-a76-pp512-tg128-001` | 143 | `-pg 512,128` was additive to defaults and produced three workloads; terminated before measurement |
| `bonsai27b-q1-cpu-a76-pp512-tg128-002` | 143 | correct workload, but collectors were not isolated from A76; terminated before measurement |

These failures are useful protocol evidence, not performance samples. Run 003
corrected workload selection and collector affinity; the A55 run additionally
used the corrected automatic profiler lifecycle.

## A76 pp512/tg128 run 003

- Run ID: `bonsai27b-q1-cpu-a76-pp512-tg128-003`
- Model affinity: CPUs 6–7; two threads; strict mask `0xc0`
- Profiler and thermal guard affinity: CPU 0
- Workloads: separate pp512 and tg128, batch/ubatch 512, one warmup and five
  measured repetitions each
- Exit: 0; elapsed: 2,951.326 s
- Prompt samples: 1.584820, 1.582870, 1.582550, 1.583030, 1.582660 tok/s;
  median 1.582870 tok/s; sample CV 0.0589%
- Decode samples: 0.652354, 0.649546, 0.651914, 0.649836, 0.649455 tok/s;
  median 0.649836 tok/s; sample CV 0.2147%
- Peak workload RSS: 7,483,988 KiB (7,308.582 MiB)
- Peak CPU-zone temperature: 69.089 °C; median hottest CPU-zone temperature:
  63.488 °C
- Both cpufreq cooling devices remained at state 0; PWM fan median/max state
  was 4; swap delta was 0; the exact kernel interval contained no matching
  throttle, thermal, OOM, cpufreq, NPU, VIP, or galcore event

The raw phase file is empty because this run used the profiler version from
before automatic lifecycle events were added. Raw files are immutable and were
not repaired after the fact. Therefore the throughput is recorded as
`unqualified`, despite its clean exit and stable samples. The raw bundle is
retained on the target and in the external local evidence store; the canonical
compact summary records every raw SHA-256.

## Instrumentation correction

The profiler now emits `profiler_start` and `profiler_complete`, enforces a
stable process-group append boundary, and the exact-statistics summarizer uses
a bounded-memory, temporary on-disk SQLite spool. A live guarded `/bin/true`
probe on the board produced two valid lifecycle events and summarized with
exit 0. Subsequent A55 and all-core runs use this corrected instrumentation.

## A55 pp512/tg128 run 001

- Run ID: `bonsai27b-q1-cpu-a55-pp512-tg128-001`
- Model affinity: CPUs 0–5; six threads; strict mask `0x3f`
- Profiler and thermal guard affinity: CPU 7
- Workloads: separate pp512 and tg128, batch/ubatch 512, one warmup and five
  measured repetitions each
- Exit: 0; elapsed: 3,067.786 s; lifecycle events:
  `profiler_start`, `profiler_complete`
- Prompt samples: 1.444660, 1.444670, 1.444010, 1.443890, 1.444640 tok/s;
  median 1.444640 tok/s; sample CV 0.0270%
- Decode samples: 0.726325, 0.726362, 0.725669, 0.725747, 0.726175 tok/s;
  median 0.726175 tok/s; sample CV 0.0449%
- Peak workload RSS: 7,483,548 KiB (7,308.152 MiB); minimum available memory:
  7,421,152 KiB; swap delta: 0
- Hottest CPU-zone peak/median/p90: 63.612/56.854/59.272 °C
- Both cpufreq cooling devices remained at state 0; policy 0 stayed at
  1,794,000 kHz; PWM fan median/max state was 4
- The exact 18:49:19.883784–19:40:27.670252 UTC kernel interval contained no
  matching throttle, thermal, cpufreq, OOM, NPU, VIP, GPU, or galcore event

The external cooler was present throughout. All five samples are highly
stable and the raw hashes match between target and the external local evidence
store. The run remains `unqualified` because `llama-bench` does not supply the
application TTFT or deterministic generated-token quality contract.

## First topology comparison

| Partition | Prompt median tok/s | Decode median tok/s | Peak CPU °C | Peak RSS MiB |
|---|---:|---:|---:|---:|
| A76 CPUs 6–7, t2 | 1.582870 | 0.649836 | 69.089 | 7308.582 |
| A55 CPUs 0–5, t6 | 1.444640 | 0.726175 | 63.612 | 7308.152 |

Against the isolated A76 pair, the A55 cluster is 8.733% slower for pp512 but
11.747% faster for tg128. This is the expected trade-off for a bandwidth-heavy
Q1 model: the two large cores win prompt matrix work while six little cores
provide the better current single-token decode. It does not yet identify the
best end-to-end partition. The next CPU topology test must measure all eight
cores and separately quantify the profiler/guard overhead because no core can
be reserved for collection.
