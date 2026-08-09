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
