# E055 direct phase timing-qualification failure

Status: **INVALID PHASE — 18 qualified raw samples, 2 invalid raw samples**.

No sample was deleted or replaced. No model was run. The two invalid samples
are `cpu6-pair1-hot` and `cpu6-pair3-hot`. Both violate the accepted one-sided
clock-domain qualifier `T + 1,000,000 ns >= H`, although E049c reports
`sample_valid=true`, all three event ratios are exactly `1.0`, and all S/A/E
Boolean markers are true.

## Exact evidence paths

Every attempted run has five immutable raw roles under `raw/<run-id>/`:

- `e049c.json` — parent interval `M`, perf `T/R`, thermal state, PMU counts,
  command, and S/A/E Boolean state;
- `harness.stdout.raw` — harness interval `H`, calls, golden result,
  conditioning, checksum, and S/A/E Boolean state;
- `harness.stderr.raw`, `wrapper.stdout.raw`, and `wrapper.stderr.raw` — empty
  for all 20 attempted phase runs.

For example, the first invalid pair is in
`raw/cpu6-pair1-hot/{e049c.json,harness.stdout.raw}`. The complete numeric
comparison is in `timing-comparison.csv`.

## Clock and marker boundaries in source

The clocks are deliberately not assumed to share a domain:

1. The child writes `S` and waits for `A` in
   `tooling/e055_q1_hotcold.cpp:412-428`.
2. The parent observes `S`, resets/enables the perf group, records the
   `CLOCK_MONOTONIC` start for `M`, and sends `A` in
   `tooling/a733_pmu_exec.c:1712-1723`.
3. After the ACK read returns, the child records a `CLOCK_MONOTONIC_RAW` start,
   executes the requested calls, records `H`, and then writes `E` in
   `tooling/e055_q1_hotcold.cpp:593-620`.
4. The parent observes `E`, disables the perf group, and records `M` in
   `tooling/a733_pmu_exec.c:1723-1729`.
5. E049c reads perf's `time_enabled` (`T`) and `time_running` (`R`) after the
   group is disabled in `tooling/a733_pmu_exec.c:721-761`.

The raw schema stores only whether S, A, and E were observed. It does **not**
store absolute or relative timestamps for individual marker operations, so no
marker timestamp may be reconstructed from this phase. The requested CPU is
present, but per-sample cpufreq sysfs values were also not captured. The
`cpu_cycles / T` column is an explicitly derived PMU rate, not a captured
cpufreq reading.

## Comparison

All three events in each group report the same T/R value. Important rows are:

| Run | H (ns) | M (ns) | T=R (ns) | M-H (ns) | T-H (ns) | calls | cycles/T (GHz, derived) | temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU6 pair1 hot — invalid | 9,904,416 | 10,049,245 | 8,597,252 | 144,829 | -1,307,164 | 250 | 0.811520 | 33.542 °C |
| CPU6 pair2 hot — valid | 3,618,041 | 3,697,456 | 3,670,791 | 79,415 | 52,750 | 250 | 1.894130 | 33.542 °C |
| CPU6 pair3 hot — invalid | 11,023,709 | 11,181,328 | 9,947,127 | 157,619 | -1,076,582 | 250 | 0.710503 | 33.852 °C |
| CPU6 pair4 hot — valid | 3,435,125 | 3,546,957 | 3,510,500 | 111,832 | 75,375 | 250 | 1.997865 | 33.976 °C |
| CPU6 pair5 hot — valid | 5,681,292 | 5,736,080 | 5,718,250 | 54,788 | 36,958 | 250 | 1.194824 | 34.100 °C |

The five CPU0 hot rows remain inside the accepted tolerance. Their T-H values
range from -697,209 ns to +82,334 ns, and their derived cycles/T values range
from 1.326519 to 1.609661 GHz. Their exact values are retained in the CSV.

The two invalid A76 runs still have nearly the same work counts as the valid
A76 runs: 6.98/7.07 million cycles versus 6.83–7.01 million cycles and about
18.8 million instructions throughout. What changes is wall duration and the
derived cycles/T rate.

## Single root-cause hypothesis

The strongest current hypothesis is **child descheduling combined with perf
per-task enabled-time accounting**. `H` is child wall time and therefore
continues across descheduling. The per-process perf context's `T` can exclude
time in which the measured task is not scheduled. The two slow A76 hot runs
appear to contain approximately 1.1–1.3 ms of such time. This explains all
three observations at once:

- `M` still encloses `H` with only 0.145–0.158 ms parent/marker overhead;
- all three T/R values agree exactly and retain ratio 1.0;
- work counts stay nearly constant while wall latency and cycles/T vary
  sharply under the `ondemand` governor.

This remains a source- and data-grounded hypothesis, not a proven scheduler
trace. Proving it requires an independently approved short diagnostic that
captures context switches and/or task-clock alongside marker timestamps. No
such run has been started.

## Why the phase continued after the first invalid sample

The direct OpenSSH phase replaced the rejected remote-helper path. Its manual
per-run gate checked E049c status, E049c `sample_valid`, S/A/E Booleans, ratio
1.0, affinity, golden/checksum, thermals, and process cleanup, but omitted the
cross-file `H/M/T` qualifier. E049c cannot apply that check alone because `H`
is emitted in the separate child stream, and E049c's own `sample_valid` checks
only PMU support plus positive T/R and the configured running-ratio floor.

The reviewed executor also currently parses the two JSON objects but does not
perform the semantic cross-file timing qualification before advancing to the
next run. A preserved RED test now demonstrates this exact defect: an initial
sample with `T=R=H-1,000,001 ns` does not stop the current executor. No
production change is made here pending root-cause review.
