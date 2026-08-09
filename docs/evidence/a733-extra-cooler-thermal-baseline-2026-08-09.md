# A733 extra-cooler CPU thermal baseline — 2026-08-09

## Scope and environment boundary

Run `a733-thermal-cpu-fan30-extra-cooler-001` is the first clean sustained CPU
thermal baseline after the operator installed an additional physical cooler.
The exact cooler model, placement, fan speed, and ambient temperature were not
measured, so the result applies only to the observed board configuration. The
approved 30 °C CPU fan-trip service was active and the profiled command was
wrapped by an independent 85 °C fail-safe guard.

The earlier run `a733-thermal-cpu-fan30-002` is retained as mixed evidence only:
the cooling hardware changed during its final recovery phase, so it is not a
valid before/after comparator.

## Workload and completeness

The workload contained 120 seconds idle, then 120 seconds each with 1, 4, and
8 CPU workers, followed by two 120-second recovery phases. It completed in
930.364 seconds with child exit `0`, 8,628 profiler samples, 1,200 guard samples
per measured phase, and a complete phase event stream. The profiler and guard
both sampled at 100 ms.

## Results

| Phase | CPU workers | CPU-big median / peak | CPU-little median / peak | CPU cooling | Fan median / max |
|---|---:|---:|---:|---:|---:|
| Idle | 0 | 34.038 / 38.874 °C | 34.162 / 40.362 °C | 0 | 4 / 4 |
| Measured 1 | 1 | 36.518 / 41.292 °C | 35.960 / 41.354 °C | 0 | 4 / 4 |
| Measured 2 | 4 | 42.966 / 47.926 °C | 41.602 / 49.476 °C | 0 | 4 / 4 |
| Measured 3 | 8 | 51.150 / 55.242 °C | 50.964 / 56.296 °C | 0 | 4 / 4 |

During the all-core measured phase, CPU policy 0 remained at 1,794,000 kHz and
policy 6 remained at 2,002,000 kHz in every sample. Both CPU-frequency cooling
devices remained at state `0`; the PWM fan's median and maximum state were
`4/4`. NPU devfreq remained at 1,008,000,000 Hz and swap counters did not
change. The kernel journal restricted to the exact run interval contained no
thermal, throttle, cpufreq, critical-temperature, or overheat event.

Under this exact fan-30 °C plus extra-cooler configuration, CPU thermal
throttling was absent for the tested duration. This does not predict NPU/DDR
thermals during model inference and does not remove the 85 °C abort ceiling.

## Evidence integrity

The result directory was copied from the target and every target/local file
hash matched. Large raw traces remain external to Git; these hashes bind this
summary to the retained evidence bundle:

| File | Bytes | SHA-256 |
|---|---:|---|
| `metadata.json` | 947 | `70a1592540dd0c42ae0eb57b2be23665060de52baa061d1e9bc98f0c5bb7bf74` |
| `phases.jsonl` | 1,134 | `23015238a59aa2a3a97e7562f3f37765f3b8ab04fb1f93b76123a66e867c689b` |
| `stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `stderr.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `telemetry.jsonl` | 11,133,030 | `e94076bd2a4f891896c545e86c0fb60de1680aae5e4f5406707dff2868324e6e` |
| `thermal-guard.jsonl` | 21,909,271 | `bc3f3055911dc682c71f1edc7e76d285253860d65d928e7a03306bb7f65412bd` |
| Sanitized analysis JSON | external | `2b93aa1ed7521d2942de4be41d8ebe5f51e78bfc219a8230d7b32c4ca1b50f06` |
