# Profiling contract

Every experiment is evidence only when its identity, workload, raw artifacts,
measurements, quality result, and interpretation are retained together. Missing
sensors remain missing; they must not be replaced with invented values.

## Run identity

Each run has a unique `run_id`. Record:

- repository commit and upstream runtime commit;
- model filename, format, group size, byte size, and SHA-256;
- compiler, SDK, driver, kernel, command, and runtime identity;
- target board, OS, kernel, device identifier, and relevant paths;
- context, batch, micro-batch, thread count, affinity, governor, cooling,
  thermal limit, seed, sampling settings, and deterministic mode.

Model and runtime hashes may be unresolved while planning, but a `qualified`
result must resolve them.

## Workload and statistics

Measure prompt/prefill, time to first token, and steady decode separately.
Declare warmups and measured repetitions; keep every sample and every failed
attempt. Report median, p10, p90, minimum, maximum, and coefficient of
variation with explicit units. A single average or a marker window is not a
full-model decode result.

For a qualified generation result, the machine contract contains:

```text
software: repository_commit, runtime, runtime_commit, compiler, sdk,
          driver, kernel, command
workload: tokenizer, prompt_suite, prompt_tokens, generated_tokens, seed,
          deterministic, warmup_iterations
performance.statistics: prompt_tps, decode_tps, ttft_ms
```

Each statistic contains `median`, `p10`, `p90`, `min`, `max`, `cv`, and
`samples`. Keep `model` and `configuration` separate from these objects.

## Raw artifacts and phase boundaries

The standard profiler writes `metadata.json`, `stdout.log`, `stderr.log`,
`telemetry.jsonl`, and a phase file under a new immutable run directory. It
records direct-child process state plus board-wide readable `/proc` and `/sys`
telemetry; it is not a claim of complete process-tree, kernel, or vendor-driver
accounting.

When the runtime exposes them, record startup, VIP initialization, graph load,
buffer allocation/import, input preparation and cache flush, first run, warm
run, synchronization, output invalidate/copy, teardown, and copy byte counts.
Keep setup and first-run costs separate from the steady KPI.

For shared A733 memory:

- `H2D` is the host-to-device visibility path, including map, copy, unmap, and
  cache flush; it is not a PCIe transfer to dedicated VRAM.
- `D2H` is the device-to-host visibility path, including invalidate and copy.
- `run` is wall time around the synchronous runtime call.
- `device` is the runtime-reported device interval.
- `golden` is an independently computed reference. Equality with the first
  device output is repeatability, not correctness.

## Quality and promotion

Pin a CPU reference before interpreting a candidate. Generation requires exact
deterministic token agreement for the same model, prompt suite, context, seed,
sampling, and software. Operator work requires an independent CPU golden and
appropriate numerical thresholds. Model-format changes require a pinned task
or perplexity evaluation in addition to a smoke test.

The exact status taxonomy is defined in the
[provenance and gates](provenance-and-gates.md): `qualified` is a benchmark
ledger status, while the experiment registry additionally uses `accepted`,
`rejected`, `failed`, `diagnostic`, `unqualified`, and `planned`.

## Safety and memory interpretation

Wrap sustained target commands with the thermal execution guard and retain
frequency, cooling, thermal, swap, minimum-memory, and kernel-event evidence.
An 85 C guard is an emergency stop, not proof that throttling did not occur.

The A733 LPDDR5-4800 / 19.2 GB/s figure is a theoretical SoC ceiling only.
PMU `mem_access`, `bus_access`, and cache-refill events are event counts, not
DDR bytes. Do not multiply them by a cache-line size and publish a bandwidth
claim without a source-verified event definition and a separate method.

## Tool entry points

- `tooling/profile_command.py` creates an immutable run bundle.
- `tooling/thermal_exec_guard.py` enforces the target thermal boundary.
- `tooling/summarize_target_run.py` validates and summarizes raw telemetry.
- `tooling/record_model_result.py` appends validated model rows.
- `tooling/check_repository_index.py` validates this canonical layer and its
  experiment provenance.

See the [target evidence](../evidence/vip9000-phase-profile-2026-08-10.md) for
an example of explicit H2D, run, device, D2H, repeatability, and quality
boundaries.
