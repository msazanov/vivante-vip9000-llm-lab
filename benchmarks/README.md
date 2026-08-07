# Benchmark contract

This directory is the canonical record for Bonsai/VIP9000 model experiments.
The primary KPI is steady-state decode tok/s. Prompt/prefill tok/s and TTFT
are secondary metrics and are always reported in separate fields and table
columns. The objective is higher decode throughput without significant quality
loss, preferably none.

## Directory conventions

```text
benchmarks/
  README.md
  models/<model-card>.md
  results/model-runs.jsonl
  results/<run_id>/
    metadata.json
    stdout.log
    stderr.log
    telemetry.jsonl
    phases.jsonl
    result.json
    summary.json
  schema/model-run.example.json
  inventory/<inventory_id>/
```

`results/model-runs.jsonl` is the canonical machine-readable ledger. Model
cards are append-only human indexes rendered by
`tooling/record_model_result.py`; they are not edited by hand after a result
is recorded. A run directory is immutable once created. The example under
`schema/` is synthetic documentation, not a benchmark claim.

## Required run evidence

Every candidate records exact repository/runtime/model/tokenizer/compiler/SDK/
driver/kernel/command identities and hashes when available. Model identity
includes the exact local filename, byte size, SHA-256, format/quantization, and
group size. Configuration includes backend, CPU/NPU partition, context, batch,
micro-batch, threads, affinity, seed, and deterministic generation settings.

The profiler requires `--run-id <run_id>` and the output-directory basename
must match it. Its optional relative `--phase-file` defaults to `phases.jsonl`
and is passed to the direct child as `VIP9000_PHASE_FILE`. A launch failure
still retains metadata and raw files. A child terminated by signal is recorded
with exit status `128 + signal`; collector failures terminate the child process
group after recording the profiler error.

The first run is deep-profiled. Retain raw stdout/stderr/telemetry and phase
timings for startup, VIP init, graph load, buffer allocation/import, first run,
input preparation/copy/flush, warm run, sync, output invalidate/copy, and
teardown. Record warmup count and measured repetitions. For each metric retain
median, p10, p90, min, max, and CV. Sample RSS/HWM, threads, CPU ticks/load,
system memory, CPU/NPU frequencies, thermals, copies/bytes, and power when the
target exposes them.

Decode and prefill are separate workloads. A result may report both, but it
must never combine them into one unlabeled throughput number.

## CPU reference and quality policy

Before interpreting a speed candidate, pin an exact CPU reference for the same
model, tokenizer, prompt suite, context, batch, micro-batch, threads/affinity,
seed, sampling settings, and software revision. Declare the quality guardrail
before comparing the candidate. Use deterministic token agreement for
generation; use operator/logit error, cosine/top-k/KL, perplexity, or a pinned
task evaluation when those are the applicable measures.

The recorder status is one of `qualified`, `unqualified`, `rejected`, or
`failed`:

- `qualified` has non-null headline prompt/decode/TTFT/RSS metrics, positive
  repetitions, required software/workload/performance.statistics fields, and a passing
  quality result against an existing compatible qualified CPU reference (or a
  self-referencing CPU baseline);
- `unqualified` is retained but cannot support a speed claim because evidence
  is incomplete;
- `rejected` is retained because a declared quality threshold failed;
- `failed` is retained because execution or measurement failed.

Failed and rejected tests remain recorded. Missing optional board sensors are
also recorded as missing rather than treated as a reason to discard a run.

Qualified results carry a `software` object with exactly these keys:
`repository_commit`, `runtime`, `runtime_commit`, `compiler`, `sdk`, `driver`,
`kernel`, and `command`; and a `workload` object with exactly these keys:
`tokenizer`, `prompt_suite`, `prompt_tokens`, `generated_tokens`, `seed`,
`deterministic`, and `warmup_iterations`. The nested
`performance.statistics` object has `prompt_tps`, `decode_tps`, and `ttft_ms`,
each containing exactly `median`, `p10`, `p90`, `min`, `max`, `cv`, and
`samples`. The `model` and `configuration` objects remain separate. These
fields preserve reproducibility even when optional telemetry is unavailable.

`record_model_result.py` constrains `raw_result` to a safe JSON path under
`benchmarks/results/<run_id>/`. It serializes ledger/card updates with a POSIX
`flock`, fsynced sibling temporary files, and recovery after a partial
replacement. This is a lock/recovery protocol, not a cross-filesystem atomic
transaction. Pass `--repo-root PATH` (default `.`) when the repository root is
not the current directory. For `qualified`, the raw JSON and the same run
directory's `metadata.json`, `stdout.log`, `stderr.log`, and `telemetry.jsonl`
must exist. Metadata must match the result `run_id`, report `exit_code: 0`, and
have null `launch_error` and `profiler_error`; `metadata.files.phases` must
name an existing safe relative phase file within that run directory, including
when a custom or nested phase path was used. Other statuses may reference
staged or missing raw files.

## Model-card table interface

Each model card contains exactly one `MODEL_RESULTS_START` marker and one
`MODEL_RESULTS_END` marker. The recorder inserts one row immediately before the
end marker. Keep this header unchanged because
`record_model_result.py` renders rows for it:

| Date | Run | Experiment | Backend / partition | Quantization | ctx / batch / ubatch / threads | Prompt tok/s | Decode tok/s | TTFT ms | Peak RSS MiB | Quality | Status | Raw |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|

## Run IDs and raw links

Use a stable, descriptive run ID such as
`tb27b-cpu-reference-001` or `tb27b-vip9000-candidate-001`. Do not reuse an ID
for a changed model hash, runtime, prompt suite, or configuration. The
`raw_result` field is a repository-relative path such as
`benchmarks/results/tb27b-vip9000-candidate-001/summary.json`; it must be a
JSON file under that run's directory. Absolute paths, `..` components, and a
different run ID are invalid.

## Inventory safety

`target_inventory.sh` and `inspect_acuitylite.sh` write temporary output and
publish it only after successful completion. Mandatory inventory failures
propagate; optional target sensors are marked unavailable. The container
inspection uses no network, read-only root, dropped capabilities,
`no-new-privileges`, PID/memory limits, and a temporary `/tmp`; license text is
hashed and redacted rather than copied into the output.
