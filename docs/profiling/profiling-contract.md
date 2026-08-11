# Profiling contract

This document is the execution contract for repeatable Bonsai-family inference
experiments on the A733/VIP9000 target. A run is evidence only when its raw
artifacts, exact identities, workload parameters, performance statistics, and
quality decision are retained together. Missing optional sensors are recorded
as missing; they do not turn a command failure into a successful result.

## Objective and KPI hierarchy

The primary KPI is **steady-state decode throughput in tokens per second**.
Optimization decisions are made against this value first. Prompt/prefill
throughput (prompt tokens per second) and time to first token (TTFT) are
secondary metrics and must be reported separately. A single unlabeled
"throughput" number is not a valid benchmark result.

The goal is to maximize decode tokens per second with no significant quality
loss, preferably with no quality loss. Every speed candidate requires a
pinned CPU reference run and a declared quality guardrail before the candidate
is interpreted.

## Run identity and immutable artifacts

Each run receives a unique `run_id`. The profiler requires `--run-id` and
requires the output-directory basename to equal that ID; this keeps the
directory, metadata, and ledger identity aligned. The run directory is created
using a non-existing path and is never overwritten. The result recorder rejects
duplicate IDs in the canonical JSONL ledger.

The profiler accepts an optional relative `--phase-file`, defaulting to
`phases.jsonl`. It creates that file in the run directory and passes its
absolute path to the direct child as `VIP9000_PHASE_FILE`; a runtime harness
can append phase events there without relying on stdout parsing.

At minimum, a run summary identifies:

- repository revision, upstream runtime revision, model and tokenizer IDs;
- exact model filename, byte size, SHA-256, format/quantization, and group size;
- compiler, SDK, driver, kernel, and command identifiers, with hashes when
  available;
- backend and CPU/NPU partition;
- context length, batch and micro-batch (`ubatch`) sizes, thread count,
  affinity, and deterministic seed/settings;
- target identity, OS/kernel, and relevant device/runtime paths.

Model and runtime hashes that are not yet known are written as unresolved in
planning documentation. They must be resolved before a result can be
`qualified`.

## Warmup, repetitions, and statistics

The harness declares warmup count and measured repetition count in the run
summary. Warmups are excluded from reported statistics. Decode and prefill are
measured as separate workloads with their own prompt/token counts and timing.

For every measured metric, retain the individual observations and report:

- median;
- p10 and p90;
- minimum and maximum;
- coefficient of variation (CV = standard deviation / mean).

Report units explicitly: tokens/second for prompt and decode, milliseconds for
TTFT, and MiB for resident memory. Repetitions that fail remain in the raw
record and are not silently dropped; the summary records the failed count and
the reason.

## Raw output and deep profiling

The standard profiler retains `metadata.json`, `stdout.log`, `stderr.log`, and
newline-delimited `telemetry.jsonl` in the immutable raw-result directory.
Both successful and failed child commands receive these artifacts. A result
summary links to that directory using a safe repository-relative `raw_result`
path.

From the first run, collect the following whenever the runtime or board makes
them observable:

- exact environment, model, and configuration hashes;
- process start/end UTC timestamps, monotonic duration, exit status, and launch
  error;
- phase timings for startup, VIP initialization, graph load, buffer
  allocation/import, first run, input preparation/copy/flush, warm run,
  synchronization, output invalidate/copy, and teardown;
- sampled RSS and high-water RSS, thread count, user/system CPU ticks, CPU load,
  available system memory, CPU and NPU frequencies, and thermal zones;
- copy counts and bytes, cache-maintenance operations, and power/energy when
  available;
- raw stdout, stderr, profiler metadata, telemetry, and any runtime trace.

The raw directory also retains the phase file. A launch failure still writes
metadata and all raw files that can be created, including the launch error.

### Словарь фаз для SoC с общей памятью

- `H2D` означает host-to-device visibility path. Для текущего VIPLite runner
  это `map + memcpy + unmap + cache flush`, а не PCIe transfer в отдельную
  память. Указанное GB/s является эффективной скоростью всего пути.
- `D2H` означает device-to-host visibility path:
  `cache invalidate + map + memcpy + unmap`. Для очень малого output fixed API
  overhead делает GB/s малоинформативным.
- `run` — wall-clock синхронного runtime call; `device` — время, сообщённое
  device profiler. Их разность интерпретируется только как host/driver/API
  overhead данного вызова.
- `first` — первая итерация после prepare; `steady` — последующие итерации
  resident graph/buffers. Setup и first нельзя смешивать со steady KPI.
- `golden` — независимо вычисленный reference output. Повторное byte-equality
  с первым device output является проверкой repeatability, но не correctness.

`profile_command.py` supplies lifecycle timing, direct-child process samples,
and board-wide readable `/proc`/`/sys` telemetry. It does not infer internal
VIP phases, copy byte counts, power, or runtime quality. The tested command or
an adjacent harness must emit explicit phase timestamps and counters for those
fields; unavailable values are recorded as unavailable.

## Profiler semantics

The profiler starts the requested command as one direct child in its own
process group, samples the child while it runs, and writes one telemetry JSON
object per interval. Its scope is explicitly:

> direct child process plus board-wide readable `/proc` and `/sys` telemetry

It is not a claim of full process-tree accounting. Child processes, daemonized
workers, kernel work, or vendor runtime threads that are not represented by the
direct child may be absent from process RSS/thread/CPU totals. Board telemetry
is likewise limited to readable interfaces exposed by the target.

The exact CLI is:

```bash
python3 tooling/profile_command.py \
  --output-dir benchmarks/results/<run_id> \
  --run-id <run_id> \
  --interval-ms 100 \
  --label <experiment-label> \
  -- <command> <arg>...
```

The profiler propagates a normal child exit status and records non-zero and
launch failures. A signal-terminated child is represented as `128 + signal`
(for example, SIGTERM becomes 143) and still retains its raw files. If the
collector itself fails, it records `profiler_error`, terminates the child
process group, and writes the metadata it can produce. An existing output
directory is never overwritten.

## Qualified-result machine contract

In addition to the run identity and performance fields above, a `qualified`
result carries these machine-readable objects:

- `software`, with exactly these keys: `repository_commit`, `runtime`,
  `runtime_commit`, `compiler`, `sdk`, `driver`, `kernel`, and `command`;
- `workload`, with exactly these keys: `tokenizer`, `prompt_suite`,
  `prompt_tokens`, `generated_tokens`, `seed`, `deterministic`, and
  `warmup_iterations`;
- `performance.statistics`, nested under `performance`, with `prompt_tps`,
  `decode_tps`, and `ttft_ms`; each metric contains exactly `median`, `p10`,
  `p90`, `min`, `max`, `cv`, and `samples`.

The `model` and `configuration` objects remain separate from `software`,
`workload`, and `performance`.

The field names and object shape are part of the result contract even when a
particular sensor or metric is unavailable. Failed and unqualified records
remain explicit rather than being made to look qualified.

## CPU reference and quality guardrails

The CPU reference is pinned per model and workload before candidate runs. Pin
at least the model file/hash, tokenizer, prompt suite, context, batch,
micro-batch, thread/affinity settings, seed, sampling settings, and software
revision. If the reference changes, assign a new reference `run_id`; do not
silently rebase older candidates.

The smallest applicable guardrail is declared before interpretation:

- kernel/operator: max and mean absolute error, relative error, cosine
  similarity, and integer exact-match rate as applicable;
- logits: max/mean logit error, cosine similarity, top-k agreement, and KL
  divergence where practical;
- generation: deterministic token agreement over a fixed prompt suite;
- quantization or model-format change: perplexity or a pinned task evaluation
  in addition to deterministic smoke tests.

A quality result names its method, pinned reference run, metric, threshold,
observed value, and pass/fail decision. A missing quality result cannot be
promoted as a speed result.

## Status and promotion gates

The recorder accepts exactly four statuses:

- `qualified`: performance and required quality evidence are present; the
  headline prompt/decode/TTFT/RSS metrics are non-null, repetitions are
  positive, and the candidate may be compared for promotion;
- `unqualified`: the run is retained, but required quality evidence or another
  required contract field is missing, so no speed claim is eligible;
- `rejected`: the run is retained and a declared quality threshold failed;
- `failed`: command, setup, or measurement execution failed; raw output and
  telemetry remain part of the record.

Promotion requires all of the following:

1. a unique run ID and complete model/configuration identity;
2. a model SHA-256 and raw-result link;
3. separate decode and prefill/TTFT measurements with repetitions and full
   dispersion statistics;
4. a pinned CPU reference for the same model, prompts, and deterministic
   settings;
5. a declared quality guardrail with a passing recorded result;
6. no unexplained material regression in the secondary metrics, memory,
   thermals, or failure rate that would make the decode result misleading.

The fastest decode number with missing quality data is `unqualified`, not
`qualified`. A threshold failure is `rejected`, not deleted. Failed tests and
rejected candidates remain visible in the ledger and model card so that the
research record cannot select only favorable runs.

For a qualified candidate, the recorder requires either an existing compatible
qualified CPU reference in the ledger or a self-referencing CPU baseline. The
self-reference is allowed only for a CPU backend and CPU partition; a candidate
cannot qualify by naming itself as an NPU reference.

The recorder requires `raw_result` to be a safe JSON path under
`benchmarks/results/<run_id>/`. It serializes updates with a POSIX `flock` lock,
constructs fsynced sibling temporary files, and attempts recovery if ledger
replacement succeeds but card replacement fails. This is a lock/recovery
protocol, not a cross-filesystem atomic transaction.

The recorder accepts `--repo-root PATH`, defaulting to `.`; it resolves and
checks qualified evidence beneath that root. A `qualified` record must point to
an existing raw-result JSON file and its run directory must also contain
`metadata.json`, `stdout.log`, `stderr.log`, and `telemetry.jsonl`. The parsed
metadata must have a matching `run_id`, `exit_code` equal to zero, and null
`launch_error` and `profiler_error` fields. Its `files.phases` value names the
phase artifact: that path must be a safe relative path within the run directory
and the named file must exist, so custom phase filenames and nested paths are
supported. `unqualified`, `rejected`, and `failed` records may retain staged or
missing raw files.

## Inventory publication and proprietary inputs

Target and host inventory scripts write temporary outputs and publish them only
after successful completion. Mandatory inventory sections propagate failure;
optional unreadable sensors are recorded as unavailable. The AcuityLite
inspection uses Docker with no network, a read-only root, dropped capabilities,
`no-new-privileges`, PID and memory limits, and a temporary writable `/tmp`.
License text is never published verbatim: the inspection records a hash and a
strictly sanitized marker. SDK archives, runtime libraries, model weights, and
other proprietary files remain external.

## Evidence and interpretation

Use explicit evidence labels in notes and summaries: verified on target,
verified in supplied tooling, verified upstream, prior local observation,
hypothesis, or unknown/contradictory. Prior artifacts from `orange-RAG` may
inform hypotheses and hashes, but are not verified on target under this
contract until reproduced with the current profiler and quality policy.
