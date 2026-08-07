# Profiling-First Research Foundation Design

**Date:** 2026-08-07

**Status:** Approved

## Purpose

Make `msazanov/vivante-vip9000-llm-lab` the canonical, clean-sheet record for repeatable research into high-throughput LLM inference on the Allwinner A733 and Vivante VIP9000. The immediate emphasis is repeated execution of Bonsai-family models, aggressive profiling from the first run, and optimization for decode tokens per second without an unexplained quality regression.

`orange-RAG` is an evidence source only. Its code and architectural assumptions are not imported. Prior artifacts may be referenced only through sanitized metadata, hashes, observed paths, and clearly labelled observations.

## Objective hierarchy

1. Maximize steady-state decode throughput in tokens per second.
2. Preserve output quality. A speed result is not eligible for promotion unless its quality result is recorded against the pinned CPU reference.
3. Improve prompt-processing throughput and time to first token where this does not reduce the primary objective.
4. Minimize launch, synchronization, tensor-layout, quantization, copy, cache-maintenance, and scheduler overhead.
5. Record memory footprint, CPU availability, thermal stability, and energy when measurable.

Decode and prefill are separate workloads. Results must never combine them into one unlabeled throughput number.

## Evidence model

Every material statement uses one evidence class:

- **Verified on target** — reproduced on the actual board and linked to raw output.
- **Verified in supplied tooling** — observed in locally available SDK/tool files, with version and hash where possible.
- **Verified upstream** — supported by a primary upstream source at a pinned revision.
- **Prior local observation** — recovered from `orange-RAG`, backups, or old artifacts but not yet reproduced under this protocol.
- **Hypothesis** — a candidate explanation or optimization awaiting a test.
- **Unknown/contradictory** — missing or conflicting evidence is retained explicitly.

No result inherited from `orange-RAG` is promoted to **Verified on target** without a new reproducible run.

## Architecture

### Host compiler layer

Document AcuityLite, the embedded VeriSilicon SDK, TIM-VX/OpenVX libraries, NBG exporters, compiler targets, license markers, Docker image IDs, and hashes. Vendor binaries remain outside Git. The repository stores only original scripts, sanitized manifests, checksums, and acquisition/path instructions.

### Target runtime layer

Document VIPLite headers, `libNBGlinker`, `libVIPhal`, `/dev/vipcore`, the NPU devfreq domain, existing `vpm_run` and persistent-runner assets, and their hashes. Persistent runtime and resident buffers are the default measurement architecture because per-token process startup or graph loading is unacceptable for decode.

### Experiment and profiling layer

Every executable experiment is wrapped by a standard profiler that records the command lifecycle and samples process and board telemetry. Every model-level test produces a machine-readable result and one append-only Markdown table row in the corresponding model card, including failed and rejected tests.

## Profiling contract

Each run receives a unique `run_id` and an immutable raw-result directory. The
profiler requires `--run-id` and requires the output-directory basename to
match it. Its optional relative `--phase-file` defaults to `phases.jsonl` and
is passed to the direct child as `VIP9000_PHASE_FILE`. At minimum a run records:

- repository, upstream runtime, model, tokenizer, compiler, SDK, driver, kernel, and command identifiers;
- exact model filename, byte size, SHA-256, quantization, group size, context, batch, micro-batch, thread count, affinity, and backend partition;
- process start/end UTC timestamps, monotonic duration, exit status, stdout, and stderr;
- launch failures, profiler errors, signal status mapping (`128 + signal`), and
  process-group cleanup when the collector fails;
- sampled process RSS/high-water RSS, thread count, user/system CPU ticks, system memory availability, CPU frequencies, NPU frequency, and thermal zones;
- cold initialization, graph compile/load, buffer allocation/import, first invocation, warm invocation, input preparation/copy/flush, output invalidate/copy, synchronization, and teardown when the tested runtime exposes them;
- raw `phases.jsonl` events from the direct child when phase instrumentation is available;
- prompt tokens per second, decode tokens per second, TTFT, total wall time, peak RSS, and dispersion across repetitions;
- warm-up count, measured repetition count, median, p10, p90, minimum, maximum, and coefficient of variation;
- for a qualified result, a `software` object with exactly `repository_commit`, `runtime`, `runtime_commit`, `compiler`, `sdk`, `driver`, `kernel`, and `command`; a `workload` object with exactly `tokenizer`, `prompt_suite`, `prompt_tokens`, `generated_tokens`, `seed`, `deterministic`, and `warmup_iterations`; and `performance.statistics` nested under `performance`, with `prompt_tps`, `decode_tps`, and `ttft_ms` each containing `median`, `p10`, `p90`, `min`, `max`, `cv`, and `samples`. The `model` and `configuration` objects remain separate.
- quality method, reference run, threshold, observed value, and pass/fail decision.

The profiler must use only the Python standard library and readable Linux `/proc` and `/sys` interfaces. Missing sensors are recorded as missing; they never abort a model run.

Launch failures still retain metadata and raw files. A signal-terminated child
maps to `128 + signal`; if collection fails, the profiler records the error and
cleans up the child process group before writing metadata.

## Quality gates

Every optimization is compared with an exact pinned CPU reference using deterministic settings. The smallest applicable guardrail is required:

- kernel/operator: max/mean absolute error, relative error, cosine similarity, and integer exact-match rate as applicable;
- logits: max/mean logit error, cosine similarity, top-k agreement, and KL divergence where practical;
- generation: deterministic token agreement plus a fixed prompt suite;
- quantization or model-format change: perplexity or a pinned task-level evaluation in addition to deterministic smoke tests.

Thresholds are declared before interpreting a run. A faster run with missing quality data is `unqualified`; a threshold failure is `rejected`, not silently deleted.

The recorder requires a qualified result to have non-null headline performance
metrics and positive repetitions, and to reference an existing compatible
qualified CPU result or be a self-referencing CPU baseline. Raw-result paths
must be safe JSON paths under `benchmarks/results/<run_id>/`. Its
`--repo-root` option defaults to `.` and qualified recording verifies the raw
JSON plus `metadata.json`, `stdout.log`, `stderr.log`, and `telemetry.jsonl`
under the resolved run directory. Metadata must match the result `run_id`,
report `exit_code: 0`, and contain null `launch_error` and `profiler_error`.
The safe relative path in `metadata.files.phases` must point to an existing
phase file within that run directory; custom and nested phase paths are valid.
Nonqualified, rejected, and failed records may retain staged or missing raw
files.

## Model records

`benchmarks/models/ternary-bonsai-27b.md` and `benchmarks/models/bonsai-27b.md` are long-lived model cards. Each contains immutable model identity information, the reference configuration, quality thresholds, and an append-only test table. One table row represents one test configuration and links to its raw result.

The canonical machine-readable ledger is `benchmarks/results/model-runs.jsonl`. Markdown is a rendered human index, not the source of truth. Duplicate `run_id` values are rejected.

Recording uses POSIX `flock`, fsynced sibling temporary files, and rollback or
recovery after a partial replacement. This is a lock/recovery protocol, not a
cross-filesystem atomic transaction.

Target and host inventory scripts publish temporary outputs only after success,
propagate mandatory failures, and keep optional sensor gaps explicit. The
AcuityLite inspection uses no network, read-only root, dropped capabilities,
`no-new-privileges`, PID/memory limits, and a temporary `/tmp`; license text is
hashed and sanitized rather than copied verbatim. Proprietary files remain
external.

## Initial repository changes

- Add this design and an implementation plan.
- Add hardware, toolchain, prior-evidence, profiling, and benchmark documentation.
- Add the two Bonsai model cards and the result ledger contract.
- Add a generic command profiler with automated tests.
- Add a result recorder that validates JSON, appends the JSONL ledger, and updates the selected model card, with automated tests.
- Add read-only inventory recipes for the host AcuityLite container and the target board.

## Non-goals for this change

- Running a new NPU inference benchmark.
- Committing SDK binaries, model weights, NBG files, runtime libraries, credentials, or private documentation.
- Claiming AcuityLite's bundled `GCNANOULTRA31_VIP2_PID0X15` license target is compatible with the A733 VIP9000.
- Selecting a final CPU/PPU/native-NN partition before measurements.

## Acceptance criteria

- A new engineer can find every currently discovered tool without re-searching the host or board.
- A model test cannot be treated as a valid speed result without a model hash, workload parameters, raw result link, and quality status.
- The profiling wrapper retains output and telemetry for both successful and failed commands.
- The result recorder rejects incomplete and duplicate results.
- All added executable behavior is covered by tests that were observed failing before implementation and passing afterwards.
- The change is published on an isolated GitHub branch as a draft pull request.
