# AGENTS.md

## 1. Project purpose

This repository is an evidence-driven research lab for accelerating local LLM inference on an Allwinner A733 system by combining its Arm CPU cores with the Vivante VIP9000-class NPU.

The primary engineering objective is to determine whether a useful, legally distributable, reproducible `ggml`/`llama.cpp` backend or companion execution path can be built for this hardware.

The project is not committed to any specific implementation until measurements justify it.

## 2. Target platform

Initial target:

- Board: Orange Pi Zero 3W
- SoC: Allwinner A733
- CPU: heterogeneous Arm cores
- NPU: Vivante VIP9000-class accelerator
- Memory: target unit reported as 12 GB LPDDR5; must be verified on-device
- OS/BSP: unknown until captured from the actual board
- NPU software: board/vendor SDK, likely involving VIPLite, VeriSilicon OpenVX, TIM-VX, Acuity tooling, or a subset thereof; exact stack must be verified

Do not copy specifications from a similar A733 board and present them as facts about the target board.

## 3. Non-negotiable research rules

### 3.1 Separate evidence from hypotheses

Every technical statement must be classed mentally as one of:

- **Verified on target** — reproduced on the actual Orange Pi device.
- **Verified upstream** — confirmed in an official upstream repository or vendor document.
- **Observed elsewhere** — reproduced by a third party on similar hardware.
- **Hypothesis** — plausible but not yet tested.
- **Unknown** — information is missing or contradictory.

Documents should state the class when ambiguity matters.

### 3.2 Prefer primary sources

Source priority:

1. Files and version output from the target board.
2. License/EULA and documentation shipped with the exact SDK.
3. Official Allwinner, Orange Pi, VeriSilicon, Khronos, `ggml`, and `llama.cpp` materials.
4. Source code of relevant open-source projects.
5. Reproducible third-party reports.
6. Forum posts and social media only as leads.

### 3.3 Never invent hardware or SDK behavior

Do not infer supported operators, precisions, tensor shapes, dynamic-shape behavior, memory coherence, or concurrency from the marketing TOPS number.

Do not assume VIP9000 variants are interchangeable. Record the exact product/device identifier.

### 3.4 Measure end to end

An NPU kernel being faster than a CPU kernel is not sufficient. Always account for:

- graph conversion and compilation
- graph loading and cache warm-up
- tensor packing and layout conversion
- quantization and dequantization
- host↔accelerator copies
- synchronization
- fallback transitions
- scheduler overhead
- memory use
- thermal throttling
- output error

### 3.5 Protect proprietary assets

Never commit any of the following before the license is reviewed:

- SDK archives
- vendor libraries
- NPU firmware
- kernel modules
- confidential documentation
- model weights
- generated network binaries
- files extracted from a vendor image

Prefer checksums, version strings, acquisition steps, expected local paths, and scripts that operate on user-supplied assets.

## 4. Development strategy

Evaluate these approaches in order, unless evidence strongly changes the order.

### Strategy A — Companion runtime prototype

Build a standalone NPU microbenchmark and graph runner outside `llama.cpp`.

Purpose:

- validate SDK usability
- enumerate operator and datatype support
- understand graph compilation and caching
- measure transfer and launch overhead
- establish numerical behavior

This is the lowest-risk first implementation.

### Strategy B — Operator-level `ggml` backend

Implement a minimal dynamically loaded `ggml` backend that accepts only supported tensors/operations and falls back to CPU for the rest.

Purpose:

- validate backend registration and scheduling
- exercise buffer allocation and synchronization
- offload one operation without modifying model semantics

Risk: per-operation launch and transfer overhead may erase all benefit.

### Strategy C — Subgraph compilation backend

Identify stable transformer subgraphs, translate them to the vendor graph API or ONNX-derived representation, compile once, cache the executable, and dispatch whole subgraphs.

Purpose:

- amortize dispatch cost
- reduce layout conversions
- increase NPU utilization

Risk: dynamic sequence lengths, KV-cache mutation, unsupported operations, and graph-cache explosion.

### Strategy D — Model-specific hybrid runtime

Create a model-specific execution path for a small reference architecture while retaining `llama.cpp` for tokenization, sampling, file formats, and unsupported computation.

Purpose:

- prove feasibility before designing a general backend

Risk: technical debt and limited generality.

### Strategy E — Full general backend

Only begin after prior milestones show measurable value and stable SDK behavior.

## 5. Recommended phase order

### Phase 0 — Legal and asset inventory

Exit criteria:

- exact SDK and runtime licenses are archived privately and summarized
- redistribution boundaries are documented
- required proprietary dependencies are identified
- repository license decision is recorded

### Phase 1 — Target fingerprint

Collect and commit sanitized output from:

```bash
uname -a
cat /etc/os-release
lscpu
cat /proc/cpuinfo
free -h
ls -l /dev
lsmod
find /sys -maxdepth 4 -iname '*vip*' -o -iname '*npu*' -o -iname '*galcore*'
ldconfig -p
```

Also record package versions, kernel config, device tree compatible strings, NPU logs, thermal zones, cpufreq policies, and memory type where verifiable.

Exit criteria: another engineer can identify the exact target software and hardware environment.

### Phase 2 — CPU baseline

Build upstream `llama.cpp` without NPU modifications.

Record:

- commit SHA
- compiler and flags
- Arm features used
- model and exact hash
- prompt length and output length
- `llama-bench` results
- TTFT, prompt tokens/s, decode tokens/s
- peak RSS
- temperature and clocks
- power, when measurable

Exit criteria: stable repeated baseline with variance reported.

### Phase 3 — NPU microbenchmarks

Start with fixed-shape graphs and persistent runtime objects.

Minimum tests:

- runtime initialization
- graph compile/load
- FP32/FP16/INT8 matrix multiplication where supported
- elementwise add/multiply
- activation functions
- normalization candidates
- transpose/reshape costs
- repeated warm invocation
- host↔NPU copy bandwidth

Exit criteria: a capability and cost table based on target measurements.

### Phase 4 — Minimal backend

Implement:

- backend registration
- device enumeration
- buffer type
- supported-op predicate
- graph dispatch
- synchronization
- controlled CPU fallback
- debug tracing

Exit criteria: a `ggml` test graph produces numerically valid output through the NPU path.

### Phase 5 — Transformer experiments

Test independently:

- token embedding lookup or post-embedding graph
- Q/K/V projections
- output projection
- MLP projections and activation
- RMSNorm
- attention subgraphs
- full transformer block
- prefill vs decode

Exit criteria: at least one useful partition improves a declared objective.

## 6. Correctness requirements

Every offloaded path needs a CPU reference.

Record as applicable:

- max absolute error
- mean absolute error
- relative error
- cosine similarity
- top-k agreement
- logits divergence
- generated-token agreement under deterministic settings
- perplexity or task-level quality for larger changes

A speedup with unexplained severe quality loss is a failed result.

## 7. Benchmark protocol

- Run cold and warm cases separately.
- Use at least five measured repetitions after warm-up when practical.
- Report median, minimum, maximum, and dispersion.
- Pin or record CPU governor and core affinity.
- Record thermals before and after.
- Do not compare runs with different model files, context lengths, batch sizes, or sampling settings without clearly labeling the difference.
- Keep raw machine-readable results under `benchmarks/results/` when they contain no secrets.

## 8. Code rules

- Prefer small, reversible patches.
- Keep vendor integration behind a narrow abstraction.
- Avoid invasive forks until a standalone prototype validates the design.
- Preserve CPU-only builds.
- Treat unsupported operations as a normal fallback condition, not an assertion failure.
- Add verbose tracing controlled by environment variables or build flags.
- Document ownership and lifetime of every accelerator buffer.
- Never compile ONNX or vendor graphs during token generation unless deliberately measuring a cold-start experiment.

## 9. Documentation rules

Each research document should contain:

- question
- current answer
- evidence
- confidence
- unresolved items
- next experiment
- source links or repository paths
- date and relevant version identifiers

Do not silently replace contradictory evidence. Record the contradiction.

## 10. Initial success criteria

The first meaningful success is not “the NPU is used.” It is one of:

- lower time-to-first-token
- higher prompt-processing throughput
- higher decode throughput
- lower energy per token
- lower CPU utilization at comparable latency
- ability to run a useful model/configuration otherwise impractical

The result must be reproducible on the target device and compared with an optimized CPU baseline.

## 11. Immediate next actions

1. Import only legally shareable board and SDK metadata.
2. Capture the target fingerprint.
3. Obtain the exact SDK license and runtime version.
4. Reproduce a vendor NPU sample.
5. Build upstream `llama.cpp` and record CPU baselines.
6. Implement a persistent NPU matrix-multiplication microbenchmark.
7. Inspect `ggml` backend APIs at a pinned upstream commit.
