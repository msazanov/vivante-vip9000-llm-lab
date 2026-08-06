# Vivante VIP9000 LLM Lab

Research and engineering workspace for using the Vivante VIP9000 NPU together with the Arm CPU cores of the Allwinner A733 SoC for local large-language-model inference.

The primary target device is an Orange Pi Zero 3W with an Allwinner A733 and 12 GB of system memory. The intended long-term result is a maintainable `ggml`/`llama.cpp` backend or companion execution path that can offload useful transformer workloads to the NPU while retaining reliable CPU fallback.

> Status: **research bootstrap**. No performance, compatibility, or licensing claim should be treated as final until it is reproduced on the target board and recorded in this repository.

## Mission

1. Inventory the exact A733/VIP9000 hardware, drivers, runtime libraries, SDK, and legal terms available on the target board.
2. Establish reproducible CPU-only `llama.cpp` baselines.
3. Characterize the NPU: supported data types, operators, tensor layouts, shape restrictions, compilation flow, memory-transfer cost, and concurrency behavior.
4. Prototype a `ggml` accelerator backend with transparent CPU fallback.
5. Determine the most effective partition between CPU and NPU for prompt processing and token decoding.
6. Publish reproducible benchmarks, correctness checks, patches, and integration notes.

## Preferred technical direction

The working hypothesis is a hybrid backend:

- `ggml` remains responsible for model loading, graph construction, scheduling, tokenization, sampling, and CPU fallback.
- TIM-VX, VIPLite, or the board-vendor Vivante runtime compiles supported operations or subgraphs for VIP9000.
- Compiled graphs are cached and reused; model execution must not invoke an ONNX conversion pipeline per token.
- Unsupported or inefficient nodes remain on the optimized Arm CPU backend.
- Data movement is measured explicitly. Offload is accepted only when end-to-end latency, throughput, energy, or CPU availability improves.

This is a hypothesis, not a commitment. Operator-level offload, subgraph offload, and model-specific execution will all be evaluated.

## Repository map

```text
AGENTS.md                         Project rules and research protocol
benchmarks/README.md              Benchmark definitions and result format
docs/architecture/backend-plan.md Backend design and phased implementation
docs/hardware/a733.md             A733 hardware facts and validation checklist
docs/hardware/orange-pi-zero-3w.md Target-board inventory
docs/legal/licensing.md           Licensing and redistribution matrix
docs/npu/tim-vx.md                TIM-VX/VIPLite integration notes
docs/research/related-work.md      Relevant projects and prior art
experiments/README.md              Experiment template and reproducibility rules
research/open-questions.md         Unresolved technical and legal questions
references/sources.md              Curated primary and secondary sources
```

## Initial milestones

### M0 — Evidence and legal inventory

- Obtain the exact board SDK package, release version, license/EULA, headers, libraries, compiler tools, and sample projects.
- Identify which materials may be committed, linked, redistributed, or only referenced.
- Record the exact kernel, BSP, NPU driver, firmware, runtime, and device identifier.

### M1 — Reproducible baselines

- Build upstream `llama.cpp` on the target board.
- Benchmark representative small GGUF models on CPU.
- Capture CPU topology, clocks, thermals, memory bandwidth, power mode, and build flags.

### M2 — Minimal NPU execution

- Compile and execute a tiny known-correct graph on VIP9000.
- Measure cold compilation, warm execution, host↔NPU transfer, and numerical error.
- Test at least matrix multiplication, elementwise operations, normalization candidates, and supported quantized types.

### M3 — `ggml` proof of concept

- Register an accelerator device through the `ggml` backend API.
- Implement buffer management, capability checks, graph execution, synchronization, and CPU fallback.
- Offload one useful operation or repeatable subgraph with correctness tests.

### M4 — Transformer path

- Evaluate MLP, projection, embedding, attention, and full-block partitioning separately.
- Compare prefill and decode behavior; they have different shapes and bottlenecks.
- Add compiled-graph caching and cost-aware partitioning.

### M5 — Real model benchmark

- Run a selected GGUF or model-specific graph end to end.
- Report prompt-processing throughput, decode throughput, time-to-first-token, memory, power, CPU load, NPU utilization, and output-quality checks.

## Safety and licensing

Do not commit private SDK archives, vendor runtime binaries, firmware, model weights, generated NPU binaries, or documentation unless their license explicitly permits repository storage and redistribution. Keep hashes, version strings, acquisition instructions, and local path conventions instead.

See [`docs/legal/licensing.md`](docs/legal/licensing.md).

## Current evidence snapshot

Confirmed from primary upstream sources:

- Allwinner describes A733 as an SoC with 2× Cortex-A76, 6× Cortex-A55, a PowerVR BXM GPU, a 3-TOPS NPU, and LPDDR4/LPDDR4X/LPDDR5 support.
- TIM-VX is a permissively licensed C++ integration layer for VeriSilicon ML accelerators, but it still requires a VeriSilicon OpenVX SDK; platform-specific SDKs are obtained from the relevant SoC vendor.
- `llama.cpp` exposes a backend device, buffer, scheduler, capability-check, graph-compute, and dynamic-backend-loading API suitable for an experimental accelerator backend.

Board-specific behavior and SDK licensing remain to be verified from the actual Orange Pi Zero 3W package.
