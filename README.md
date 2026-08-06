# Vivante VIP9000 LLM Lab

Research and engineering workspace for using the Vivante VIP9000 NPU together with the Arm CPU cores of the Allwinner A733 SoC for local large-language-model inference.

The primary target device is an Orange Pi Zero 3W with an Allwinner A733 and 12 GB of system memory. The intended long-term result is a maintainable `ggml`/`llama.cpp` backend or companion execution path that can offload useful transformer workloads to the NPU while retaining reliable CPU fallback.

> Status: **NPU capability research**. No performance, compatibility, or licensing claim should be treated as final until it is reproduced on the target board and recorded in this repository.

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

This is a hypothesis, not a commitment. Operator-level offload, subgraph offload, precompiled NBG dispatch, and model-specific execution will all be evaluated.

## Repository map

```text
AGENTS.md                                      Project rules and research protocol
benchmarks/README.md                           Benchmark definitions and result format
docs/architecture/backend-plan.md              Backend design and phased implementation
docs/hardware/a733.md                          A733 hardware facts and validation checklist
docs/hardware/orange-pi-zero-3w.md             Target-board inventory
docs/legal/licensing.md                        Licensing and redistribution matrix
docs/npu/vip9000-stack-and-capabilities.md     SDK/runtime layers and preliminary capability map
docs/research/llm-operation-compatibility.md   Transformer/ggml operation mapping
docs/research/related-work.md                  Relevant projects and prior art
experiments/E001-vip9000-capability-probe/     First target-side SDK/operator experiment
experiments/README.md                           Experiment template and reproducibility rules
research/open-questions.md                      Unresolved technical and legal questions
references/sources.md                           Curated primary and secondary sources
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

Confirmed in current upstream source:

- TIM-VX is a permissively licensed C++ graph integration layer, but a platform-specific VeriSilicon OpenVX SDK still comes from the SoC vendor.
- Its public tensor vocabulary includes FP16/FP32, INT8 and INT4 types, per-tensor and per-channel quantization, constant/variable/transient tensors, host handles, DMA-buffer descriptors, graph compilation and binary-graph export.
- The operator catalogue includes MatMul/Dense, Softmax, LayerNormalization, Gather, GELU/Swish, reductions and the other common building blocks of a transformer.
- Current internal source includes dedicated RMSNorm and RoPE operations, but their public accessibility and support in the exact Allwinner SDK remain unverified.
- `llama.cpp` exposes backend devices, buffers, capability predicates, graph execution, synchronization and dynamic backend loading suitable for an experimental accelerator plugin.
- Community A733 work has executed complete transformer-body graphs through VIPLite, demonstrating feasibility but not yet autoregressive LLM decode performance.

The next hard gate is [`E001`](experiments/E001-vip9000-capability-probe/README.md): target-verified SDK, operation, shape, type, memory and overhead measurements.
