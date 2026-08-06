# Backend architecture plan

## Goal

Design a hybrid execution path that allows `ggml`/`llama.cpp` graphs to use a Vivante VIP9000-class NPU where beneficial while preserving correct CPU execution for unsupported or unprofitable work.

The first backend is experimental and target-specific. Generalization comes after measurement.

## Candidate integration layers

### 1. Direct vendor runtime

Use VIPLite/OpenVX/vendor APIs directly from a `ggml` backend.

Advantages:

- minimum abstraction overhead;
- direct access to buffers, compilation, and synchronization;
- best chance of exploiting target-specific features.

Risks:

- proprietary API and ABI;
- poor portability;
- larger licensing surface;
- potentially limited documentation.

### 2. TIM-VX integration

Translate supported `ggml` graphs or subgraphs into TIM-VX operations.

Advantages:

- public C++ frontend;
- existing tensor/operation abstraction;
- possible portability across supported VeriSilicon platforms.

Risks:

- still requires vendor OpenVX runtime;
- operation coverage and dynamic-shape behavior may not match LLM workloads;
- additional graph-translation layer.

### 3. ONNX-to-vendor compilation pipeline

Export stable subgraphs to ONNX, transform unsupported nodes, compile to NPU binary, and load the result at runtime.

Advantages:

- uses vendor-supported conversion flow;
- suitable for model-specific proof of concept.

Risks:

- not acceptable per token;
- dynamic dimensions and KV cache are difficult;
- generated artifact may be model-, shape-, and SDK-version-specific.

### 4. Companion process

Run NPU graphs in a separate persistent process and communicate through shared memory or IPC.

Advantages:

- isolates proprietary runtime crashes and ABI conflicts;
- enables rapid prototyping without modifying `llama.cpp` deeply.

Risks:

- IPC and copy overhead;
- more complex deployment;
- not a final architecture unless zero-copy is available.

## Recommended progression

1. Standalone persistent NPU runner.
2. `ggml` test integration for one fixed-shape graph.
3. Dynamic backend plugin loaded at runtime.
4. Cost-aware subgraph offload.
5. Model-specific end-to-end prototype.
6. General backend only after repeatable wins.

## Minimal `ggml` backend responsibilities

The experimental backend should eventually provide:

- device discovery and descriptive metadata;
- backend initialization and teardown;
- accelerator buffer allocation or imported host buffers;
- tensor transfer operations;
- operation support predicate;
- graph or subgraph execution;
- asynchronous event/synchronization support if available;
- error translation and diagnostic tracing;
- CPU fallback without corrupting graph state;
- compiled-graph cache keyed by operation graph, shapes, types, layout, runtime version, and device identifier.

## Critical design distinction: prefill vs decode

### Prefill

Prompt processing has larger matrix dimensions and more parallelism. It is the most likely first target for graph accelerators.

Potential candidates:

- Q/K/V projections;
- output projection;
- MLP up/gate/down projections;
- whole transformer blocks where supported;
- embedding or preprocessing graphs.

### Decode

Token-by-token generation often becomes matrix-vector dominated with small batch dimensions. Accelerator launch, packing, and transfer costs can dominate.

Decode may remain faster on optimized CPU unless:

- the NPU supports efficient small-M matmul/GEMV;
- weights remain resident in accelerator-accessible memory;
- graph launch overhead is very small;
- several layers or operations are fused into a single invocation.

Benchmarks must report prefill and decode separately.

## Memory model questions

The backend design depends on answers to:

- Can NPU and CPU share physically contiguous or IOMMU-mapped memory?
- Can user-space buffers be imported without a copy?
- Are cache flush/invalidate operations explicit?
- Can model weights remain resident across invocations?
- Is accelerator memory separate or carved out of system RAM?
- What alignment and stride restrictions apply?
- Are tensor layouts fixed or selectable?
- How many compiled networks can remain loaded?
- Can multiple graphs execute concurrently?

## Graph partitioning policy

A node/subgraph is eligible only when all conditions hold:

1. Operation and tensor types are supported.
2. Shapes and layout satisfy runtime restrictions.
3. Required inputs can be made available without excessive conversion.
4. Expected compute savings exceed transfer, launch, and synchronization costs.
5. A validated CPU reference exists.
6. The compiled artifact is cached or compilation cost is intentionally excluded from warm execution.

Initial policy should be conservative and opt-in.

## Compiled graph cache

Suggested cache key fields:

```text
backend version
vendor runtime version
device/product ID
operation graph hash
tensor dimensions
tensor strides
data types
quantization parameters
layout
compile options
model or weight hash where embedded
```

The cache must be invalidated on mismatched runtime/device metadata.

## Weight handling options

1. **Host weights copied per call** — simplest, likely too slow.
2. **Persistent NPU buffers** — preferred if API supports reusable tensors.
3. **Weights embedded in compiled graph** — useful for model-specific subgraphs; increases artifact size and invalidation complexity.
4. **Shared system-memory import** — ideal if zero-copy and cache coherency are available.
5. **Repacked weight cache** — store NPU-specific layouts derived from GGUF tensors.

Each option needs memory and latency measurements.

## Quantization

Do not assume GGUF quantization types map to NPU-native types.

Research tasks:

- enumerate NPU-supported integer and floating-point formats;
- identify per-tensor vs per-channel scales and zero points;
- test symmetric/asymmetric quantization;
- determine whether weights can be consumed directly or require dequantization/repacking;
- compare NPU INT8/FP16 against CPU GGUF Q4/Q5 paths end to end;
- investigate ternary/binary execution only if the runtime exposes suitable primitives or custom-kernel capability.

## Diagnostics

Suggested environment variables:

```text
GGML_VIVANTE_DEBUG=1
GGML_VIVANTE_TRACE_GRAPH=1
GGML_VIVANTE_DISABLE=1
GGML_VIVANTE_FORCE_OP=<name>
GGML_VIVANTE_CACHE_DIR=<path>
GGML_VIVANTE_VERIFY=1
```

Every offloaded subgraph should be traceable with shape, dtype, cache hit/miss, transfer bytes, execution time, and fallback reason.

## Proof-of-concept acceptance criteria

A minimal backend milestone is complete when:

- it is optional at build/runtime;
- CPU-only behavior remains intact;
- a known `ggml` test graph runs through the NPU;
- output error is quantified;
- warm execution is measured against optimized CPU;
- failure or unsupported operations fall back safely;
- the exact SDK/runtime/device versions are recorded.
