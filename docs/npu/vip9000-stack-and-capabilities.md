# Vivante VIP9000 software stack and preliminary capability map

**Research date:** 2026-08-06  
**Status:** preliminary; upstream-source analysis, not yet verified on the target Orange Pi board  
**Primary TIM-VX revision:** `702715d714277376d7ff466b7b72833ff57564c1`  
**Primary llama.cpp revision:** `9de0fcf2b3e587a43f293d9a2b6ec0a32991f768`

## 1. Research question

What can the A733 Vivante VIP9000-class NPU software stack actually execute, which APIs are available to us, and which parts of a transformer/LLM graph are plausible candidates for acceleration?

The answer must be separated into three layers:

1. **TIM-VX source capability** — operations and data types visible in the open upstream integration layer.
2. **Vendor SDK capability** — what the exact Allwinner/Orange Pi OpenVX/VIPLite libraries and compiler expose.
3. **Target-device capability** — what the exact VIP9000 product ID, driver, firmware, memory configuration, and runtime accept in practice.

Only layer 1 is substantially known today. Layers 2 and 3 require the actual SDK and board.

## 2. Stack decomposition

The observed stack is not one library. It is a toolchain and runtime pipeline.

```text
Model / graph source
  PyTorch, ONNX, TFLite, direct C++ graph construction
        |
        v
Conversion / graph authoring
  Acuity / Pegasus                  TIM-VX C++ API
        |                                |
        v                                v
OpenVX / Ovxlib graph representation and vendor compilation
        |
        +--> NBG / network binary graph (optional offline artifact)
        |
        v
VIPLite or VeriSilicon OpenVX runtime
        |
        v
VIPhal / unified user-space driver libraries
        |
        v
aw_nna_galcore kernel driver + firmware
        |
        v
VIP9000 hardware
```

### 2.1 TIM-VX

TIM-VX is the open C++ integration layer maintained by VeriSilicon. Its README describes it as a backend binding used by TFLite, TVM, ONNX Runtime, Android NN and other frameworks. It provides dynamic graph construction, shape/layout inference, custom layers, and more than 150 mapped operators.

Important boundary: TIM-VX is not the complete hardware SDK. It must be compiled and linked against a VeriSilicon OpenVX SDK. The project supplies an x86 simulation SDK, while target-specific SDKs must be obtained from the relevant SoC vendor.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/README.md>

### 2.2 Ovxlib / OpenVX

The TIM-VX programming guide describes Ovxlib as a C wrapper around the OpenVX driver, with TIM-VX acting as a C++ wrapper. A graph consists of tensors and operation nodes and is frozen/verified before repeated execution.

This graph lifecycle matters for LLMs: construction and compilation should happen outside the token loop. A compiled graph must be reused for many invocations.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/docs/Programming_Guide.md>

### 2.3 Acuity / Pegasus

Acuity/Pegasus is the observed vendor conversion and code-generation path. Community A733 experiments use it to:

1. import a fixed-shape ONNX graph;
2. apply graph rewrites for unsupported nodes;
3. select a target such as `VIP9000NANODI_PLUS_PID0X1000003B`;
4. export an NBG binary;
5. execute that binary through VIPLite on the board.

This is currently the best documented path for model-level experiments, but it is not automatically suitable for a general `ggml` backend because conversion is offline, shape-specific, tool-version-specific, and potentially proprietary.

Observed examples:
- <https://github.com/waz664/vip9000-embeddinggemma/blob/main/docs/export_commands.md>
- <https://github.com/northwindlight/a733-npu/blob/main/README.md>

### 2.4 VIPLite

VIPLite is a lower-level runtime for loading and executing an already compiled NBG network. An observed persistent runner uses the following lifecycle:

```c
vip_init();
vip_create_network(...);
vip_query_network(...);
vip_create_buffer(...);
vip_prepare_network(...);
vip_set_input(...);
vip_set_output(...);

// repeated:
vip_map_buffer(...);
vip_flush_buffer(... FLUSH);
vip_run_network(...);
vip_flush_buffer(... INVALIDATE);
vip_map_buffer(...);

vip_finish_network(...);
vip_destroy_buffer(...);
vip_destroy_network(...);
vip_destroy();
```

This proves that a network and its buffers can remain alive across requests. It also exposes explicit cache flush/invalidate operations, which are critical for a shared-memory SoC.

Observed source:
- <https://github.com/waz664/vip9000-embeddinggemma/blob/main/tools/persistent_viplite_runner.c>

The example links against `libNBGlinker` and `libVIPhal`, using headers from a board SDK `viplite-tina` directory:
- <https://github.com/waz664/vip9000-embeddinggemma/blob/main/tools/build_persistent_viplite_runner.sh>

### 2.5 Kernel driver

The Radxa Allwinner BSP contains an `AW_NNA_GALCORE` kernel driver configuration with an SPDX `GPL-2.0` marker. This establishes that at least the kernel-facing driver source in that BSP is published under GPL-2.0; it does **not** establish the license of the user-space SDK, firmware, compiler, OpenVX libraries, VIPLite libraries, or NBG artifacts.

Source:
- <https://github.com/radxa/allwinner-bsp/blob/cubie-aiot-v1.4.6/drivers/npu/aw_nna_galcore/Kconfig>

## 3. Public tensor and memory model

TIM-VX exposes the following tensor attributes:

- input;
- output;
- variable/state;
- constant;
- transient/intermediate.

This is encouraging for transformers:

- model weights can be constant tensors;
- activations can be transient;
- KV-cache or other recurrent state may potentially be represented as variable tensors;
- host-updated input/output tensors can be reused across inference calls.

The public API supports:

- creation of I/O tensors backed by a host pointer;
- creation using a DMA buffer file descriptor;
- pointer/handle swapping;
- map/unmap;
- cache flush and invalidation;
- graph compilation;
- compilation to a binary graph;
- repeated `Run()`.

These APIs make zero-copy or reduced-copy integration plausible, but the exact A733 implementation must be measured. An API being present does not prove that a particular memory path is efficient or even enabled by the board SDK.

Sources:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/tensor.h>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/graph.h>

## 4. Public data types and quantization

The current TIM-VX public type declarations include:

- `FLOAT32`, `FLOAT16`;
- signed and unsigned 8/16/32/64-bit integer variants;
- `INT4`, `UINT4`;
- `BOOL8`.

Public quantization modes include:

- asymmetric per-tensor;
- symmetric per-channel;
- asymmetric per-channel;
- dynamic fixed point.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/types.h>

### Important interpretation

This does **not** mean every operation on the A733 VIP9000 supports every declared type. Type declarations describe the frontend vocabulary. Hardware/runtime support is operator-, shape-, layout-, and SDK-version-specific.

### Ternary Bonsai implication

A ternary weight uses the value set `{-1, 0, +1}`. TIM-VX exposes no public ternary tensor type. Therefore the Bonsai weights cannot be assumed to map directly to a native NPU ternary format.

Initial possibilities:

1. unpack/repack ternary weights into INT4 or INT8 before NPU execution;
2. generate a custom kernel that consumes packed ternary data;
3. embed converted weights into an NBG graph;
4. use the NPU only for FP16/INT8 subgraphs and leave ternary kernels on CPU;
5. discover a vendor-private low-bit primitive unavailable in public TIM-VX.

The direct-memory and performance cost of options 1–3 must be measured before any architecture decision.

## 5. Operator coverage relevant to LLMs

The TIM-VX operator catalogue includes many operations needed to express a transformer:

- matrix multiplication and fully connected layers;
- Add, Subtract, Multiply, Divide;
- Reshape, Transpose, Slice, StridedSlice, Split, Concat;
- Gather, GatherND, embedding lookup;
- Softmax and LogSoftmax;
- LayerNormalization;
- reductions: Sum, Mean, Max, Min, Product;
- Rsqrt, Sqrt, Exp, Log, Pow;
- GELU, Swish, Sigmoid, Tanh;
- casting and data conversion;
- broadcast;
- NBG execution.

The catalogue itself warns that actual implementations may differ from reference operations in supported dimensions and parameters.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/ops/README.md>

## 6. LLM-specific internal kernels found upstream

A particularly important finding is that current TIM-VX internal source contains explicit LLM-oriented operations.

### 6.1 RMSNorm

An internal `RMSNORM` implementation accepts an activation tensor and a scale tensor, with configurable axis and epsilon. The type-validation table includes FP32 and FP16 paths as well as several quantized input/output combinations.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/ops/vsi_nn_op_rmsnorm.c>

Caution: no corresponding `include/tim/vx/ops/rmsnorm.h` public header was found at this revision. The kernel may be internal-only, used through conversion/codegen, or not exposed by the stable TIM-VX C++ API. We must test whether the Allwinner SDK compiler can emit it and whether direct graph construction can reach it.

### 6.2 RoPE

The internal source contains a dedicated `ROPE` operation with axis and interleaving controls. It supports FP32, FP16, BF16 and multiple 8/16-bit quantized combinations in the source-level type checker. Multiple implementations are present, including EVIS, OpenCL and VX kernel paths.

Sources:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/ops/vsi_nn_op_rope.c>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/kernel/evis/rope_evis.c>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/kernel/cl/rope_cl.c>

Again, public accessibility and A733 compatibility are not yet proven.

## 7. Custom operation capability

TIM-VX documents two extension mechanisms:

1. compose a new operation from built-in operations;
2. register a custom standard OpenCL 2.0 kernel.

The custom-kernel API accepts kernel source, tensor inputs/outputs, scalar parameters, build options, and global/local work sizes.

Source:
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/docs/customized_op.md>

### Why this matters

Potential uses include:

- packed ternary unpacking or ternary matmul experiments;
- fused RMSNorm + projection preprocessing;
- fused dequantization + matmul;
- model-specific RoPE variants;
- layout/repacking kernels.

### Critical uncertainty

The documentation calls this a standard OpenCL 2.0 custom operator, but the actual execution target may be a programmable shader/core path rather than the fixed-function NPU datapath. We must determine:

- whether custom OpenCL kernels execute on VIP9000, a companion programmable unit, or CPU fallback;
- which OpenCL language subset and data types are accepted;
- whether the A733 SDK includes the required compiler;
- whether a custom kernel can consume the same buffers as native NPU nodes without copies;
- whether custom kernels can be included in NBG artifacts.

Until tested, custom operations are an opportunity, not a guaranteed solution.

## 8. Graph shape and compilation implications

TIM-VX describes dynamic graph construction and shape inference, but vendor graph accelerators typically compile concrete tensor shapes. Community A733 examples use fixed input shapes and target-specific offline NBG artifacts.

This creates two different LLM workloads:

### Prefill

- sequence length can be bucketed;
- larger matrices create enough parallelism;
- fixed graph variants for sequence buckets may be viable;
- whole attention/MLP subgraphs may amortize launch overhead.

### Decode

- one token per invocation produces GEMV-like shapes;
- launch and synchronization overhead can dominate;
- KV-cache length changes every token;
- fixed-shape graph caching may require padding or many graph variants;
- CPU may remain better unless many layers are fused or the runtime supports efficient dynamic state.

Initial priority: **prefill and large projection/MLP subgraphs**, not token embedding lookup and not single-node decode offload.

## 9. Confirmed community evidence on A733-class hardware

These are not vendor guarantees, but they establish that the stack is usable in practice.

### 9.1 EmbeddingGemma transformer body

A community project reports an EmbeddingGemma transformer body compiled to a VIPLite NBG for A733/VIP9000, with input embeddings and attention bias as inputs and hidden states as output. It keeps tokenization, embedding lookup, pooling, dense tail, and normalization on CPU.

The project reports that correcting the attention mask improved cosine similarity substantially and that a persistent VIPLite runner keeps the network loaded.

Source:
- <https://github.com/waz664/vip9000-embeddinggemma>

This proves that transformer blocks are not categorically impossible on the hardware. It does not prove useful autoregressive LLM decode performance.

### 9.2 CNN and image models

Another community project documents ONNX → Acuity/Pegasus → quantized NBG → VIPLite inference for LeNet, ShuffleNetV2, YOLOv5s and EDSR on an A733 target identified as `VIP9000NANODI_PLUS_PID0X1000003B`.

Source:
- <https://github.com/northwindlight/a733-npu>

This provides useful toolchain and target-ID clues, but all measurements must be reproduced independently on our board.

## 10. Preliminary capability classification

### Green: clearly expressible upstream; test first

- fixed-shape FP16/FP32 MatMul;
- INT8 quantized MatMul/Dense where the target accepts it;
- Add/Multiply and broadcast;
- Reshape and simple views/layout transforms;
- Softmax;
- LayerNorm or composed normalization;
- GELU/Swish;
- Gather/embedding lookup;
- fixed-shape complete MLP subgraph;
- fixed-shape Q/K/V projection subgraph;
- persistent NBG load/run;
- constant weights embedded in a graph;
- host/DMA-backed I/O buffers.

### Yellow: source support exists, target/runtime behavior uncertain

- RMSNorm public accessibility;
- native RoPE accessibility;
- dynamic sequence lengths;
- variable tensors for KV cache;
- zero-copy host pointer import;
- DMA-BUF interoperability;
- concurrent CPU/NPU execution;
- INT4 matmul rather than merely INT4 tensor declaration;
- per-channel quantized LLM projections;
- custom OpenCL kernels on the actual accelerator path;
- large transformer graph compilation and memory usage;
- repeated decode graph with changing KV length.

### Red: no evidence of direct native support

- native ternary tensor/matmul format;
- direct consumption of Bonsai packed ternary weights;
- direct consumption of arbitrary GGUF Q2/Q3/Q4/Q5/Q6 formats;
- arbitrary-stride `ggml` views without repacking;
- free-form dynamic graphs compiled inside the token loop;
- assuming all 150 TIM-VX operators work on this VIP9000 variant.

## 11. Most likely integration paths

### Path A — offline model-specific NBG

Best for proving that a complete fixed-shape transformer subgraph runs.

- export one Bonsai/Qwen-derived layer in FP16 or INT8;
- compile with Acuity/Pegasus;
- load through VIPLite;
- compare against a CPU reference;
- measure warm latency and memory.

This is the fastest feasibility proof, but not a general `llama.cpp` backend.

### Path B — direct TIM-VX graph construction

Best long-term candidate if the target SDK can compile graphs on-device or on a build host with a stable deployment interface.

- translate selected `ggml` subgraphs into TIM-VX;
- create constant weight tensors;
- compile once and cache;
- reuse input/output buffers;
- fall back to CPU for unsupported nodes.

### Path C — direct VIPLite NBG backend

Best if runtime deployment is stable but graph compilation is available only through proprietary offline tools.

- backend recognizes a supported, precompiled model/subgraph;
- dispatches an NBG network;
- uses a manifest to map GGUF tensors and graph inputs;
- does not pretend to be a universal operation translator.

### Path D — custom low-bit kernels

Only investigate after confirming where custom OpenCL executes and after establishing FP16/INT8 baselines. A ternary kernel is not useful if it runs on a weak programmable path or requires repacking every token.

## 12. Immediate experiments required

1. **SDK inventory** — exact Acuity, OpenVX, VIPLite, VIPhal, driver and firmware versions.
2. **Runtime hello-world** — persistent load/run of the smallest vendor NBG.
3. **Buffer test** — copy, map/unmap, cache operations, host pointer and DMA-BUF import.
4. **MatMul shape sweep** — FP32, FP16, INT8; prefill-like and decode-like dimensions.
5. **Operator probe** — compile/run the LLM-relevant operation set and record accepted shapes/types.
6. **RMSNorm probe** — direct graph, converter path, or composed implementation.
7. **RoPE probe** — determine whether the internal kernel is exposed by the exact SDK.
8. **MLP subgraph** — two/three projections + SiLU/Swish + multiply.
9. **Attention prefill subgraph** — Q/K/V, RoPE, score matmul, mask, softmax, value matmul, output projection.
10. **Ternary feasibility** — compare unpack-to-INT8, preconverted INT8 weights, and custom-kernel possibilities.

## 13. Current conclusion

The VIP9000 stack is materially more capable than a CNN-only accelerator API: upstream source contains the majority of transformer primitives, including internal RMSNorm and RoPE kernels, graph compilation, reusable buffers, low-bit integer declarations and custom operation mechanisms.

However, the research bottleneck is now very specific:

> Determine which of these capabilities survive through the exact Allwinner A733 SDK and VIP9000 product variant, with which shapes, data types, layouts, memory paths and runtime overheads.

A general `llama.cpp` backend should not be started before the operator/shape/runtime probe. The first implementation should be a persistent capability-test harness and a fixed-shape transformer-subgraph proof of concept.
