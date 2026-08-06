# Packed ternary Bonsai on A733/VIP9000

**Research date:** 2026-08-06  
**Status:** source-level feasibility assessment; target-board validation required  
**Target:** `VIP9000NANODI_PLUS_PID0X1000003B` / product ID `0x1000003b`  
**Primary question:** Can Bonsai weights remain in their packed ternary representation and be consumed by a custom kernel without permanently converting the model to INT4 or INT8?

## Executive answer

**Yes, a custom packed-ternary kernel is technically plausible on this A733 VIP9000 configuration.** The target includes a programmable shader/PPU core with EVIS support, the SDK contains OpenVX/OpenCL compiler components, and TIM-VX contains working examples of custom OpenCL and EVIS/VXC kernels.

However, three different claims must not be confused:

1. **Keep weights packed on disk and in system memory:** likely achievable.
2. **Consume packed weights directly without expanding the complete model:** likely achievable through a custom PPU/EVIS kernel.
3. **Make all eight NN matrix cores natively consume the packed ternary format:** no public evidence currently supports this. The target feature database exposes native NN 4-bit support at an early feature level, but explicitly reports no packed 4-bit coefficient mode and no group-quantization feature.

The most credible first result is therefore a **direct packed-ternary GEMV/GEMM prototype on the programmable PPU/EVIS path**, followed by a comparison against a **tile-unpack-to-native-NN** path. We should not assume that either will beat an optimized A733 CPU ternary kernel until measured.

## 1. Exact Bonsai format relevant to this project

Prism ML describes Ternary Bonsai weights as:

```text
w_i = scale_g * t_i

t_i in {-1, 0, +1}
```

with one FP16 scale per group. The original Ternary Bonsai release uses groups of 128; current mainline `llama.cpp` also includes a group-64 `Q2_0` variant.

The current mainline `llama.cpp` representation is structurally simple:

```c
#define QK2_0 64
typedef struct {
    ggml_half d;
    uint8_t qs[QK2_0 / 4];
} block_q2_0;
```

Four weights are packed into each byte. The current code mapping is:

```text
00 -> -1
01 ->  0
10 -> +1
11 -> +2 / reserved for a more general Q2_0 value
```

For the true Bonsai tensors only the first three codes are expected. The group-128 fork uses the same general design with a different scale-group size.

This is important: the format is already suited to direct consumption by a vector kernel. A kernel needs to load one packed byte, extract four 2-bit symbols, multiply the accumulated dot product by the group FP16 scale, and never materialize an FP16 weight matrix.

Primary references:

- <https://github.com/ggml-org/llama.cpp/blob/15586e2d7165570fb3aa7c26e0d442e289ef69de/ggml/src/ggml-common.h>
- <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>
- <https://github.com/ggml-org/llama.cpp/discussions/22019>
- <https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf>

## 2. Exact A733 VIP9000 execution resources

The Radxa/Allwinner BSP feature database has an entry for product ID `0x1000003b`. It reports:

| Resource | Target value | Interpretation |
|---|---:|---|
| VIP cores | 1 | one accelerator instance |
| NN cores | 8 | fixed-function/programmable NN datapaths used by compiled graphs |
| Active NN cores | 8 | all eight are enabled |
| INT8 NN cores | 8 | strong native INT8 path |
| INT16 NN cores | 8 | native INT16 path |
| Shader/PPU cores | 1 | programmable OpenCL/EVIS execution path |
| Shader threads | 256 | PPU thread capacity reported by the feature DB |
| EVIS | enabled | extended vector/data-path instructions are present |
| VIP SRAM | 512 KiB | on-chip working storage / bandwidth reduction |
| SRAM physical width | 128 bytes | wide on-chip access path |
| AXI bus width | 16 bytes | 128-bit external-memory interface |
| Minimum/DDR burst | 256 bytes | layouts should be tiled and burst-friendly |
| NN MAD per core | 64 | native NN arithmetic capability descriptor |
| Stream processor | enabled | additional NN-side processing capability exists |
| GEMM phase 1 | enabled | matrix multiplication path exists |
| FP16 ALU | enabled | FP16 arithmetic capability is exposed |
| FP32 I/O | enabled | FP32 graph boundaries are accepted |
| 4-bit phase 1 | enabled | some 4-bit NN behavior exists |
| 4-bit packed coefficient mode | disabled | no evidence of direct packed low-bit coefficient consumption |
| group quantization | disabled | modern per-group low-bit LLM quantization is not native in this configuration |
| high-performance decode | disabled | token-at-a-time LLM decode is not a dedicated strength |
| dynamic shapes | disabled | graph bucketing or fixed dimensions are expected |
| tensor DMA | disabled | data movement requires careful measurement |
| TP engine cores | 0 | no separate tensor-processing cores in this SKU |

Primary BSP sources:

- <https://github.com/radxa/allwinner-bsp/blob/2045a3ca2a01f088c0314dc924bda59d154e363e/drivers/npu/aw_nna_galcore/inc/gc_feature_database.h>
- <https://github.com/radxa/allwinner-bsp/blob/2045a3ca2a01f088c0314dc924bda59d154e363e/drivers/npu/aw_nna_vip/vip2/inc/vip_feature_database.h>

### Interpretation

This is a **heterogeneous accelerator inside the SoC**, not one uniform compute array:

```text
8 NN cores
  best chance for large, fixed-shape, supported GEMM / convolution-like work

1 programmable PPU/shader core + EVIS
  custom kernels, data conversion, irregular low-bit operations, RoPE, packing

512 KiB VIP SRAM
  tiles, short-lived expanded data, activation reuse

Arm CPU
  control flow, tokenizer, unsupported graph nodes, possibly decode GEMV
```

A custom OpenCL or EVIS kernel normally targets the single programmable PPU path. It does **not automatically turn into a new native instruction for all eight NN cores**.

## 3. Evidence that custom code can be compiled

### 3.1 TIM-VX public custom OpenCL operation

TIM-VX provides `CustomOpBase`. Its implementation registers `VSI_NN_KERNEL_TYPE_CL`, accepts OpenCL source and build options, creates an OpenVX node, binds tensors/scalars, and configures global/local work sizes.

The upstream `custom_gemm.h` sample implements a custom matrix multiplication kernel in OpenCL C. This is direct evidence that custom graph operations are supported by the software stack.

Sources:

- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/ops/custom_base.h>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/ops/custom_base.cc>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/samples/custom_op_test/custom_gemm.h>

### 3.2 OpenVX-to-OpenCL interop in the A733 SDK

The published A733 SDK mirror contains the Khronos OpenVX/OpenCL interop API, including:

```c
vxAddOpenCLAsSourceKernel(...)
vxAddOpenCLAsBinaryKernel(...)
```

Source:

- <https://github.com/ZIFENG278/ai-sdk/blob/fc90006d0f6569da2f6726c2d8395877686f5aca/unified-tina/inc/VX/vx_khr_opencl.h>

### 3.3 Compiler/runtime binaries present

The same SDK tree includes:

```text
libOpenVX.so
libCLC.so
libVSC.so
libNNVXCBinary.so
```

This strongly indicates that the target SDK includes the compiler/runtime pieces needed for OpenCL/VXC graph kernels, not only the minimal VIPLite NBG loader.

References:

- <https://github.com/ZIFENG278/ai-sdk/tree/fc90006d0f6569da2f6726c2d8395877686f5aca/unified-tina/lib/aarch64-none-linux-gnu>

### 3.4 EVIS/VXC source kernels are visible upstream

TIM-VX includes VXC source files using:

- `_viv_asm(...)`;
- `VXC_DP4x4` and `VXC_DP2x8`;
- vector image loads/stores;
- 512-bit programmable data-path descriptors (`gpu_dp_inst_t`).

The RoPE implementation is a concrete LLM-oriented example.

Sources:

- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/libnnext/ops/vx/rope_0.vx>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/kernel/evis/rope_evis.c>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/include/kernel/vsi_nn_gpu.h>

This is the most promising route for bit extraction and vector accumulation. It is also an internal, version-sensitive API rather than a stable portable public interface.

## 4. Candidate ternary execution architectures

## A. Direct packed ternary GEMV/GEMM on the PPU

```text
packed Q2_0 weights in memory
          +
FP16 or INT8 activations
          |
          v
custom OpenCL/VXC/EVIS kernel
  extract four 2-bit symbols per byte
  q=0: subtract activation
  q=1: skip
  q=2: add activation
  accumulate by group
  multiply by FP16 group scale
          |
          v
FP16/FP32 output activation
```

### Advantages

- no full-model expansion;
- preserves approximately 2-bit storage and memory traffic;
- exact Bonsai semantics are possible;
- natural first proof of concept;
- decoding is often bandwidth-bound, so compact reads may matter more than nominal TOPS.

### Risks

- only one programmable PPU core is exposed;
- custom kernel peak arithmetic may be far below the eight NN cores;
- image/tensor layouts and vector-load alignment may be awkward;
- FP16 group scaling adds work;
- small-GEMV scheduling and launch overhead may dominate;
- exact VXC compiler headers may be available only inside the vendor package.

### Current confidence

- **Feasibility:** medium-high.
- **Performance win:** unknown.
- **Ability to use all 3 TOPS:** low confidence; likely no.

## B. PPU tile unpack followed by native NN GEMM

```text
packed ternary weights
        |
        v
PPU/EVIS unpack one tile -> INT8 transient tile in VIP SRAM/system buffer
        |
        v
8-core NN GEMM consumes INT8 tile
```

This keeps the model stored in packed ternary form but expands only a tile at a time.

### Advantages

- gives the eight NN cores supported INT8 input;
- can reuse the 512 KiB VIP SRAM for tiles;
- model file and long-lived RAM remain compact;
- may overlap unpacking and NN execution if the runtime truly supports PPU/NN parallelism.

### Risks

- expanded tile is written and then read again;
- no tensor DMA feature is reported;
- cross-engine handoff may force DDR traffic or synchronization;
- compiled graph may not expose an easy custom-kernel-to-NN zero-copy path;
- repeated weight expansion during every layer/token may be more expensive than CPU ternary GEMV.

### Current confidence

- **Feasibility:** medium.
- **Performance win:** plausible for prefill, doubtful for single-token decode until proven.

## C. Native NN 4-bit path

Convert Bonsai weights once to a native 4-bit representation and run native NN GEMM.

### Advantages

- best chance to use all eight NN cores;
- likely easiest path for Acuity/Pegasus compilation;
- fixed-shape prefill graphs may perform well.

### Risks

- doubles packed symbol storage from about 2 bits to 4 bits before scale overhead;
- target reports `NN_4BIT_PHASE1`, but not packed coefficient mode or group quantization;
- compiler may internally expand or reject the layout;
- not the requested direct ternary execution path.

This remains a benchmark baseline, not the preferred design.

## D. CPU-native ternary kernel

Use the existing `llama.cpp` AArch64/NEON Q2_0 path as the correctness and performance baseline.

The CPU can consume packed weights directly and may outperform the single PPU for token-at-a-time GEMV. The NPU project is successful only if it improves a declared end-to-end objective versus this optimized baseline.

## 5. Recommended packed representation

The initial kernel should consume the actual `llama.cpp`/Bonsai layout rather than invent another model format.

For each group:

```text
FP16 scale
packed bytes, four 2-bit values per byte
```

Inner loop concept:

```c
byte p = packed[k >> 2];
int q0 = (p >> 0) & 3;
int q1 = (p >> 2) & 3;
int q2 = (p >> 4) & 3;
int q3 = (p >> 6) & 3;

// true ternary groups use q in {0,1,2}
acc += (q0 - 1) * a[k + 0];
acc += (q1 - 1) * a[k + 1];
acc += (q2 - 1) * a[k + 2];
acc += (q3 - 1) * a[k + 3];

out += scale * acc;
```

A VXC/EVIS implementation should avoid scalar branches. Candidate transformations:

- vector subtract-one after bit extraction;
- two bitplanes: nonzero mask plus sign mask;
- use data-path descriptors to expand packed fields into 8/16-bit lanes;
- accumulate multiple output channels per work item;
- align row starts and tiles to 256-byte external-memory bursts;
- keep scales and activation fragments in local/uniform storage;
- specialize group size 64 and 128 separately.

The bitplane representation may be faster internally, but conversion from the released GGUF format must be offline and lossless. It should be evaluated only after a direct Q2_0 reader works.

## 6. Strongest likely Bonsai partition on this SoC

### Prefill

Most promising:

- large projection/MLP matrices on native NN cores;
- PPU for RoPE, layout conversion, custom low-bit unpacking;
- fixed sequence-length buckets;
- whole MLP or transformer subgraphs to amortize launch cost;
- parallel CPU work while the NPU executes.

### Decode

Most uncertain:

- one-token decode is GEMV-like;
- high-performance decode feature is disabled;
- dynamic shape is disabled;
- packed 4-bit coefficient mode is disabled;
- PPU count is one.

Likely candidates:

1. CPU direct ternary GEMV as baseline and perhaps final decode path.
2. PPU direct packed ternary GEMV if memory bandwidth is the dominant bottleneck.
3. NPU only for selected larger linear-attention or batched/speculative-decoding work.

### Other transformer operations

The PPU/EVIS path is a natural candidate for:

- RoPE;
- RMSNorm/LayerNorm;
- activation functions;
- packing and layout conversion;
- attention masks;
- fused low-bit preprocessing.

The NN cores are a natural candidate for:

- fixed-shape Q/K/V and output projections;
- gated MLP projections;
- large prefill GEMMs;
- potentially batched decode.

## 7. Legal and maintenance boundary

The open-source pieces show enough structure to build a prototype, but the compiler/runtime boundary remains partly proprietary.

Risks:

- `libVSC`, `libCLC`, OpenVX and NBG tooling may have redistribution restrictions;
- internal EVIS headers and compiler ABI may be tied to a specific SDK release;
- precompiled VXC/NBG binaries may be incompatible with a different driver/library build;
- the public BSP feature database contains files with mixed or proprietary notices; each file must be handled according to its own notice.

Repository policy:

- commit our kernel source and build scripts;
- do not commit vendor libraries or compiler binaries without explicit permission;
- record exact SDK hashes and versions;
- require users to supply the SDK locally;
- make CPU-only builds independent from the vendor stack.

## 8. Decision gates

### Gate T1 - compiler access

A trivial custom OpenCL kernel compiles with the exact A733 SDK and runs on the target.

### Gate T2 - EVIS path

A VXC/EVIS bit-unpack kernel compiles, loads and reports correct output. Runtime profiling confirms it executes on the VIP programmable path rather than CPU fallback.

### Gate T3 - packed ternary dot product

The kernel consumes released Q2_0/group-128-style packed bytes directly and matches a CPU reference exactly within FP16 scale tolerance.

### Gate T4 - memory advantage

Measured bytes and latency show that direct packed reads beat an expanded INT8/FP16 implementation for at least one relevant shape.

### Gate T5 - end-to-end value

A complete Bonsai layer or model partition improves at least one of:

- prompt throughput;
- decode throughput;
- time to first token;
- CPU availability;
- energy per token;
- feasible model/context size.

## 9. Current verdict

The A733 VIP9000 is programmable enough to justify a real packed-ternary experiment. The SDK and upstream code provide all of the building blocks required to attempt it:

- programmable PPU/shader core;
- EVIS enabled on the exact product ID;
- OpenCL/OpenVX custom-kernel APIs;
- VSC/CLC compiler libraries;
- visible VXC/EVIS source examples;
- wide SRAM and fixed-shape graph execution.

The central limitation is architectural: the custom kernel most likely uses **one PPU core**, while the advertised NN performance comes primarily from **eight NN cores** that do not publicly expose a native packed ternary coefficient format.

Therefore the research must compare two serious designs rather than assume one answer:

1. **direct PPU/EVIS packed ternary GEMV/GEMM**;
2. **PPU tile unpack + native eight-core NN GEMM**.

A third native-INT4 path is retained only as a performance reference. The CPU packed-ternary implementation remains the mandatory baseline.