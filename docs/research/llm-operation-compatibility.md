# LLM operation compatibility: ggml/transformer → TIM-VX/VIP9000

**Date:** 2026-08-06  
**Status:** upstream capability hypothesis; requires target-board validation

## Purpose

This document maps the main operations in a decoder-only transformer to capabilities visible in the current TIM-VX source. It is not a declaration of A733 hardware support.

Legend:

- **G** — public TIM-VX operation or straightforward composition exists.
- **Y** — internal source or plausible composition exists, but public/target support is uncertain.
- **R** — no direct evidence of a native path; conversion/custom work is required.

## Compatibility table

| Transformer function | Typical ggml form | TIM-VX evidence | Status | Main uncertainty | First test |
|---|---|---|:---:|---|---|
| Token embedding lookup | `GET_ROWS` / gather | Gather and EmbeddingLookup listed | G | large table layout and random-access cost | Gather FP16/INT8 table rows |
| RMSNorm | `RMS_NORM` + multiply | internal `RMSNORM` kernel with FP16/FP32 and quantized paths | Y | not exposed by public header; SDK version | converter/direct graph probe |
| LayerNorm | normalization graph | public LayerNormalization mapping | G | supported axes and tensor ranks | FP16 last-axis norm |
| Q projection | `MUL_MAT` | Matmul/Dense/FullyConnected | G | GEMM/GEMV shapes, weight layout, quantization | shape sweep |
| K projection | `MUL_MAT` | Matmul/Dense/FullyConnected | G | same as above | shape sweep |
| V projection | `MUL_MAT` | Matmul/Dense/FullyConnected | G | same as above | shape sweep |
| Output projection | `MUL_MAT` | Matmul/Dense/FullyConnected | G | decode-size efficiency | prefill/decode comparison |
| RoPE | `ROPE` | internal RoPE op with EVIS/VX/OpenCL kernels | Y | public accessibility and exact model variant | axis/interleaving test |
| Attention score | Q × Kᵀ | Matmul with transpose options | G | rank/batch dimensions and KV length | fixed-head attention score |
| Scale scores | multiply by scalar | Multiply / Linear | G | scalar broadcasting | fused graph test |
| Causal mask | add/select/broadcast | Add, Select, Broadcast, Pad, Slice | G/Y | dynamic sequence and bool/Gather paths | fixed additive FP mask |
| Softmax | `SOFT_MAX` | public Softmax mapping | G | axis/rank/large-context behavior | FP16 attention softmax |
| Attention value | P × V | Matmul | G | batched shape and memory | fixed attention block |
| Residual add | `ADD` | Add | G | in-place/alias handling | FP16 add |
| MLP up projection | `MUL_MAT` | Matmul/Dense | G | weight residency | shape sweep |
| MLP gate projection | `MUL_MAT` | Matmul/Dense | G | weight residency | shape sweep |
| SiLU/Swish | unary op | Swish mapped; Sigmoid + Multiply available | G | numerical match and fusion | native vs composed |
| GELU | unary op | GELU mapped | G | approximation variant | exact/approx comparison |
| Gated multiply | `MUL` | Multiply | G | broadcasting/layout | MLP graph |
| MLP down projection | `MUL_MAT` | Matmul/Dense | G | output shape and decode efficiency | shape sweep |
| Reshape/view | views/reshape | Reshape | G | arbitrary strides are not guaranteed | contiguous reshape |
| Permute/transpose | `PERMUTE` | Transpose | G | conversion cost/layout inference | transpose microbench |
| Slice/KV append | views/copy/set | Slice, StridedSlice, variable tensors | Y | mutation, aliasing, dynamic positions | persistent state test |
| KV-cache state | tensors across calls | Variable tensor concept exists | Y | whether state remains device-resident and mutable | recurrent-state graph |
| Concatenate KV | concat/copy | Concat | G/Y | copying entire cache each token is unacceptable | fixed-window test |
| Cast/dequantize | conversion kernels | Cast/DataConvert and quantization metadata | G | fused conversion and exact formats | INT8↔FP16 test |
| GGUF Q4/Q5/Q6 weights | packed block formats | no direct mapping | R | must unpack/repack or custom kernel | offline conversion cost |
| Bonsai ternary weights | packed `{-1,0,+1}` | no ternary public type | R | custom kernel vs INT4/INT8 expansion | static weight conversion |
| Sampling/top-k | logits postprocess | TopK limited; random multinomial TBD | R/G | better suited to CPU | keep on CPU |
| Tokenizer | CPU string processing | not an NPU workload | R | none | keep on CPU |

## Key architectural conclusions

### 1. Embedding lookup is not the main target

Embedding lookup is mostly memory access. It can be represented through Gather/EmbeddingLookup, but it is not where the bulk of decoder compute occurs. It should be tested only after the main projection kernels, unless moving the whole transformer graph requires the embedding to be inside the NPU graph.

### 2. Prefill and decode must be treated as separate products

For prefill, projection matrices see many tokens and look like conventional GEMM. For decode, the same layers often become narrow GEMV-like operations. An NPU can be excellent at one and disappointing at the other.

Required shape families for a model with hidden size `H`, intermediate size `I`, and prompt length `T`:

```text
Prefill projections:
  [T, H] × [H, H]
  [T, H] × [H, I]
  [T, I] × [I, H]

Decode projections:
  [1, H] × [H, H]
  [1, H] × [H, I]
  [1, I] × [I, H]

Attention:
  [heads, T, head_dim] × [heads, head_dim, kv_len]
  [heads, T, kv_len] × [heads, kv_len, head_dim]
```

### 3. Whole-subgraph offload is more credible than one-op offload

A graph accelerator wants stable, reusable graphs. Calling the runtime for every individual `ggml` node risks excessive launch, synchronization, layout, and transfer overhead.

First meaningful graph candidates:

1. RMSNorm + Q/K/V projections + RoPE;
2. complete attention prefill;
3. RMSNorm + gated MLP;
4. complete transformer block;
5. multiple consecutive layers, if memory allows.

### 4. Bonsai ternary execution is a separate research branch

The NPU may still help a ternary Bonsai model even without ternary hardware:

- convert weights once to INT8 or INT4 and retain the converted layout;
- run activations and projections in a supported quantized format;
- compare against CPU ternary kernels.

This trades Bonsai's compact packed representation for NPU compatibility. The converted weights may consume significantly more memory, so the 12 GB system-memory budget and graph compiler behavior are decisive.

Direct ternary custom kernels should be investigated only after verifying custom-kernel execution hardware and memory bandwidth.

## Minimum operator probe matrix

For every candidate operation, record:

- compiler/import success;
- runtime load success;
- runtime execute success;
- accepted rank and dimensions;
- accepted layouts;
- accepted input/weight/output types;
- quantization scheme;
- cold compile/load latency;
- warm execution latency;
- host↔device bytes and time;
- numerical error;
- runtime/kernel logs;
- CPU utilization and thermals.

Suggested initial types:

```text
FLOAT32
FLOAT16
INT8 asymmetric per-tensor
INT8 symmetric per-channel weights
INT4/UINT4 only where a real operation test succeeds
```

Suggested first operations:

```text
MatMul
Add
Multiply
Reshape
Transpose
Softmax
LayerNormalization
RMSNorm
RoPE
Gather
Swish
Cast/DataConvert
```

## Decision gates

### Gate A — useful MatMul

Proceed to transformer subgraphs only if at least one realistic prefill projection is faster or materially reduces CPU load versus optimized CPU after including buffer handling.

### Gate B — persistent graph

Proceed to backend integration only if a compiled network and weights can remain resident across repeated calls.

### Gate C — state handling

Proceed to attention decode only if KV state can be updated without copying the complete cache every token.

### Gate D — quantization viability

Proceed to Bonsai integration only if converted weights fit memory and the NPU path beats the CPU ternary implementation on an end-to-end objective.

## Sources

- TIM-VX operator catalogue: <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/ops/README.md>
- TIM-VX types: <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/types.h>
- TIM-VX tensors: <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/tensor.h>
- RMSNorm internal op: <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/ops/vsi_nn_op_rmsnorm.c>
- RoPE internal op: <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/src/ops/vsi_nn_op_rope.c>
- `ggml` backend API: <https://github.com/ggml-org/llama.cpp/blob/9de0fcf2b3e587a43f293d9a2b6ec0a32991f768/ggml/include/ggml-backend.h>
