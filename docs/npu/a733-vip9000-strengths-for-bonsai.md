# A733/VIP9000 strengths and weaknesses for Ternary Bonsai

## Summary

The A733 accelerator is strongest when it receives a fixed-shape, reusable graph with enough parallel work to keep eight NN cores busy. It also contains one programmable PPU/shader core with EVIS for custom vector kernels and data transformation.

This makes it a promising **hybrid prefill accelerator**, but not an obviously ideal token-at-a-time ternary decoder.

## Strongest hardware properties

### Eight native NN cores

The exact A733 product configuration reports eight active NN cores for INT8 and INT16 workloads, native GEMM phase-1 support, NN ALU support and small-batch phase-2 behavior.

Best candidates:

- Q/K/V projections;
- output projection;
- MLP up/gate/down projections;
- fixed-shape attention/MLP subgraphs;
- prompt prefill;
- batched or speculative decode.

### Programmable PPU with EVIS

The product reports one shader/PPU core, 256 threads and EVIS enabled. The SDK contains OpenCL/VXC compiler libraries and TIM-VX contains custom-kernel examples.

Best candidates:

- packed ternary decoding experiments;
- RoPE;
- RMSNorm/LayerNorm and activation functions;
- tensor layout conversion;
- fused preprocessing;
- tile unpacking for native NN nodes.

### 512 KiB on-chip VIP SRAM

This is valuable for:

- activation tiles;
- partial accumulators;
- short-lived expanded INT8 weight tiles;
- reducing repeated DDR traffic;
- producer/consumer fusion where the compiler supports it.

### Wide, burst-oriented memory system

The feature database reports:

- 128-byte physical VIP SRAM width;
- 128-bit AXI bus width;
- 256-byte minimum/DDR kernel burst sizes.

This favors:

- large contiguous packed-weight reads;
- row/tile alignment;
- reuse of activation blocks;
- fixed-size compiled graphs.

### PPU/NN parallelism at the VIP9000 family level

VeriSilicon documents parallel execution between programmable PPU and NN accelerators. If the A733 runtime exposes this effectively, tile unpacking and native NN GEMM may overlap.

This must be verified on the exact product and SDK.

## Weakest properties for Bonsai

### No native ternary format

No public ternary tensor or matrix instruction has been identified. Packed Bonsai weights cannot be assumed to feed native NN GEMM directly.

### No packed 4-bit coefficient mode

Although the target reports `NN_4BIT_PHASE1`, it reports no `NN_4BIT_COEF_PACKED_MODE`. This weakens the case for treating the target as a modern native low-bit LLM engine.

### No group quantization feature

Bonsai uses one scale per small weight group. The target reports no NN group-quant phase-1 support. The compiler may need to split graphs, fold scales, or expand data.

### No high-performance decode feature

The product reports no dedicated high-performance decoder feature. One-token GEMV may be less efficient than large prefill GEMM.

### No dynamic-shape support

Context length and attention state cannot be assumed to vary freely inside one compiled graph. Bucketed shapes, padding, fixed windows or several cached graph variants will probably be required.

### One programmable PPU

Custom OpenCL/EVIS code likely runs on one PPU core rather than the eight NN cores. A custom ternary kernel may save memory traffic but still lose in arithmetic throughput.

### No dedicated tensor DMA feature

PPU-to-NN intermediate traffic and cache synchronization may be costly. Tile-unpack designs require direct measurement.

## Likely optimal role allocation

| Work | First-choice engine | Reason |
|---|---|---|
| Tokenizer and sampling | CPU | irregular control/string work |
| Packed ternary decode GEMV | CPU vs PPU benchmark | compact memory format; small-M workload |
| Prefill projection GEMM | NN cores | large fixed-shape parallel arithmetic |
| Batched/speculative decode | NN cores | larger M improves utilization |
| Ternary bit unpack/repack | PPU/EVIS | programmable vector/bit operations |
| RoPE | PPU/EVIS or native graph op | existing EVIS implementation upstream |
| RMSNorm/LayerNorm | PPU/EVIS/native op | reduction/vector work |
| MLP activation/gating | graph compiler / PPU | possible fusion with native projections |
| KV-cache management | CPU or fixed graph state | dynamic mutation remains uncertain |
| Vision tower | native NPU graph | conventional quantized graph workload |

## Expected project architecture

The best solution is likely a hybrid scheduler rather than “run everything on the NPU”:

```text
llama.cpp / ggml scheduler
        |
        +-- CPU backend
        |     direct packed ternary decode and fallback
        |
        +-- VIP PPU/EVIS backend
        |     custom low-bit and vector kernels
        |
        +-- VIP native NN graph backend
              fixed-shape prefill and large projections
```

## What would change the conclusion

A full native ternary backend becomes much more attractive if target experiments show any of the following:

- a compiler-supported custom instruction can execute inside or directly feed the eight-core NN datapath;
- the stream processor can consume packed ternary weights and accumulate efficiently;
- PPU and NN nodes share transient SRAM tensors without a DDR round trip;
- PPU direct GEMV substantially exceeds CPU memory bandwidth efficiency;
- fixed graph variants handle decode state with low launch/synchronization cost.

Until then, the project should optimize for **measured heterogeneous execution**, not maximum nominal NPU utilization.