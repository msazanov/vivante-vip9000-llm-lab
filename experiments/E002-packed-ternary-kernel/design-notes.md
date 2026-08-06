# E002 design notes

## Target data path

The first prototype must preserve the released Q2_0-style storage format:

```text
[group FP16 scale][packed 2-bit symbols]
```

Four symbols are stored per byte. A direct kernel should load packed bytes and activations, decode symbols in registers/vector lanes, accumulate, and apply the group scale. It must not create an expanded full weight matrix.

## Candidate implementations

### Portable OpenCL reference

Use ordinary shifts, masks and vector arithmetic. This is the easiest path to validate the compiler and memory layout.

Pseudo-code:

```c
for each output row:
    float acc_total = 0;
    for each scale group:
        float acc_group = 0;
        for each packed byte:
            uchar p = packed_weight[byte_index];
            int4 q = (int4)(
                (p >> 0) & 3,
                (p >> 2) & 3,
                (p >> 4) & 3,
                (p >> 6) & 3);
            q -= 1;
            acc_group += dot(convert_float4(q), load4(activation));
        acc_total += scale[group] * acc_group;
    output[row] = acc_total;
```

The production kernel should remove scalar loops where the compiler cannot unroll them.

### EVIS/VXC extraction

Use VXC vector loads and programmable DP descriptors to:

1. load packed bytes;
2. replicate/extract 2-bit fields into 8/16-bit lanes;
3. subtract the code bias;
4. multiply/add against activation lanes;
5. accumulate multiple rows or output channels.

The existing TIM-VX RoPE kernels demonstrate the mechanism:

- VXC source in `.vx` files;
- `_viv_asm` operations;
- `VXC_DP4x4` / `VXC_DP2x8`;
- host-side `gpu_dp_inst_t` descriptors passed as uniforms.

### Two-bitplane representation

Optional offline, lossless repacking:

```text
nonzero bitplane
sign bitplane
FP16 scale per group
```

Computation:

```text
nonzero=0 -> skip
nonzero=1 and sign=0 -> add
nonzero=1 and sign=1 -> subtract
```

This remains 2 bits per weight and can be friendlier to bitwise/vector logic. It is not the released GGUF layout, so direct Q2_0 support must be completed first.

### PPU unpack to NN tile

For tile dimensions selected to fit within the 512 KiB VIP SRAM budget:

```text
packed weight tile -> INT8 tile -> native MatMul/Dense
```

The experiment should reserve space for:

- expanded weight tile;
- activation tile;
- partial output/accumulator;
- runtime metadata and alignment.

Do not assume the full 512 KiB is available to one user graph.

## Optimization principles

- align packed rows and tile starts to 256-byte external-memory bursts;
- reuse activation fragments across many output rows;
- specialize group sizes 64 and 128;
- avoid divergent branches for zero/sign handling;
- accumulate in FP32 initially, then test FP16;
- separate symbol decoding cost from scale/accumulation cost;
- keep graph/kernel objects resident;
- measure map/flush/invalidate and synchronization explicitly;
- test decode and prefill independently;
- use CPU NEON direct-packed execution as the mandatory reference.

## Critical unknowns

- Does the exact board SDK expose the VXC compiler headers used by TIM-VX internal kernels?
- Can custom VXC kernels be packaged into NBG and loaded by VIPLite?
- Can a custom kernel output be consumed by a native NN node without DDR round-trip?
- Can PPU and NN cores overlap on this product/runtime?
- Can weights be mapped from host memory without a copy?
- Does the runtime expose reliable per-engine profiling?
- Is the stream processor directly programmable by user code or only by the graph compiler?
- Are custom kernel binaries redistributable?

## Expected first-order outcome

The most likely outcome is not a single all-NPU backend. A realistic optimal partition may be:

```text
CPU:
  tokenizer, sampling, graph control, unsupported operations,
  possibly token-at-a-time packed ternary GEMV

PPU/EVIS:
  RoPE, normalization, packing/unpacking, direct ternary experiments

8 NN cores:
  large fixed-shape prefill projections and MLP graphs,
  possibly batched/speculative decode
```

E002 exists to replace this hypothesis with measured target data.