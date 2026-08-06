# E002 - Packed ternary kernel on A733/VIP9000

## Objective

Determine whether the A733 `VIP9000NANODI_PLUS_PID0X1000003B` can consume Ternary Bonsai Q2_0-style packed weights directly, without permanently expanding the model to INT4, INT8 or FP16, and whether doing so provides an end-to-end advantage over optimized AArch64 CPU execution.

This experiment tests two accelerator designs:

1. direct packed ternary GEMV/GEMM on the programmable PPU/EVIS path;
2. PPU/EVIS tile unpack followed by native NN-core GEMM.

Native INT8/INT4 and CPU packed-ternary paths are mandatory baselines.

## Research hypotheses

### H1 - compiler availability

The A733 SDK can compile and run a custom OpenCL kernel through OpenVX/TIM-VX.

### H2 - EVIS availability

The exact target can compile or load a VXC/EVIS kernel that uses vector data-path instructions.

### H3 - direct packed consumption

A custom kernel can read released Bonsai Q2_0/group-128-style bytes, decode four ternary symbols per byte, apply the group FP16 scale and produce a correct dot product without an expanded weight tensor.

### H4 - useful performance

For at least one representative Bonsai projection shape, the packed accelerator path improves latency, bandwidth, CPU availability or energy compared with the optimized A733 CPU implementation.

## Non-goals

- claiming access to all eight NN cores from an OpenCL custom kernel;
- replacing the complete `llama.cpp` backend in this experiment;
- modifying model values or retraining Bonsai;
- distributing proprietary SDK binaries;
- presenting family-level VIP9000 marketing features as confirmed target behavior.

## Required local assets

Do not commit these files unless their licenses explicitly permit redistribution:

- exact A733/Orange Pi SDK;
- OpenVX/VIPLite/VIPhal/VSC/CLC libraries;
- Acuity/Pegasus or Vivante IDE package;
- target driver/firmware package;
- Bonsai GGUF model weights;
- generated NBG/VXC binaries where redistribution is unclear.

Record SHA-256 hashes and version output in a sanitized manifest.

## Phase 0 - target and SDK fingerprint

Capture:

```bash
uname -a
cat /etc/os-release
lscpu
ls -l /dev/vipcore /dev/galcore 2>/dev/null || true
find /lib /usr/lib /opt "$HOME" -type f \
  \( -name 'libOpenVX.so*' -o -name 'libCLC.so*' -o -name 'libVSC.so*' \
     -o -name 'libNNVXCBinary.so*' -o -name 'libNBGlinker.so*' \
     -o -name 'libVIPhal.so*' \) 2>/dev/null
strings <path-to-runtime-library> | grep -Ei 'version|VIP9000|1000003b' | head -100
```

Also record:

- SDK archive/package version;
- compiler container/image digest;
- `VSIMULATOR_CONFIG`;
- NPU product/customer ID;
- driver and firmware versions;
- exact TIM-VX revision, if used;
- licenses for headers, libraries and generated artifacts.

## Phase 1 - custom OpenCL smoke test

Build the smallest custom operation using the upstream TIM-VX `CustomOpBase` pattern.

Suggested graph:

```text
input FP32/FP16 -> custom vector add or copy -> output
```

Required checks:

- source compilation succeeds;
- graph verification succeeds;
- execution succeeds repeatedly;
- output matches CPU reference;
- compiler cache behavior is understood;
- profiler/log evidence identifies the execution device;
- no silent CPU fallback occurs.

Record cold compile, graph creation, graph verification and warm execution separately.

## Phase 2 - VXC/EVIS bit extraction

Implement a kernel that accepts packed bytes and expands only a small output vector:

```text
input byte: [q3 q2 q1 q0], two bits each
output lanes: q0-1, q1-1, q2-1, q3-1
```

Test all 256 input byte values. For strict ternary tests reject or separately count code `11`.

Implementations:

1. portable OpenCL shifts/masks;
2. VXC/EVIS vector extraction using `_viv_asm` / programmable DP descriptors;
3. optional two-bitplane representation generated offline.

Success criteria:

- exact output for all cases;
- confirmed VIP execution;
- measured packed-input bandwidth;
- EVIS implementation is faster than portable OpenCL or its maintenance cost is justified.

## Phase 3 - direct packed ternary dot product

Reference operation:

```text
y = scale * sum_i(t_i * x_i)
t_i in {-1, 0, +1}
```

Initial activation types:

- FP16;
- FP32 for reference;
- optional INT8 activation path after correctness is stable.

Test group sizes:

- 64 for current mainline `llama.cpp` Q2_0;
- 128 for the original Bonsai release/fork format.

Test vector lengths:

```text
128
256
512
1024
2048
4096
8192
```

For each case record:

- packed bytes read;
- scale bytes read;
- activation bytes read;
- output bytes written;
- warm latency;
- effective GB/s;
- effective ternary weight operations/s;
- CPU utilization;
- maximum/mean absolute error;
- FP16-scale error separately from symbol-decoding error.

The symbol path should be bit-exact. Differences should come only from accumulation/scaling precision.

## Phase 4 - GEMV projection

Implement one matrix-vector projection with packed rows:

```text
Y[N] = W[N,K] * X[K]
```

Suggested synthetic shape families until the exact Bonsai metadata is imported:

```text
K: 1024, 2048, 4096, 5120, 8192
N: K, 2*K, 3*K, 4*K
M: 1
```

Then replace them with exact hidden/intermediate/projection dimensions from the selected Bonsai model.

Kernel variants:

1. one output row per work item;
2. multiple rows per work group to reuse activation tiles;
3. vectorized packed-byte loads;
4. scale-group specialization for 64 and 128;
5. output accumulation in FP16 and FP32;
6. optional output-channel tiling aligned to 256-byte DDR bursts.

## Phase 5 - prefill GEMM

Extend to:

```text
Y[M,N] = X[M,K] * W[K,N]
```

Test:

```text
M: 8, 16, 32, 64, 128, 256
```

This phase determines whether the custom path is useful for prompt processing even if decode remains on CPU.

## Phase 6 - tile unpack plus native NN GEMM

Build a graph or paired runtime sequence:

```text
packed ternary tile
    -> custom PPU/EVIS unpack to INT8 tile
    -> native MatMul/Dense node on NN cores
```

Required variants:

1. expanded tile in ordinary system memory;
2. shared/handle-swapped tensor where supported;
3. on-chip/transient graph tensor where supported;
4. double-buffered tiles to overlap unpack and NN execution;
5. pre-expanded weight cache as an upper-bound reference.

Measure:

- unpack latency;
- intermediate bytes written/read;
- synchronization overhead;
- native NN GEMM latency;
- overlap efficiency;
- total end-to-end latency.

This path fails if the intermediate traffic removes the benefit of the compact model.

## Phase 7 - baselines

### CPU packed ternary

Use the pinned upstream/fork AArch64 implementation with NEON enabled.

Record:

- commit SHA;
- compiler flags;
- thread count and affinity;
- memory governor and clocks;
- direct Q2_0/Q2_0-g128 packed execution;
- warm GEMV and prefill performance.

### Native INT8

Expand or convert weights once to INT8 and benchmark the best native NN graph. This estimates the eight-core NN upper bound while preserving exact ternary values numerically.

### Native INT4

Attempt only if the exact compiler accepts the target graph. Record whether packed coefficients are retained or internally expanded. This is a comparison path, not the project objective.

## Correctness protocol

Generate random activations and ternary weights using deterministic seeds.

Reference:

```python
for group in groups:
    out += scale[group] * dot(ternary[group], activation[group])
```

Record:

- exact decoded symbol agreement;
- max absolute error;
- mean absolute error;
- relative error;
- cosine similarity for vectors;
- layer output comparison using real Bonsai weights;
- final logits and deterministic-token agreement for integrated tests.

## Benchmark protocol

- separate compile/load/first-run/warm-run timings;
- at least 20 warm repetitions for microkernels;
- report median, p10, p90, minimum and maximum;
- record thermals and clocks before/after;
- record memory bandwidth where counters are available;
- record CPU time, not only wall time;
- pin runtime and model hashes;
- keep raw results in JSONL.

Suggested JSONL fields:

```json
{
  "experiment": "E002",
  "backend": "ppu-opencl|ppu-evis|ppu-unpack-nn|cpu-neon|nn-int8|nn-int4",
  "operation": "unpack|dot|gemv|gemm",
  "model_format": "q2_0_g64|q2_0_g128",
  "m": 1,
  "n": 4096,
  "k": 4096,
  "activation_type": "f16",
  "accumulator_type": "f32",
  "packed_weight_bytes": 0,
  "temporary_bytes": 0,
  "latency_us": 0.0,
  "cpu_time_us": 0.0,
  "max_abs_error": 0.0,
  "device_id": "0x1000003b",
  "sdk_version": "unknown",
  "runtime_version": "unknown",
  "git_commit": "unknown"
}
```

## Decision matrix

| Result | Decision |
|---|---|
| Direct PPU packed GEMV beats CPU decode | pursue a `ggml` packed-ternary accelerator backend |
| Direct PPU loses decode but wins prefill | hybrid CPU decode + NPU prefill |
| Tile-unpack + NN wins prefill | develop fused graph/subgraph compiler path |
| Tile-unpack + NN wins decode | investigate weight-tile caching and layer fusion |
| Native INT8 wins but doubles memory | evaluate memory/performance tradeoff per model size |
| Every NPU path loses optimized CPU | keep NPU for embeddings/other graphs; optimize CPU Bonsai |
| Custom compiler path unavailable | focus on precompiled NBG and vendor-supported operations |

## Deliverables

```text
artifacts/target-manifest.yaml
artifacts/sdk-manifest.yaml
src/opencl-smoke/
src/evis-unpack/
src/ternary-dot/
src/ternary-gemv/
src/unpack-native-nn/
results/raw/*.jsonl
results/summary.md
docs/npu/target-verified-ternary-capabilities.md
```

Do not commit proprietary binaries or model weights.

## Exit criteria

E002 is complete when we can answer, from target measurements:

1. Can custom OpenCL execute on the A733 VIP programmable core?
2. Can EVIS/VXC source be compiled with the exact SDK?
3. Can released Bonsai packed weights be consumed directly?
4. Does direct PPU execution beat CPU for decode or prefill?
5. Can packed-to-INT8 tiles feed native NN GEMM without prohibitive traffic?
6. Which Bonsai layers should run on CPU, PPU and NN cores?
7. Is a general `ggml` backend justified, or should integration remain model-specific?