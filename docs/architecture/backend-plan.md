# Backend architecture and bottleneck plan

## Current bottleneck hypothesis

The qualified CPU baseline and the NPU probes point to a heterogeneous,
memory-sensitive decode problem rather than a single missing instruction:

- E035's native Q1 CPU path is the current full-model reference at
  `0.972497 tok/s`.
- The packed Q1 EVIS kernel is correct on synthetic and real tiles, but a
  measured `1024x5120` tile takes `24.137 ms` end to end while the CPU completes
  the full `17408x5120` layer in `3.850 ms`.
- E049d-v2 sees large instruction, backend-stall, and cache/bus event counts,
  but those PMU events do not identify DDR bytes or separate Q1 unpack from
  weight streaming.
- VIPLite overhead and shared-memory visibility are material for small graphs;
  repeated resident execution is required before comparing steady state.

The working bottleneck model is therefore **Q1 weight movement plus unpack and
GEMV scheduling**, with the dominant share still to be separated. The next
measurement must distinguish memory traffic, unpack, dot product, and launch
overhead without claiming unsupported DDR bandwidth.

## Backend sequence

1. Keep optimized AArch64 CPU Q1 as the correctness and throughput reference.
2. Keep VIPLite/NBG graphs resident and measure init, load, buffer import,
   cache maintenance, `run`, device time, and output visibility separately.
3. Test one fused EVIS-unpack to native NN FC/Conv graph on a small tile. The
   intermediate must remain on-chip or in a proven resident buffer; a DDR spill
   is a rejection condition.
4. Integrate only a validated fixed-shape operation into a `ggml` seam with a
   capability predicate, explicit buffers, synchronization, graph-cache key,
   and CPU fallback.
5. Expand to a cost-aware subgraph only after prefill and decode A/B runs show
   an end-to-end gain with exact quality.

Do not place graph conversion, compilation, or a per-token ONNX pipeline in the
decode loop. Cache artifacts by device/product ID, runtime/SDK identity,
operation graph, shapes, types, layout, and backend version.

## Partition policy

An operation is eligible for NPU work only if its types, shapes, layout, and
memory path are proven; conversion and launch costs are bounded; a CPU golden
exists; and the compiled graph is reusable. Unsupported or unprofitable nodes
remain on CPU. Preserve model Q1 storage and avoid full INT8/FP16 weight
expansion in DDR.

Prefill has larger matrices and is the first likely accelerator target. Decode
is matrix-vector dominated and must be treated separately: small-M launch,
packing, cache, and synchronization can erase a device-time win. A partition
that improves a device-only tile but worsens full decode is rejected.

## Measurement gates for the next candidate

- Same model SHA, prompt suite, seed, software, and target state as the pinned
  reference.
- Independent CPU golden and deterministic token comparison.
- At least the declared warmups and five comparable measured repetitions.
- H2D, D2H, host run, device interval, total wall time, bytes, and phase data.
- No thermal trip, unexplained reset, swap growth, fallback, or missing raw
  boundary.
- No inference from PMU event counts to DDR bytes.

## Evidence pointers

- [CPU optimization matrix](../evidence/a733-cpu-optimization-matrix-2026-08-09.md)
- [Packed Q1 EVIS scaling boundary](../evidence/q1-vip9000-evis-bonsai-2026-08-11.md)
- [VIPLite phase profile](../evidence/vip9000-phase-profile-2026-08-10.md)
- [Bonsai Q1 partition audit](../evidence/bonsai-q1-npu-partition-audit-2026-08-10.md)
- [Current experiment registry](../experiments/registry.json)
