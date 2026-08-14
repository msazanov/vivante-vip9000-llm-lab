# Q1 representation and partitioning

## Canonical Q1 carrier

For the Bonsai Q1_0 workload, one `K=128` block is 18 bytes:

- 2 little-endian bytes for the FP16 scale `d1`;
- 16 bytes for 128 LSB-first sign bits;
- bit `1` represents `+d1` and bit `0` represents `-d1`.

The Q8_0 activation uses four 34-byte subblocks: a 2-byte FP16 scale followed
by 32 signed INT8 values. The exact integer identity for one subblock is:

```text
sum((2 * bit - 1) * q) = 2 * sum(bit * q) - sum(q)
```

The direct EVIS proof uses register-local `0/1` bits and this identity. It does
not materialize a full `-1/+1` or INT8 weight tensor in DDR and does not change
the GGUF model format.

## What the evidence says

The synthetic Q1 x Q8 EVIS gate is exact and stable. Real Bonsai tiles also
pass an independent golden, but the large programmable tile is much slower
than the CPU full layer. Therefore direct EVIS Q1 is a capability proof and a
diagnostic building block, not a qualified full-model backend.

The fused packed-Q1 path is the preferred next hypothesis: EVIS unpacks a small
tile, then a native NN FC/Conv operation performs the dense work. Preserve the
scale semantics and output layout, keep the tile on-chip where proven, and
reject a DDR-spilling intermediate or any quality loss.

## CPU / EVIS / native-NN partition

| Work | Initial owner | Promotion condition |
|---|---|---|
| Tokenization, sampling, graph scheduling, unsupported ops | CPU | Keep on CPU unless a complete subgraph proves otherwise. |
| Optimized Q1 decode GEMV | CPU | Reference remains E035 until a full-model gate beats it exactly. |
| Packed sign unpack microkernel | EVIS/PPU | Accept only with independent golden and a measured role in a fused graph. |
| Large prefill FC/Conv candidate | Native NN | Require exact shape/layout, selected native plan, no DDR spill, and end-to-end gain. |
| KV-cache updates and dynamic sequence state | CPU first | Move only after target runtime semantics and cache cost are proven. |

Prefill and decode are different shapes. Prefill offers enough parallelism to
amortize a resident graph; decode is small-M and launch/visibility sensitive.
Run separate gates for each. Do not use a prefill win to claim decode speed.

## Correctness and memory gates

- Compare every output against an independent CPU golden for operator work.
- Preserve Q1 scale and Q8 scale arithmetic exactly; report max/mean error and
  exact-match status.
- Keep packed model bytes as the long-lived representation.
- Record all copies, cache maintenance, layout conversions, intermediate size,
  and peak RSS.
- Do not infer DDR bytes from ARM PMU `mem_access`, `bus_access`, or refill
  counts.

## Evidence pointers

- [Packed Q1 EVIS and Bonsai scaling](../evidence/q1-vip9000-evis-bonsai-2026-08-11.md)
- [Packed carrier design](../evidence/q1-uint8-nbg-packed-carrier-design-2026-08-10.md)
- [Bonsai Q1 NPU partition audit](../evidence/bonsai-q1-npu-partition-audit-2026-08-10.md)
- [Current qualified result](../experiments/current-best.md)
