# E055 — cache-hot/cold Q1 microbenchmark

Status: **REVISED STAGE 1 IMPLEMENTED; NO TARGET RUN**.

E055 is a bounded experiment designed to distinguish cache/data-carrier cost
from unpack/compute cost in the stock Q1_0 4×4 kernel on A733. It is not a
Bonsai-27B run and it provides no tokens/s result. A full Bonsai `n_predict=32`
run remains prohibited until this gate is reviewed and passes on the board.

## Scope and duplicate preflight

Prior branches already tested 4×8, simple PRFM, Q8 reuse, whole-K pairing,
register LUTs, scheduling, and several repacks. E055 does not introduce another
prefetch or representation. It holds the stock layout and traversal constant
while changing cache conditioning and operation controls. The all-local/all-
remote branch audit is preserved in
[`data/branch-preflight.json`](data/branch-preflight.json). The original Stage 1
review rejection is preserved in
[`data/review-rejection-stage1-51d1c1c.md`](data/review-rejection-stage1-51d1c1c.md).

## Exact kernel and controls

The harness includes the E039 fixture and calls its unmodified
`native_simd_group`: one native carrier is a 72-byte `block_q1_0x4` plus four
34-byte `block_q8_0` blocks, or 208 input bytes and 128 Q1 values. E039's scalar
oracle must match the DOTPROD output bit-for-bit over 18 cases. No GGUF or model
tensor is published.

| Mode | Retained work | Deliberately absent |
|---|---|---|
| `packed_stream` | stock-order Q1 scale/sign and Q8 scale/data loads; four bounded vector consumers; one final reduction/sink | sign unpack, SDOT, FMA |
| `unpack_scale` | exact two-byte LUT sign expansion, all Q8/scale loads; four bounded vector consumers; one final reduction/sink | SDOT, FMA |
| `full_dotprod` | exact stock E039 4×4 NEON/DOTPROD and FP32 accumulation | nothing from the stock kernel |

The controls no longer contain the rejected per-byte serial hash or per-block
reductions/stores. Their vector XOR/add consumers, scale-bit accumulation, loop
control, one final reduction, and one global sink are still overhead. They are
controls, not zero-cost substitutes. Optimized AArch64 disassembly is checked
for retained loads/vector consumes, no control-kernel calls or SDOT, and stock
SDOT in `native_simd_group`; see
[`data/disassembly-review.json`](data/disassembly-review.json). Static
disassembly proves instruction shape, not board timing.

## Hot/cold protocol and normalization

- `hot_repeat` warms the fixture, then repeatedly traverses the same carrier
  working set for a time budget or explicit iteration count.
- `cold_conditioned` writes and verifies every 64-byte line of a separate
  thrash buffer before the marker, then performs **exactly one** traversal.
  Any override to more than one iteration or a nonzero time budget is rejected.
  This is named conditioning, not proof that every architectural cache set
  missed.

All comparisons normalize to `ns/traversal` (or its inverse,
`traversals/s`). For example, 250 hot calls taking 250 ms and one cold call
taking 1 ms both equal 1 ms/traversal, so the cold/hot penalty is exactly 1.0.
Total windows with different call counts are never divided directly.

Cold penalties are compared like-for-like only: full cold/full hot,
packed-stream cold/packed-stream hot, and unpack-scale cold/unpack-scale hot at
the same CPU, carrier size, and block count. Cross-mode elapsed times are not
blindly subtracted. A ratio-of-ratios may show whether the full-kernel penalty
tracks its load-only control, but it remains a directional inference.

## Qualification contract

The standalone harness emits `e055-q1-hot-cold-harness/v1`, explicitly marked
unqualified. A target runner must join it with E049c and produce strict
`e055-q1-hot-cold/v2`. Qualification requires:

- exact fd9 `S`, fd8 `ACK`, fd9 `E` synchronization;
- E049c v2 `sample_valid=true`, one nonempty exact `core`, `cache`, or `memory`
  group, supported/valid events, and `running_ratio == 1.0` for every event;
- readable thermal telemetry with no trip and maximum temperature at or below
  its limit;
- affinity containing only CPU0 (A55) or CPU6 (A76), identical start/end CPU,
  and zero migrations;
- run ID, pair ID, pair order, within-pair order, zero exit code, exact golden,
  and source/binary/compiler SHA-256 provenance.

`sync=false`, missing/empty PMU events, multiplexed counters, a thermal failure,
CPU migration, or missing provenance fails closed. PMU values are event counts,
not bytes. E055 has no direct DDR-byte counter and never relabels refill or
access events as traffic.

## Planned board matrix

Working sets round upward to complete 208-byte carriers:

`64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB, 4 MiB, 12.5 MiB`.

CPU0 and CPU6 are measured first. Every cell needs at least five alternating
pairs/repeats in each E049c PMU group. Only if the resulting component model
predicts at least a 4% Q1 gain may one bounded all-core production-shape gate
and a subsequent optimization be recommended. Failures are published as raw
evidence rather than converted to zero-valued samples.

## Reproduction of revised Stage 1

Run the adversarial host, cross/QEMU, and disassembly gates:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e055_q1_hotcold tests.test_e055_harness_contract \
  tests.test_e055_aarch64_gate -v
```

The cross command used by the test is:

```text
aarch64-linux-gnu-g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -march=armv8.2-a+dotprod tooling/e055_q1_hotcold.cpp \
  experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S \
  -o /tmp/e055-q1-hotcold-aarch64
```

QEMU must report `golden_pass=true` and `golden_cases=18`. QEMU results are
functional evidence only and are not used as A733 performance evidence. The
first uninitialized-LUT failure remains at
[`data/failure-qemu-uninitialized-lut.txt`](data/failure-qemu-uninitialized-lut.txt):
it was a harness initialization bug, was rejected, and must not be interpreted
as a hardware or mathematical result.

No board workload, model run, OPP/DDR change, NPU run, or full-model bottleneck
claim is part of this revised Stage 1.
