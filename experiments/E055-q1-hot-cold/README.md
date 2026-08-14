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
The rejected `bba62cb` revision and its defects are preserved separately in
[`data/review-rejection-stage2-bba62cb.md`](data/review-rejection-stage2-bba62cb.md).
The rejected `0ed991b` target qualifier is preserved in
[`data/review-rejection-stage3-0ed991b.md`](data/review-rejection-stage3-0ed991b.md).

## Exact kernel and controls

The harness includes the E039 fixture and calls its unmodified
`native_simd_group`: one native carrier is a 72-byte `block_q1_0x4` plus four
34-byte `block_q8_0` blocks, or 208 input bytes and 128 Q1 values. E039's scalar
oracle must match the DOTPROD output bit-for-bit over 18 cases. No GGUF or model
tensor is part of this bounded Stage 1 publication. The upstream source is the
immutable `ggml-org/llama.cpp` commit
`38c66ad0241da4f9fcce541cda8edc219086cec5`. Its
`ggml/src/ggml-cpu/arch/arm/repack.cpp` is pinned by SHA-256
`6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d`;
the source binding is recorded in
[`data/upstream-source-binding.json`](data/upstream-source-binding.json).

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
  working set for a time budget or explicit iteration count. The raw result
  records requested/actual bytes, covered lines, warmup count, and a nonzero
  conditioning checksum.
- `cold_conditioned` writes and verifies every 64-byte line of a separate
  thrash buffer before the marker, then performs **exactly one** traversal.
  Any override to more than one iteration or a nonzero time budget is rejected.
  The thrash allocation must be an exact multiple of 64 bytes, so the verifier
  touches every requested byte and never silently drops a remainder.
  This is named conditioning, not proof that every architectural cache set
  missed.

All comparisons normalize to `ns/traversal` (or its inverse,
`traversals/s`). For example, 250 hot calls taking 250 ms and one cold call
taking 1 ms both equal 1 ms/traversal, so the cold/hot penalty is exactly 1.0.
Total windows with different call counts are never divided directly.

Cold penalties are compared like-for-like only: full cold/full hot,
packed-stream cold/packed-stream hot, and unpack-scale cold/unpack-scale hot at
the same CPU, carrier size, and block count. Cross-mode elapsed times are not
blindly subtracted. The analyzer validates every raw row, requires globally
unique run IDs, and admits exactly one hot plus one cold row for each pair ID.
Within every mode/shape/PMU cell, `pair_index` is contiguous from one and
`pair_order` alternates with its parity. It computes each pair's cold/hot
penalty first and then takes the median of those penalties; it never divides
independent hot and cold medians. A ratio-of-ratios may show whether the full-
kernel penalty tracks its controls, but it remains a directional inference.

## Qualification contract

The standalone harness emits `e055-q1-hot-cold-harness/v1`, explicitly marked
unqualified. A target runner must join it with E049c and produce strict
`e055-q1-hot-cold/v2`. Qualification requires:

- exact fd9 `S`, fd8 `ACK`, fd9 `E` synchronization;
- E049c v2 `sample_valid=true`, `event_source=armv8_pmuv3_raw_config`, one
  nonempty exact `core`, `cache`, or `memory` group, supported/valid events,
  exact name/config bindings from `tooling/a733_pmu_exec.c`, and runtime-float
  `running_ratio == 1.0` for every event;
- readable thermal telemetry with no trip and maximum temperature at or below
  its limit; Boolean temperatures are not numbers;
- affinity containing only CPU0 (A55) or CPU6 (A76), identical start/end CPU,
  and zero migrations, with every identifier/count an integer rather than a
  Boolean or floating-point lookalike;
- exact full cache-conditioning coverage metadata and checksum;
- `golden_pass=true`, exactly 18 golden cases, and a canonical SHA-256 binding
  of every joined field to the exact harness result;
- source, allowed O3/O3-LTO binary, compiler, immutable upstream commit/ref,
  and repack SHA-256 provenance loaded from the committed manifest,
  disassembly report, and source binding. Callers cannot supply substitute
  expected hashes.

The exact E049c event/config groups are:

| Group | Events and raw configs |
|---|---|
| `core` | `cpu_cycles=0x11`, `instructions=0x8`, `stall_backend=0x24` |
| `cache` | `l1d_cache_refill=0x3`, `l2d_cache_refill=0x17`, `l3d_cache_refill=0x2a` |
| `memory` | `mem_access=0x13`, `bus_access=0x19` |

`sync=false`, missing/empty PMU events, negative or Boolean counts, non-float or
non-finite running ratios, a thermal failure, CPU migration, an unverified hot
or cold conditioning state, malformed checksum, or unbound provenance fails
closed. PMU values are event counts, not bytes. E055 has no direct DDR-byte
counter and never relabels refill or access events as traffic.

## Planned board matrix

Working sets round upward to complete 208-byte carriers:

`64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB, 4 MiB, 12.5 MiB`.

Here `12.5 MiB` is binary: exactly `13,107,200` bytes. Working-set rounding
checks `target + 207` for unsigned overflow before division.

CPU0 and CPU6 are measured first. Every cell needs at least five alternating
hot/cold pairs in each E049c PMU group. Promotion unconditionally requires all
seven sizes × two CPUs × three modes × two cache states × three PMU groups:
126 paired phase cells and at least 1,260 qualified rows for five pairs. A
partial matrix raises an error rather than returning a preliminary promotion.
Only if the complete component model predicts at least a 4% Q1 gain may one
bounded all-core production-shape gate and a subsequent optimization be
recommended. The analyzer implements this as
`promotion.threshold_fraction = 0.04`; its projected gain is an optimistic
component bound from replacing full cold latency with full hot latency, not an
end-to-end model speedup claim. Failures are published as raw evidence rather
than converted to zero-valued samples.

## Publication provenance

A commit cannot contain its own SHA-1. E055 therefore uses the accepted E047
two-phase binding instead of writing a stale parent commit as if it identified
the publication:

1. With every non-self-referential artifact staged, branch preflight binds the
   exact stage-0 index tree while excluding only `branch-preflight.json` and
   `manifest.json`.
2. The manifest copies that base commit, tree binding, local ref, and upstream
   ref without claiming a future commit hash.
3. After commit and push, `python3 tooling/verify_e055_publication.py` verifies
   that committed `HEAD` has the staged tree, that the preflight base is its
   ancestor, every manifested file hash/size matches, and both local and
   remote-tracking refs equal the exact containing commit.

## Reproduction of revised Stage 1

Run the adversarial host, cross/QEMU, and disassembly gates:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e055_q1_hotcold tests.test_e055_harness_contract \
  tests.test_e055_aarch64_gate tests.test_e055_publication -v
```

The cross build deliberately compiles fixed-name objects before linking. A
one-shot driver invocation leaves a random temporary assembler-object name in
the ELF string table and therefore cannot support an exact reproducible binary
SHA-256. The O3 commands used by the test are:

```text
aarch64-linux-gnu-g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -march=armv8.2-a+dotprod -c tooling/e055_q1_hotcold.cpp -o e055_cpp.o
aarch64-linux-gnu-g++ -march=armv8.2-a+dotprod -c \
  experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S -o e055_asm.o
aarch64-linux-gnu-g++ -O3 e055_cpp.o e055_asm.o -o e055
```

The disassembly gate repeats the C++ compile and link with `-flto`, retaining
the fixed `e055_lto_cpp.o` and `e055_asm.o` basenames. Two clean build
directories must produce identical final hashes. Both binaries must
retain load/vector-consume controls without calls or SDOT and retain SDOT in
the stock full kernel.

QEMU must report `golden_pass=true` and `golden_cases=18`. QEMU results are
functional evidence only and are not used as A733 performance evidence. The
first uninitialized-LUT failure remains at
[`data/failure-qemu-uninitialized-lut.txt`](data/failure-qemu-uninitialized-lut.txt):
it was a harness initialization bug, was rejected, and must not be interpreted
as a hardware or mathematical result.

No board workload, model run, OPP/DDR change, NPU run, or full-model bottleneck
claim is part of this revised Stage 1.
