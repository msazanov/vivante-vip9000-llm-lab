# E055 — cache-hot/cold Q1 microbenchmark

Status: **BOUNDED DIRECT TARGET MICROGATE PASS — 20/20 QUALIFIED; FULL-MATRIX PROMOTION REJECTED**.

E055 is a bounded experiment designed to distinguish cache/data-carrier cost
from unpack/compute cost in the stock Q1_0 4×4 kernel on A733. It is not a
Bonsai-27B run and it provides no tokens/s result. A full Bonsai `n_predict=32`
run remains prohibited until the complete promotion matrix passes.

## Qualified no-TTY target microgate

The corrected fresh phase is published under
[`results/e055a-direct-native-rerun2-o3-core-20260814t173636z/`](results/e055a-direct-native-rerun2-o3-core-20260814t173636z/).
It used the exact native PMU and harness artifacts, direct `/usr/bin/ssh -T` and
`/usr/bin/scp`, and live validation after every copied sample. All 20 samples
qualified; requested CPU0/CPU6 affinity matched the harness's post-measurement
`sched_getcpu()` observation. Maximum temperature was 39.618 °C.

The bounded result does not authorize promotion. It covers only 20 of the required
1,260 rows and one of 126 phase cells. Its meaningful repeated-pair chart and exact
limits are documented in the result README. No model or NPU workload ran.

One preceding PTY-contaminated fresh phase was stopped after one sample because
wrapper stderr contained a single newline. It is preserved unchanged under
[`results/e055a-direct-native-rerun-o3-core-20260814t172604z/`](results/e055a-direct-native-rerun-o3-core-20260814t172604z/).

## First direct target phase: invalid, fully preserved

The first 20-sample attempt used direct system `/usr/bin/ssh` and
`/usr/bin/scp` only. It ran CPU0 and CPU6, five alternating hot/cold pairs,
64 KiB, `full_dotprod`, and the `core` PMU group. No model or NPU workload ran.
All raw files, ABI-compatible native build evidence, failures, analysis, and a
meaningful paired chart are published under
[`results/e055a-direct-native-o3-core-20260814t161316z/`](results/e055a-direct-native-o3-core-20260814t161316z/).

The attempt is **not a qualified 20-run phase**. `cpu6-pair1-hot` and
`cpu6-pair3-hot` violate `T + 1 ms >= H`, so the exact classification is 18
qualified raw samples and two invalid raw samples. The source-grounded timing
investigation is in
[`TIMING_ROOT_CAUSE.md`](results/e055a-direct-native-o3-core-20260814t161316z/TIMING_ROOT_CAUSE.md).
No result from this phase supports promotion or an optimization claim.

The previously developed helper/transport adapter is rejected historical
pre-target evidence and was not used for the direct phase. Project policy is
to use the installed OpenSSH executables directly; the custom adapter must not
be invoked for target work.

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
The rejected `7679828` caller-created-row qualifier is preserved in
[`data/review-rejection-stage4-7679828.md`](data/review-rejection-stage4-7679828.md).
The rejected `f8acab9` noncanonical protocol/timing qualifier is preserved in
[`data/review-rejection-stage5-f8acab9.md`](data/review-rejection-stage5-f8acab9.md).
The rejected `eca4b03` transport-provenance revision is preserved in
[`data/review-rejection-stage6-eca4b03.md`](data/review-rejection-stage6-eca4b03.md).
The rejected `6f7a6cb` process-ownership and SFTP-provenance revision is
preserved in
[`data/review-rejection-stage7-6f7a6cb.md`](data/review-rejection-stage7-6f7a6cb.md).
Rejected revisions remain evidence of failed approaches; none is target data.

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

- `hot_repeat` performs exactly 16 warmup calls, then repeatedly traverses the same carrier
  working set for a time budget or explicit iteration count. The raw result
  records requested/actual bytes, covered lines, warmup count, and a nonzero
  conditioning checksum.
- `cold_conditioned` writes and verifies every 64-byte line of one exact
  67,108,864-byte thrash buffer before the marker, then performs **exactly one** traversal.
  It must report exactly 1,048,576 touched lines and checksum
  `0x8d3ea13d15850279`.
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

## Git-sealed raw qualification contract

The standalone harness emits `e055-q1-hot-cold-harness/v1`, explicitly marked
unqualified. The analyzer does not accept a caller-created joined sample
mapping, even if all fields and an unkeyed JSON hash are internally
consistent. Its only input is the path to a committed
`e055-raw-bundle/v2` manifest under this experiment's `raw/<phase>/`
directory. The loader derives every sample from these exact roles:

- harness stdout and stderr stream envelopes;
- E049c raw JSON and stderr stream envelope;
- runner argv, minimal environment, exit, affinity, migration, and provenance;
- one exact transport-evidence object duplicated in the bundle and every
  successful runner record;
- the exact committed O3 or O3-LTO executable.

Each non-manifest file has a declared SHA-256, byte size, and Git blob object
ID. The loader requires unique canonical relative paths and roles, regular
single-link files, no symlinks, no hardlink aliases, no duplicate blob reuse,
bounded sizes, and no undeclared files in the phase. It then verifies that the
worktree bytes, stage-0 index, and `HEAD` blob are identical. The manifest is
not self-referential: its containing commit, tree, and own blob ID are derived
from Git and attached to every derived row. Git proves byte immutability, not
that a device produced those bytes; this limitation is deliberate and must be
reported with every result.

Target measurements use the installed `/usr/bin/ssh` and `/usr/bin/scp`
directly. The rejected custom SSH transport is historical pre-target evidence
and must not be invoked. Credentials stay outside the repository and are never
copied into raw evidence. The passing phase used `ssh -T`, strict external
known-host checking, an external key, and an echo-disabled local terminal only
to carry interactive sudo input; the board received no SSH TTY. Each target
file was checked by direct stat/SHA-256 and copied before live validation. The
published raw manifest binds the copied local bytes; it is not a cryptographic
device attestation and does not claim a sealed per-file remote stat receipt.

Raw qualification requires:

- exact fd9 `S`, fd8 `ACK`, fd9 `E` synchronization;
- E049c v2 `sample_valid=true`, `event_source=armv8_pmuv3_raw_config`, one
  nonempty exact `core`, `cache`, or `memory` group, supported/valid events,
  exact name/config bindings from `tooling/a733_pmu_exec.c`, and runtime-float
  `running_ratio == 1.0` for every event;
- the exact launcher sequence `<content-addressed-pmu-path> -o
  <phase-addressed-remote-e049c-path> --child-stdout
  <remote-run>/target-harness.stdout.raw --child-stderr
  <remote-run>/target-harness.stderr.raw --event-group <group>
  --min-running-ratio 0.95 --start-on-ready
  --sync-timeout-ms 5000 --max-temp-c 85 -- <exact-harness-argv>`; the 0.95
  launcher floor never relaxes the accepted runtime ratio of exactly 1.0;
- readable thermal telemetry with no trip and maximum temperature at or below
  the exact 85 °C limit; Boolean temperatures are not numbers;
- affinity containing only CPU0 (A55) or CPU6 (A76), identical start/end CPU,
  and zero migrations, with every identifier/count an integer rather than a
  Boolean or floating-point lookalike;
- exact full cache-conditioning coverage metadata and checksum;
- positive equal `pid` and process-group IDs;
- `golden_pass=true`, exactly 18 golden cases, a nonzero output checksum, and
  the exact harness stdout artifact cross-bound by the runner's raw-artifact
  hashes;
- harness source, allowed O3/O3-LTO binary and compiler, E049c source,
  reproducible E049c AArch64 binary and compiler, immutable upstream
  commit/ref, and repack SHA-256 provenance loaded from the committed
  publication manifest, disassembly/build reports, source binding, and
  committed executables. The E049c artifact is built twice in unrelated clean
directories from one fixed object with no linker build ID; both SHA-256
  values must equal the committed artifact, and its exact byte size is part of
  the runtime contract.
  Callers cannot supply substitute expected hashes. The bundle and runner bind
  the canonical SHA-256 of the complete `runtime_qualification` object, not the
  mutable publication-manifest bytes. This avoids a hash cycle when the
  manifest's staged-tree binding is regenerated to include a new raw phase.

The exact E049c event/config groups are:

| Group | Events and raw configs |
|---|---|
| `core` | `cpu_cycles=0x11`, `instructions=0x8`, `stall_backend=0x24` |
| `cache` | `l1d_cache_refill=0x3`, `l2d_cache_refill=0x17`, `l3d_cache_refill=0x2a` |
| `memory` | `mem_access=0x13`, `bus_access=0x19` |

`sync=false`, nonempty stderr, missing/empty PMU events, wrong raw event config,
negative or Boolean counts, non-float or
non-finite running ratios, a thermal failure, CPU migration, an unverified hot
or cold conditioning state, malformed checksum, or unbound provenance fails
closed. PMU values are event counts, not bytes. E055 has no direct DDR-byte
counter and never relabels refill or access events as traffic.

The wrapper reserves the two child capture files before pipes, perf setup, or
fork using exclusive no-follow opens. A collision therefore cannot start the
workload. Child stdout/stderr are redirected only after `setsid`; wrapper
stderr remains a distinct stream. Fixed synchronization descriptors 8 and 9
cannot alias capture descriptors, and only one final successful invocation may
populate each reserved local role.

E049c starts the PMU group immediately before ACK and stops it immediately
after the `E` marker, but its clocks are not interchangeable. Let `H` be the
child harness `CLOCK_MONOTONIC_RAW` elapsed time, `M` the parent
`CLOCK_MONOTONIC` measured interval, and `T/R` an event's perf
`time_enabled_ns/time_running_ns`. Qualification requires positive uint64
`H`, `M`, `T`, and `R`, `M + 1 ms >= H`, `T + 1 ms >= H`, `R == T`, and the
runtime-float ratio exactly 1.0. The 1 ms term is a one-sided cross-clock
tolerance. There is deliberately no upper bound on `M - H` or `T - M`:
parent descheduling and perf accounting may make them large. Events in one
group are not required to report identical `T` unless E049c source semantics
prove that invariant. These are qualifier checks, not conversions to bytes.

The publication-only `sample.schema.json` describes analyzer-derived output
and carries `x-e055-analyzer-input=false`. Its evidence object includes the
containing commit/tree, manifest path/blob, publication identity, and all five
raw paths/hashes/sizes/blob IDs. It is never an alternate analyzer input.

## Planned board matrix

Working sets round upward to complete 208-byte carriers:

`64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB, 4 MiB, 12.5 MiB`.

Here `12.5 MiB` is binary: exactly `13,107,200` bytes. Working-set rounding
checks `target + 207` for unsigned overflow before division. Raw ingestion
accepts only these seven canonical target values, enforces the C++ native-block
allocation ceiling, and bounds every emitted checksum as canonical nonzero
uint64 hexadecimal.

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
   `manifest.json`. Every raw phase path remains included; excluding raw
   evidence would weaken provenance and is forbidden.
2. The manifest copies that base commit, tree binding, local ref, and upstream
   ref without claiming a future commit hash.
3. After commit and push, `python3 tooling/verify_e055_publication.py` verifies
   that committed `HEAD` has the staged tree, that the preflight base is its
   ancestor, every manifested file hash/size matches, and both local and
   remote-tracking refs equal the exact containing commit and the complete
   index/worktree has no modified, staged, or untracked file.

The verifier also requires the excluded preflight and manifest worktree/index
bytes to equal their committed `HEAD` blobs. It computes the committed-tree
binding with its own fixed list of exactly those two paths and rejects any
caller-declared raw, source, or other extra exclusion. This closes the
self-reference gap without writing a stale parent SHA into either file. Three deterministic
AArch64 executables are committed under `artifacts/` (two harness builds and
the E049c wrapper); their SHA-256 values,
source/compiler provenance, and disassembly checks are publication-bound.
`infer_bottleneck()` invokes this global verifier unconditionally before it
loads the phase-scoped sealed bundle. A locally sealed phase cannot bypass a
stale ref, mismatched publication tree, or unrelated dirty replacement file.
The raw bundle's runtime-contract digest is stable across regeneration of the
publication timestamp and tree-binding envelope, but changes if any runtime
qualification field changes.
The global `*.bin` ignore remains in force except for the two exact reviewed
phase executable names
`experiments/E055-q1-hot-cold/raw/<phase>/artifacts/harness-O3.bin` and
`harness-O3-flto.bin`; no other raw or repository binary is made eligible.

`tooling/e055_capture_scaffold.py` is reservation-only. It creates a new phase
and every planned output with `O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`, checks
regular-file and single-link invariants, records the original device/inode and
role size bound, and closes all descriptors on success or partial failure. If
any reservation or post-open validation fails, it unlinks only pathnames that
still resolve to inodes created by that invocation. Every created directory is
recorded with its owner, device, and inode; rollback removes it only when the
current path has that same identity and owner and remains empty. Pre-existing,
moved-original, or concurrently replaced data is not removed.
`populate_reserved()` reopens one original inode without following links,
takes a nonblocking exclusive lock, requires a zero-length placeholder, writes
and synchronizes one bounded nonempty payload, and refuses a second population
or a replaced inode. A producer such as E049c that creates its own output with
`O_EXCL` must not receive a placeholder path directly; a future runner must
capture its output separately and populate the reserved role through this
primitive. The CLI reservation report contains paths only and is not a
standalone population token. The scaffold contains no process, remote-login,
harness, or board execution path.

`tooling/e055_target_executor.py` composes those primitives but remains
disabled unless a reviewed transport object and independent expected pins are
explicitly injected. The narrow transport adapter invokes exactly
`/usr/bin/ssh` and `/usr/bin/sftp`; it does not implement SSH. Fake executable
paths exist only through an explicit test-only constructor and cannot qualify
operational evidence. Before any transport preparation, the executor checks
the actual known-hosts, SSH, SFTP, and helper bytes against independent pins
and verifies the helper's clean HEAD/index blob. Its only plan is 20 O3 runs:
CPU0 then CPU6,
pair indexes 1–5, odd pairs hot→cold, even pairs cold→hot, 64 KiB,
`full_dotprod`, and the `core` PMU group. It reserves the complete local phase
before transport preparation, deploys only the publication-bound harness and
E049c bytes through the injected interface, records child stdout/stderr
separately from wrapper stderr, and always invokes transport restoration. If
execution and restoration both fail, both exceptions are retained in order
rather than letting restoration hide the primary failure. A failed run leaves
the bundle manifest empty, preserves every
available captured byte plus a failure runner record, and stops before the
next run. A successful uncommitted bundle is still not evidence: all files
must be staged, committed, pushed, publication-verified, and reloaded by the
sealed analyzer.

Before the first capture, the executor derives a phase-unique deployment root
from the phase ID plus the exact harness and PMU SHA-256 values. Both executable
paths include their content hashes. It passes a fixed target-concurrency lock
path to the transport and requires two independent API results: first an
exclusive-deployment receipt, then a fresh readback/stat/hash proof. The caller
creates a cryptographically random 256-bit request ID and nonce for each API
step. All four tokens are distinct, each response must echo the exact requested
tokens, and the request/response sequence is sealed into the bundle and every
runner record. This makes copied or cached prior observations fail the live
request check. Each result must report the exact role, path, SHA-256, byte size,
mode, device and inode for both executables, and both observations must agree
exactly. A future real transport must perform a new remote stat/read/hash for
every readback request and must never return a cached observation. The endpoint
identity must also agree. The adapter requires exact caller-provided board,
host-key, known-hosts, endpoint, helper, SSH, and SFTP pins; unpinned
`operationally_trusted` evidence is rejected.

Before executing E049c, the fixed helper starts a fixed Python gate in a new
session. The gate writes and synchronizes an exclusive `owned-process.json`
record containing the owner, deployment device/inode, run ID, PID, PGID,
session ID, and `/proc` start time before it calls `execve` on the exact argv.
If the request helper is killed, a later restore accepts only that exact record,
rechecks the deployment and process identities, signals matching session
members through pidfds, and requires quiescence before deleting deployment
state. A missing, replaced, reused, malformed, or non-quiescent identity fails
closed without signaling an unproven process.

Helper bootstrap creates the random staging directory in a separate strict
SFTP operation. Cleanup removes it only after that operation succeeded and
therefore proved ownership; a collision is never removed. Final helper removal
rechecks the file device and inode after hashing, and the adapter requires the
returned identity to match the deploy/readback runtime identity.

These records are replay-resistant transport evidence, not cryptographic device
attestation. An injected transport and its remote endpoint can lie consistently,
so both remain inside the operational trust boundary. Git later makes the
reported bytes and records immutable; it does not turn them into independently
attested device truth.

The E049c wrapper itself ignores parent `SIGPIPE`, converts a closed ACK channel
to a recorded failure, and handles `TERM`, `HUP`, and `INT` by terminating and
reaping the complete child tree before returning the signal-derived status. The
child sets a Linux parent-death signal as defense in depth. Before forking, the
parent records the existing Linux child-subreaper state and enables subreaping;
it restores the prior state after cleanup. Descendants that escape the original
session or process group with `setsid` or `setpgid` are therefore reparented to
the wrapper when their intermediate parent exits.

Cleanup repeatedly enumerates only tasks whose `/proc` PPID is the wrapper,
opens a pidfd, re-reads the strict `/proc/<pid>/stat` identity through field 22
(`start_time`), and signals only an unchanged pidfd-bound identity. It sends
`TERM`, then bounded `KILL`, reaps with `waitpid`, and requires two empty scans
before reporting quiescence. A malformed record, unsupported pidfd operation,
enumeration error, or timeout fails the sample closed. This avoids signaling an
unrelated process after numeric PID reuse. A nonzero or signaled child after
`E` also fails the sample, and both immediate and S/A/E exit-zero paths use the
same tree-quiescence gate before success. The launcher is a dedicated process;
the PPID scope assumes it has no unrelated pre-existing children.

## Reproduction of revised Stage 1

Run the adversarial host, cross/QEMU, and disassembly gates:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e055_q1_hotcold tests.test_e055_harness_contract \
  tests.test_e055_raw_bundle tests.test_e055_sealed_promotion \
  tests.test_e055_capture_scaffold tests.test_e055_target_executor \
  tests.test_e055_aarch64_gate \
  tests.test_e055_publication -v
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

The host cross-build gate independently builds E049c twice from the fixed
object name `a733_pmu_exec.o` with `aarch64-linux-gnu-gcc -O2` and
`-Wl,--build-id=none`; its clean-directory hashes must match each other. This
is reproducible functional/QEMU evidence, not the A733 runtime binary. The
committed `a733-pmu-exec-aarch64` artifact is the separate byte-identical
two-directory native GCC 12 build recorded in `data/pmu-build-review.json`.
That native artifact avoids the rejected host/target glibc ABI drift. Its
source hash, compiler hash and ID, object hash, binary hash, interpreter,
NEEDED libraries, and glibc requirements are runtime provenance.

QEMU must report `golden_pass=true` and `golden_cases=18`. QEMU results are
functional evidence only and are not used as A733 performance evidence. The
first uninitialized-LUT failure remains at
[`data/failure-qemu-uninitialized-lut.txt`](data/failure-qemu-uninitialized-lut.txt):
it was a harness initialization bug, was rejected, and must not be interpreted
as a hardware or mathematical result.

No model run, OPP/DDR change, NPU run, or full-model bottleneck claim is part of
E055. The first hardware attempt was limited to CPU0 and CPU6, 64 KiB,
`full_dotprod`, the `core` PMU group, and five alternating hot/cold pairs. It is
published as an invalid 18+2 phase rather than relabeled as a success.
