# E055 Git-Sealed Raw Bundles Design

Status: approved design for a Stage 1 revision. No target or model workload is
part of this change.

## Problem and evidence boundary

Commit `7679828c8d0891a8d42374279e096b09db87e0f5` is rejected evidence. Its
analyzer accepted caller-created Python mappings whose canonical JSON hashes
could be recomputed after changing a measurement. That checked internal
consistency, not provenance.

The revised analyzer accepts only a path to a raw-bundle manifest committed in
the current Git `HEAD`. Git is the immutability boundary: it proves which exact
bytes were reviewed and analyzed, but it does not by itself prove that a device
produced truthful data. The capture protocol, exact executable, E049c marker
protocol, thermal gate, and later human review remain responsible for that
claim. Derived sample mappings and unkeyed hashes are never accepted as
promotion input.

## Bundle layout

One phase uses one directory below
`experiments/E055-q1-hot-cold/raw/<phase-id>/`:

```text
bundle.json
artifacts/harness-O3.bin
runs/<run-id>/harness.stdout.capture.json
runs/<run-id>/harness.stderr.capture.json
runs/<run-id>/e049c.json
runs/<run-id>/e049c.stderr.capture.json
runs/<run-id>/runner.json
```

`bundle.json` has schema `e055-raw-bundle/v1`. It contains the phase ID,
qualification/publication hashes, one shared executable artifact per build,
and one run entry per measurement. Each run entry binds its run ID, pair ID,
pair index/order, exact cell coordinates, build name, and five distinct raw
artifact roles. Every non-manifest artifact entry records a canonical
repository-relative path, SHA-256, byte size, and Git blob object ID.

Stdout and stderr are stored in schema `e055-stream-capture/v1` envelopes. An
envelope records the run ID, producer, stream name, encoding, exact payload
size, exact payload SHA-256, and Base64 payload. This preserves empty stderr
exactly while making the committed capture blob specific to one run and role.
The analyzer decodes and rehashes the payload before parsing it. A shared
executable is listed once at bundle scope; runs refer to its build name rather
than reusing its path as a run role.

`runner.json` has schema `e055-runner-capture/v1`. It records the exact harness
argv, a minimal explicitly non-secret environment, process exit status,
affinity, endpoint CPUs, migration count, build name, and exact
source/binary/compiler/upstream/publication hashes. The only permitted
environment keys are `LC_ALL`, `LANG`, and `E055_BUILD_NAME`, with fixed safe
values; unknown keys and secret-shaped keys are rejected. It also records the
SHA-256 of the other four run artifacts so swapping roles or pairs breaks a
cross-binding even before statistical analysis.

## Filesystem and Git sealing

The loader derives the containing commit and tree from current `HEAD`; neither
is trusted from caller data. `bundle.json` and every referenced artifact must:

- be a unique canonical relative path beneath the repository and phase root;
- contain no absolute path, `..`, empty component, control character, or path
  normalization alias;
- be a regular file according to `lstat`, never a symlink, and have link count
  one so hardlink aliases are rejected;
- have a unique role/path/blob binding, except that each shared executable is
  intentionally declared once and referenced by build name;
- remain within role-specific limits: 16 MiB manifest, 256 KiB stdout capture,
  1 MiB stderr capture, 256 KiB E049c JSON, 256 KiB runner metadata, and
  32 MiB executable;
- match the manifest SHA-256 and size, the declared Git blob OID, the blob in
  `HEAD`, the stage-0 index, and the current worktree bytes.

The phase directory may contain no unmanifested files. Any staged change,
worktree change, deletion, untracked replacement, appended/truncated file, or
manifest not present in `HEAD` fails closed. The manifest cannot contain its
own commit or blob without self-reference; instead, the loader derives and
publishes the manifest blob, containing commit, and tree after verifying the
committed bytes. The publication verifier independently binds the analyzer,
schemas, runner scaffold, and manifest format to the qualification commit.

## Parsing and cross-validation

The analyzer parses only bytes returned by the sealed-artifact loader. It
requires exactly one JSON object in E055 stdout, no non-whitespace successful
stderr, exact E049c v2 JSON, no non-whitespace E049c stderr, and exact runner
metadata. It cross-checks:

- run, pair, order, cell, mode, cache state, target/actual size, blocks, CPU,
  build, and command identity across manifest, runner, stdout, and E049c;
- runner argv against a canonical argument sequence for that cell and E049c's
  recorded command array;
- runner environment against the fixed non-secret allowlist;
- zero process/E049c exit, exact stdout/stderr artifact hashes, one pinned CPU,
  equal start/end CPU, and zero migrations;
- the exact 18-case golden, conditioning strategy/coverage/checksum, calls,
  iterations, and elapsed time from raw harness stdout;
- E049c status/sample validity, exact `S/ACK/E`, measured elapsed time, thermal
  pass, event group size, exact event names/configs/count semantics, integer
  counts, enabled/running times, and float running ratio exactly 1.0;
- source, executable, compiler, upstream commit/ref/repack, PMU source, and
  publication artifact hashes against the committed E055 qualification
  artifacts.

Only after every cross-check passes does the loader construct a private derived
sample. The public `infer_bottleneck` entry point accepts committed manifest
paths only. The existing normalization, exact-pair, complete
7-size × 2-CPU × 3-mode × 2-state × 3-PMU-group matrix, median-of-pair
penalties, and 4% promotion threshold operate only on those private rows.

## Future runner scaffold

Stage 1 adds a capture-reservation scaffold, not a board executor. It creates a
new phase directory and every planned output with
`O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC`, checks regular-file/link-count
invariants, and refuses existing paths or symlinks before any workload could
start. It emits only fixed schemas, canonical paths, and the minimal safe
environment. Actual command execution remains disabled until independent
acceptance.

## Tests and acceptance

Adversarial tests first demonstrate that the rejected analyzer accepts a
fabricated mapping. New RED gates then require rejection when PMU, thermal, or
timing data is changed and a public derived hash is recomputed; when a complete
1,260-row mapping has no raw bundle; when raw roles are swapped between pairs
or builds; and when any artifact is truncated, appended, duplicated, linked,
untracked, dirty, staged, oversized, or path-traversing. Positive gates use a
temporary Git repository containing committed fixtures. Existing Q1 golden,
QEMU, O3/O3-LTO reproducibility/disassembly, PMU, publication, normalization,
pairing, full-matrix, and 4% threshold gates remain.

The first hardware phase after a fourth independent acceptance is restricted
to CPU0 and CPU6, 64 KiB, `full_dotprod`, the `core` PMU group, and five
alternating hot/cold pairs. It is not executed by this revision.
