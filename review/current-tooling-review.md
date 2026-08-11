# Current Tooling Review

## Scope and verification

Reviewed the approved design and implementation plan, `tooling/profile_command.py`,
`tooling/record_model_result.py`, both inventory scripts, and all files under
`tests/`.

Fresh verification completed:

- `python3 -m unittest discover -s tests -v`: PASS, 9 tests.
- `python3 -m py_compile tooling/profile_command.py tooling/record_model_result.py`: PASS.
- `bash -n tooling/target_inventory.sh tooling/inspect_acuitylite.sh`: PASS.

Passing the current tests does not establish the full contract; the tests cover
only the happy path and a small number of validation cases.

## Findings

### Critical

1. **The ledger/card update is not atomic and is race-prone.**
   `tooling/record_model_result.py:222-257` reads both files, checks for a duplicate,
   writes two PID-named temporary files, then replaces the ledger before replacing
   the card. A card replacement failure leaves the canonical JSONL and Markdown
   index inconsistent; concurrent recorders can both pass the duplicate check and
   overwrite each other's records. This violates the append/consistency contract
   and can lose results. Add an inter-process lock covering read/validate/write,
   reject identical ledger/card paths, fsync temporary files, and use a recoverable
   transaction/journal or a single canonical commit mechanism that either updates
   both artifacts or restores the old pair after a second-replace failure.

2. **The AcuityLite “isolated” container can be de-isolated by a user-supplied image
   value.**
   `tooling/inspect_acuitylite.sh:8-10,47-53` passes `--image` directly into the
   Docker option stream. For example, `--image --privileged` is parsed as a Docker
   option and the following literal `bash` becomes the image, defeating the stated
   isolation assumptions. Validate image references (reject option-prefixed and
   shell/control-character values) before invoking Docker, and add defense-in-depth
   flags such as capability dropping/no-new-privileges and resource limits.

### Important

3. **`qualified` does not actually prove a qualified speed result.**
   `tooling/record_model_result.py:107-144` permits `decode_tps`, `prompt_tps`,
   TTFT, and RSS to be `null`, permits zero repetitions, and accepts any non-empty
   `quality.reference_run_id` without checking that it identifies an existing,
   pinned deterministic CPU reference. Thus a record with no measured speed and a
   fabricated reference ID can be marked `qualified`. Require non-null primary
   decode data and at least one measured repetition for `qualified`, require the
   quality method/metric/reference fields to be meaningful, and resolve the
   reference to a validated CPU reference record before allowing promotion.

4. **The recorder accepts materially incomplete profiling records and leaves metric
   definitions ambiguous.**
   `tooling/record_model_result.py:52-67,93-120` requires only five performance
   fields and six basic configuration fields. It does not require tokenizer,
   compiler/SDK/driver/kernel/command identifiers, warm-up count, token counts,
   median/p10/p90/min/max, dispersion/CV, or the other contract fields in the
   approved design. The card rendering at `180-204` consequently presents a few
   numbers without enough context to reproduce or interpret them. Extend the
   schema/validation (with explicit nullable rules for failed runs) and render or
   link the complete statistics.

5. **The profiler does not assign a unique run ID or implement the required deep
   lifecycle telemetry.**
   `tooling/profile_command.py:170-237` records a caller-selected directory and
   command, but no generated `run_id`, raw-result identity, warm-up/repetition
   statistics, or model-level prompt/decode/TTFT values. `snapshot()` at
   `131-142` samples shallow process and board state only; there is no interface or
   phase record for cold initialization, graph compile/load, allocation/import,
   invocation, copies/flush/invalidate, synchronization, or teardown. Add a
   generated immutable run ID and a phase/event interface (or explicit phase
   manifest) and make model result production consume those records.

6. **Profiler failure paths can leave a child running without final metadata.**
   `tooling/profile_command.py:188-208` has no `try/finally` around telemetry
   sampling and file writes. A disk-full/write error or unexpected `/proc`/`/sys`
   collector exception can terminate the profiler while the child remains alive,
   leaving only partial raw files and no `metadata.json`. Isolate optional sensor
   failures as explicit missing samples, catch telemetry I/O failures, terminate
   and wait for the child, and attempt to write a failure metadata record before
   returning non-zero.

7. **Inventory command failures are masked as success.**
   `tooling/inspect_acuitylite.sh:47-91` runs Docker without `set -e`/status
   checking, then always executes the final `printf`, so a missing Docker binary,
   failed image, or failed extraction returns zero and leaves a partial output
   file. `tooling/target_inventory.sh:38-46,133-163` similarly has no global error
   propagation; output-directory/hash write failures can be followed by a zero
   exit. Use `set -euo pipefail`, preserve deliberate optional-probe handling
   locally, write to a temporary output, and publish it only after the required
   command succeeds.

8. **The AcuityLite inventory writes proprietary license text into the purportedly
   sanitized artifact.**
   `tooling/inspect_acuitylite.sh:70-77` runs `cat` on the extracted
   `licence.txt`, while the design explicitly keeps vendor/private material out of
   the repository. Record only a hash and a narrowly parsed license-target marker
   (with redaction), and add a test that representative license text cannot appear
   in the generated inventory.

9. **Raw-result validation is syntactic rather than evidentiary and can produce
   malformed Markdown links.**
   `tooling/record_model_result.py:146-177` rejects absolute paths and `..`, but
   does not require the referenced raw result to exist or contain an immutable run
   directory. It also permits control characters, brackets, parentheses, and other
   Markdown/link syntax in a path; `raw_link()` inserts those characters without
   escaping. A record can therefore claim a raw link that is missing or renders to
   a different destination. Resolve against the repository root, require the
   expected raw-result layout/files for non-failed results, restrict path
   components to a safe character set, and escape/encode the Markdown destination.

### Minor

10. **Tests do not guard the central negative cases.**
    `tests/test_profile_command.py:34-97`,
    `tests/test_record_model_result.py:83-157`, and
    `tests/test_inventory_scripts.py:12-72` omit launch/signal/telemetry-I/O
    failures, qualified-null/zero-repetition/fake-reference records, missing raw
    paths and Markdown escaping, pair-replacement failure/concurrency, Docker
    non-zero propagation, and license redaction. Add focused regression tests for
    those cases before relying on the current green suite.

## Contract verdicts

**Spec compliance: FAIL (partial foundation only).** The implementation does
preserve successful and failed child stdout/stderr, refuses an already-existing
profiler output directory, keeps decode/prompt/TTFT as separate named fields, and
has basic quality/status checks. It does not enforce a valid qualified speed result
against a pinned CPU reference, the complete profiling contract/statistics, deep
phase telemetry, safe evidence links, reliable command exit propagation, or
consistent JSONL/Markdown publication.

**Code quality: NEEDS WORK.** The Python is small, readable, standard-library-only,
and the happy-path quoting/non-overwrite tests are useful. The two-file write race,
Docker option injection, masked failures, and permissive evidence validation are
high-impact correctness and safety defects that should be fixed before this tooling
is used to publish benchmark claims.

## Findings count

Critical: 2  
Important: 7  
Minor: 1  
Total: 10

