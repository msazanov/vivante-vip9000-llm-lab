# VIP9000 Profiling Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a profiling-first, evidence-safe repository foundation for repeated Bonsai model inference optimization on A733/VIP9000.

**Architecture:** Keep proprietary host compiler and target runtime assets external while documenting exact identities and hashes. Wrap experiments in a standard-library profiler, store canonical JSON/JSONL data, and render one append-only Markdown row per model test with explicit speed and quality status.

**Tech Stack:** Markdown, Python 3 standard library, POSIX shell, Linux `/proc` and `/sys`, `unittest`, GitHub Git Data/Contents APIs.

## Global Constraints

- `vivante-vip9000-llm-lab` is the canonical clean-sheet repository; `orange-RAG` is evidence-only.
- Primary KPI is steady-state decode tokens per second; prefill throughput and TTFT remain separate metrics.
- No speed result is promoted without a pinned CPU reference and recorded quality status.
- Do not commit SDK archives, vendor binaries, generated NBG files, model weights, credentials, or private documentation.
- Missing optional Linux sensors must be recorded and must not fail the profiled command.
- The profiler requires `--run-id`; the output-directory basename must match it; optional `--phase-file` defaults to `phases.jsonl` and is passed as `VIP9000_PHASE_FILE`.
- Qualified results require software/workload/statistics objects, non-null headline metrics, positive repetitions, and an existing compatible qualified CPU reference or a self-referencing CPU baseline.
- The recorder accepts `--repo-root` (default `.`); qualified results require the raw JSON plus `metadata.json`, `stdout.log`, `stderr.log`, `telemetry.jsonl`, and the existing custom/default phase file named by `metadata.files.phases` beneath the resolved run directory. Metadata `run_id` must match, `exit_code` must be zero, and launch/profiler errors must be null. Other statuses may retain staged or missing raw files.
- Recorder updates use POSIX `flock` plus rollback/recovery; they are not a cross-filesystem atomic transaction.
- Inventory outputs publish only after success; mandatory failures propagate, and proprietary/license content is hashed or redacted.
- New executable behavior follows test-first red-green-refactor.

---

### Task 1: Record the approved design and implementation map

**Files:**
- Create: `docs/superpowers/specs/2026-08-07-profiling-foundation-design.md`
- Create: `docs/superpowers/plans/2026-08-07-profiling-foundation.md`

**Interfaces:**
- Consumes: the user-approved profiling-first architecture and repository boundary.
- Produces: exact constraints and file responsibilities used by Tasks 2–4.

- [ ] **Step 1: Write the approved design**

Record objective hierarchy, evidence classes, host/target/experiment layers, profiling contract, quality gates, model records, non-goals, and acceptance criteria.

- [ ] **Step 2: Write this execution plan**

Map each implementation file to one responsibility and include exact validation commands.

- [ ] **Step 3: Self-review spec coverage and placeholders**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('docs/superpowers/plans/2026-08-07-profiling-foundation.md')
s = Path('docs/superpowers/specs/2026-08-07-profiling-foundation-design.md')
for path in (p, s):
    text = path.read_text()
    assert ('T' + 'BD') not in text
    assert ('fill' + ' in') not in text.lower()
assert 'decode tokens per second' in s.read_text()
assert 'quality' in s.read_text().lower()
PY
```

Expected: exit 0 with no output.

### Task 2: Implement raw command and board telemetry profiling

**Files:**
- Create: `tests/test_profile_command.py`
- Create: `tooling/profile_command.py`

**Interfaces:**
- Consumes: CLI `--output-dir PATH --run-id ID [--interval-ms N] [--label TEXT] [--phase-file PATH] -- COMMAND [ARG ...]`; the output basename must equal `ID`.
- Produces: `metadata.json`, `telemetry.jsonl`, `stdout.log`, `stderr.log`, and `phases.jsonl` by default; passes the phase path to the child as `VIP9000_PHASE_FILE`, retains launch-failure metadata/raw files, maps signals to `128 + signal`, and cleans the child process group on collector failure.

- [ ] **Step 1: Write failing behavior tests**

Tests invoke the not-yet-present profiler with a successful Python child and a failing Python child. They assert captured output, metadata exit code, monotonic duration, telemetry presence, refusal to overwrite a result directory, and propagation of a non-zero child status.

- [ ] **Step 2: Run tests and observe the expected failure**

Run:

```bash
python3 -m unittest -v tests.test_profile_command
```

Expected: FAIL because `tooling/profile_command.py` does not exist.

- [ ] **Step 3: Implement the minimal profiler**

Use `argparse`, `subprocess.Popen`, `signal`, `time`, `json`, `pathlib`, and Linux text files. Sample process status/stat, memory information, CPU frequency policies, NPU devfreq, and thermal zones. Treat unreadable optional files as absent. Write metadata and raw files even when launch or collection fails; pass `VIP9000_PHASE_FILE` to the direct child.

- [ ] **Step 4: Run tests and confirm green**

Run:

```bash
python3 -m unittest -v tests.test_profile_command
```

Expected: all profiler tests pass.

### Task 3: Implement the canonical model result recorder

**Files:**
- Create: `tests/test_record_model_result.py`
- Create: `tooling/record_model_result.py`
- Create: `benchmarks/schema/model-run.example.json`

**Interfaces:**
- Consumes: `--input RESULT.json --ledger model-runs.jsonl --card MODEL.md [--repo-root PATH]` (`--repo-root` defaults to `.`).
- Produces: one canonical compact JSONL record and one Markdown table row between `MODEL_RESULTS_START` and `MODEL_RESULTS_END`; validates safe `benchmarks/results/<run_id>/*.json` raw paths, qualified headline metrics/reference rules, qualified raw evidence beneath `--repo-root`, software/workload/statistics objects, and rejects missing fields and duplicate `run_id`.

- [ ] **Step 1: Write failing recorder tests**

Use temporary cards and ledgers. Assert a valid result is appended once, speed and quality values appear in the card, a duplicate is rejected without changing either file, and an incomplete result is rejected.

- [ ] **Step 2: Run tests and observe the expected failure**

Run:

```bash
python3 -m unittest -v tests.test_record_model_result
```

Expected: FAIL because `tooling/record_model_result.py` does not exist.

- [ ] **Step 3: Implement validation and lock/recovery writes**

Require identity, configuration, performance, quality, status, software/workload/statistics, and raw-result fields. Permit `null` performance values for failed tests but require qualified headline metrics and repetitions to be non-null/positive. Require an existing compatible qualified CPU reference or a self-referencing CPU baseline. For qualified records, require raw JSON and `metadata.json`, `stdout.log`, `stderr.log`, `telemetry.jsonl`, plus the existing safe-relative phase file named in metadata; require matching `run_id`, zero `exit_code`, and null launch/profiler errors. Lock with POSIX `flock`, fsync sibling temporary files, replace the pair in sequence, and recover the ledger if card replacement fails; do not describe this as a cross-filesystem atomic transaction.

- [ ] **Step 4: Run tests and confirm green**

Run:

```bash
python3 -m unittest -v tests.test_record_model_result
```

Expected: all recorder tests pass.

### Task 4: Add the canonical documentation and model cards

**Files:**
- Create: `docs/hardware/orange-pi-zero-3w.md`
- Create: `docs/toolchain/inventory.md`
- Create: `docs/evidence/orange-rag-prior-tests.md`
- Create: `docs/profiling/profiling-contract.md`
- Create: `benchmarks/README.md`
- Create: `benchmarks/models/ternary-bonsai-27b.md`
- Create: `benchmarks/models/bonsai-27b.md`
- Create: `tooling/README.md`
- Create: `tooling/target_inventory.sh`
- Create: `tooling/inspect_acuitylite.sh`
- Create: `tests/test_inventory_scripts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: verified inventory from the target, local container inspection, prior local artifact hashes, and Task 2/3 CLIs.
- Produces: one navigable source of truth and exact read-only recipes for repeating discovery.

- [ ] **Step 1: Document facts with evidence labels**

Record target identity, CPU/RAM/kernel/NPU sysfs/runtime files, discovered host Docker images and AcuityLite 6.51 components, old `vpm_run`/MobileNet/LaBSE artifacts, contradictions, and all known hashes. Explicitly state what has not been executed under the new protocol.

- [ ] **Step 2: Document the benchmark and quality contract**

Define repetitions, statistics, decode/prefill separation, telemetry, profiler phases, CPU reference, quality gates, failure retention, and naming conventions.

- [ ] **Step 3: Write failing inventory-script behavior tests**

Run `python3 -m unittest -v tests.test_inventory_scripts` before the scripts exist. Expected: tests fail because the commands cannot be opened.

- [ ] **Step 4: Add model cards and hardened inventory recipes**

Create empty append-only result tables for both requested models. Add recipes
that write temporary inventory outputs and publish them only after success,
propagate mandatory failures, and keep optional sensors unavailable rather than
fatal. Harden the AcuityLite container with no network, read-only root,
dropped capabilities, `no-new-privileges`, PID/memory limits, and temporary
`/tmp`; hash and redact license text without copying vendor files into the
repository.

- [ ] **Step 5: Verify the inventory-script tests**

Run `python3 -m unittest -v tests.test_inventory_scripts`. Expected: all tests pass.

- [ ] **Step 6: Update the root repository map**

Link the profiling contract, tool inventory, model cards, and tooling entrypoint from `README.md`.

### Task 5: Verify and publish the isolated branch

**Files:** all files from Tasks 1–4.

**Interfaces:**
- Consumes: the complete staged change and fresh test output.
- Produces: branch `codex/profiling-foundation` and a draft pull request to `main`.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run static and documentation validation**

Run:

```bash
python3 -m py_compile tooling/profile_command.py tooling/record_model_result.py
bash -n tooling/target_inventory.sh tooling/inspect_acuitylite.sh
python3 - <<'PY'
from pathlib import Path
required = [
    'docs/profiling/profiling-contract.md',
    'docs/toolchain/inventory.md',
    'benchmarks/models/ternary-bonsai-27b.md',
    'benchmarks/models/bonsai-27b.md',
    'tooling/README.md',
]
for name in required:
    assert Path(name).is_file(), name
for name in required[2:4]:
    text = Path(name).read_text()
    assert '<!-- MODEL_RESULTS_START -->' in text
    assert '<!-- MODEL_RESULTS_END -->' in text
PY
```

Expected: every command exits 0 and all tests pass.

- [ ] **Step 3: Publish with the GitHub connector**

Create the isolated branch from the current `main`, upload the verified files, create a commit named `research: add profiling-first experiment foundation`, and open a draft pull request.

- [ ] **Step 4: Compare the remote branch with main**

Use the GitHub compare endpoint. Expected: only the approved documentation, tooling, tests, and root README changes appear.
