# Canonical Repository Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise, English-only canonical layer that orients new agents to the A733/VIP9000 objective, evidence boundaries, experiment status, provenance, and next bottleneck without moving or rewriting legacy evidence.

**Architecture:** Human-facing canonical Markdown lives under `docs/` and links to immutable experiment/evidence paths or exact remote branch/commit references. Machine-readable experiment rows live in `docs/experiments/registry.json`; a standard-library checker validates the registry, provenance, branch inventory, hardware fact classes, English-only Markdown, and relative links. Raw historical documents remain in their existing paths and are not copied into the canonical layer.

**Tech Stack:** Markdown, JSON, Python 3 standard library, pytest, Git remote-ref metadata.

**Spec:** User request for `orange-RAG-16p.17`, with the approved repository-structure audit and foundation anchor `c071476773ad0f7fc499b6a39270a98bc1e25878`.

## Global Constraints

- All new or rewritten canonical Markdown, comments, test names, and commits use English prose.
- The root README must state the `>1 tok/s` objective, 6xA55 + 2xA76 topology, 32-bit memory interface, LPDDR5-4800 / 19.2 GB/s theoretical ceiling only, controller readback 510 MHz, unknown effective rate and sustained bandwidth, and the secure-firmware/safety overclock boundary.
- Qualified full-model best is E035 at exactly `0.972497 tok/s`; E044 is never current best.
- E049d-v2 `1.11698354 tok/s` is diagnostic three-token marker throughput, not end-to-end or an optimization claim.
- Registry rows retain accepted, rejected, failed, diagnostic, unqualified, and planned statuses and exact branch/commit/path provenance.
- Legacy raw evidence remains immutable and is referenced by pointer or exact ref; no proprietary payloads, weights, NBGs, or raw external traces are added.
- Do not modify `codex/e055-q1-hot-cold` or its worktree.

---

### Task 1: Add failing canonical-index tests

**Files:**
- Create: `tests/test_repository_index.py`
- Test: `tests/test_repository_index.py`

**Interfaces:**
- Consumes: `tooling/check_repository_index.py` functions and canonical paths introduced by later tasks.
- Produces: deterministic tests for registry statuses/current-best/provenance, English-only canonical Markdown, links, hardware evidence classes, branch inventory, and no legacy-payload duplication.

- [ ] **Step 1: Write the failing tests**

  Cover: canonical file set, required objective phrases, allowed status set, unique IDs, E035 exact best, E044 non-best, E049d-v2 diagnostic-only metadata, complete branch SHA inventory, evidence-class labels, safe relative links, and no Cyrillic in canonical Markdown.

- [ ] **Step 2: Run the focused tests to verify the expected failure**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_repository_index.py`

  Expected: collection or assertion failures because the checker and canonical files do not yet exist.

### Task 2: Implement the deterministic repository-index checker

**Files:**
- Create: `tooling/check_repository_index.py`
- Modify: `tests/test_repository_index.py`

**Interfaces:**
- Consumes: repository root, `docs/experiments/registry.json`, `docs/experiments/schema.json`, `docs/experiments/branch-inventory.json`, and canonical Markdown paths.
- Produces: `check_registry`, `check_english_only`, `check_links`, `check_hardware_facts`, `check_branch_inventory`, `check_provenance`, `run_checks`, and a CLI with exit code 0 on pass and 1 on validation failure.

- [ ] **Step 1: Add the minimal JSON/schema/provenance validators needed by the failing tests**

- [ ] **Step 2: Run the focused tests to verify the implementation is still red for missing docs/data**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_repository_index.py`

- [ ] **Step 3: Add deterministic Markdown link and English-only scanners**

- [ ] **Step 4: Run the focused tests and record the remaining missing-canonical-file failures**

### Task 3: Publish canonical orientation and hardware/profiling contract

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Create: `docs/objective.md`
- Create: `docs/hardware/a733.md`
- Modify: `docs/hardware/orange-pi-zero-3w.md`
- Modify: `docs/profiling/profiling-contract.md`
- Create: `docs/profiling/provenance-and-gates.md`

**Interfaces:**
- Consumes: target audit evidence under `docs/evidence/` and the profiling tool behavior already implemented on the foundation branch.
- Produces: concise canonical orientation, explicit hardware fact classes, memory-rate distinctions, secure-firmware boundary, and promotion gates.

- [ ] **Step 1: Add the failing content assertions to the focused tests**

- [ ] **Step 2: Run the assertions and confirm they fail against the old orientation**

- [ ] **Step 3: Replace the root orientation and add the objective/hardware/provenance docs**

- [ ] **Step 4: Rewrite Russian headings/sections in the canonical profiling and target hardware docs in English while retaining legacy evidence links**

- [ ] **Step 5: Run `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_repository_index.py` and confirm the documentation checks pass**

### Task 4: Publish architecture and NPU canonical guidance

**Files:**
- Create: `docs/architecture/backend-plan.md`
- Create: `docs/npu/vip9000-capabilities.md`
- Create: `docs/npu/q1-and-partitioning.md`

**Interfaces:**
- Consumes: existing foundation evidence links for VIPLite, EVIS/Q1, native-NN capability, and CPU baselines.
- Produces: bottleneck model, backend sequencing, NPU capability classes, Q1 layout/math guardrails, and prefill/decode partition policy.

- [ ] **Step 1: Add failing assertions for the current bottleneck, no-DDR-bytes rule, native-NN gate, and Q1 no-expansion rule**

- [ ] **Step 2: Run the focused tests to observe the expected missing-document failures**

- [ ] **Step 3: Add concise English architecture/NPU docs with exact evidence pointers**

- [ ] **Step 4: Run the focused tests and checker CLI**

### Task 5: Publish experiment registry, schema, current best, branch inventory, and migration map

**Files:**
- Create: `docs/experiments/schema.json`
- Create: `docs/experiments/registry.json`
- Create: `docs/experiments/current-best.md`
- Create: `docs/experiments/branch-inventory.json`
- Create: `docs/experiments/branch-inventory.md`
- Create: `docs/experiments/migration-map.md`

**Interfaces:**
- Consumes: exact remote refs fetched from `origin`, including `codex/profiling-foundation`, `codex/e023-ggml-q1-seam`, `codex/e049c-arm-pmu`, and `codex/e055-q1-hot-cold`.
- Produces: rows covering E001–E055 evidence known from the retained refs, explicit status taxonomy, E035 current-best pointer, diagnostic E049d-v2 pointer, and no-duplication migration guidance.

- [ ] **Step 1: Add failing tests for registry row coverage and exact key result rows**

- [ ] **Step 2: Run the tests to confirm the registry is absent**

- [ ] **Step 3: Add the schema and registry rows with exact branch/commit/path pointers**

- [ ] **Step 4: Add branch inventory and migration map without copying raw branch payloads**

- [ ] **Step 5: Run the checker and focused tests**

### Task 6: Independent review and verification

**Files:**
- Modify: only files identified by review findings.

- [ ] **Step 1: Run the focused repository-index tests and checker CLI from a clean process**

- [ ] **Step 2: Run the existing full suite and record the three unrelated baseline failures separately**

- [ ] **Step 3: Inspect `git diff --check`, changed-file list, canonical-language scan, and raw-file duplication guard**

- [ ] **Step 4: Perform an independent self-review against every global constraint and fix any gap**

- [ ] **Step 5: Commit the canonical layer, push `codex/repository-index`, and capture exact commit/test evidence**
