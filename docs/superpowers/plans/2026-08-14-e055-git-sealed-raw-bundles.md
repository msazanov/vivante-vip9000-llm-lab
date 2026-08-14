# E055 Git-Sealed Raw Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace self-declared E055 sample mappings with a Git-sealed,
raw-artifact-only qualification and promotion path.

**Architecture:** A new loader verifies a committed bundle manifest and every
referenced blob against `HEAD`, the index, the worktree, role hashes, and exact
raw schemas. It derives private qualified rows that the existing paired
component model consumes. A separate scaffold reserves future capture outputs
exclusively but never executes a workload in this revision.

**Tech Stack:** Python 3 standard library, Git plumbing commands, JSON Schema
documents, AArch64 C++/assembly harness, `unittest`, QEMU user mode.

**Spec:** `docs/superpowers/specs/2026-08-14-e055-git-sealed-raw-bundles-design.md`

## Global Constraints

- All code, comments, documentation, commit messages, and reports are English.
- No target-board or model workload is executed before fourth independent acceptance.
- Only personal/sensitive data is prohibited; all captured artifacts still receive a secret scan.
- PMU `running_ratio` must be a runtime float exactly `1.0`; E049c's launcher floor `0.95` is not acceptance.
- Promotion requires all 1,260 raw runs in the documented matrix and a predicted Q1 gain of at least 4%.
- The bundle manifest must remain below 16 MiB at the full 1,260-run shape.

---

### Task 1: Preserve the rejection and define raw schemas

**Files:**
- Create: `experiments/E055-q1-hot-cold/data/review-rejection-stage4-7679828.md`
- Create: `experiments/E055-q1-hot-cold/data/raw-bundle.schema.json`
- Create: `experiments/E055-q1-hot-cold/data/stream-capture.schema.json`
- Create: `experiments/E055-q1-hot-cold/data/runner-capture.schema.json`
- Modify: `tests/test_e055_harness_contract.py`

**Interfaces:**
- Produces schemas `e055-raw-bundle/v1`, `e055-stream-capture/v1`, and
  `e055-runner-capture/v1` for later parsers.

- [ ] **Step 1: Write failing schema and rejection-evidence tests**

```python
def test_7679828_rejection_and_raw_schemas_are_preserved(self):
    rejection = DATA / "review-rejection-stage4-7679828.md"
    self.assertIn("REJECTED EVIDENCE", rejection.read_text())
    bundle = json.loads((DATA / "raw-bundle.schema.json").read_text())
    self.assertEqual(bundle["properties"]["schema"]["const"], "e055-raw-bundle/v1")
    self.assertEqual(bundle["properties"]["runs"]["minItems"], 1260)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_harness_contract.E055HarnessContractTest.test_7679828_rejection_and_raw_schemas_are_preserved -v`

Expected: FAIL because the four evidence/schema files do not exist.

- [ ] **Step 3: Add exact schemas and rejection evidence**

The bundle schema requires `phase_id`, `qualification`, `build_artifacts`, and
1,260 `runs`. Artifact objects require `role`, canonical `path`, `sha256`,
`size_bytes`, and `git_blob_oid`. Stream envelopes require exact payload
hash/size/Base64. Runner metadata requires exact argv, safe environment,
affinity, provenance, exit, and raw role hashes. Every object sets
`additionalProperties: false`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/E055-q1-hot-cold/data tests/test_e055_harness_contract.py
git commit -m "test: define E055 sealed raw schemas"
```

### Task 2: Implement the Git-sealed artifact loader

**Files:**
- Create: `tooling/e055_raw_bundle.py`
- Create: `tests/test_e055_raw_bundle.py`
- Generate: `experiments/E055-q1-hot-cold/artifacts/e055-O3-aarch64`
- Generate: `experiments/E055-q1-hot-cold/artifacts/e055-O3-flto-aarch64`

**Interfaces:**
- Produces `load_sealed_bundle(manifest_path: str | Path) -> SealedBundle`.
- Produces frozen `SealedArtifact`, `DerivedSample`, and `SealedBundle` data classes.
- `load_sealed_bundle` has no caller-supplied expected-provenance argument.

- [ ] **Step 1: Add the immutable fixture builder and filesystem/Git RED tests**

```python
def test_loader_rejects_dirty_staged_untracked_and_uncommitted_manifest(self):
    bundle = self.fixture.committed_minimal_bundle()
    for mutation in self.fixture.git_state_mutations(bundle):
        with self.subTest(mutation=mutation.name):
            mutation.apply()
            with self.assertRaisesRegex(ValueError, "sealed|HEAD|index|worktree"):
                load_sealed_bundle(bundle.manifest)

def test_loader_rejects_path_alias_symlink_hardlink_duplicate_and_oversize(self):
    for forged in self.fixture.path_and_inode_forgeries():
        with self.subTest(forged=forged.name), self.assertRaises(ValueError):
            load_sealed_bundle(forged.manifest)
```

The fixture uses a temporary Git repository, copies the exact published
harness executable, commits every bundle file, and calls the production public
loader. It does not inject an expected hash mapping.

- [ ] **Step 2: Run the loader tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_raw_bundle.E055SealedArtifactTest -v`

Expected: ERROR importing `tooling.e055_raw_bundle`.

- [ ] **Step 3: Generate reproducible published harness binaries**

Use fixed object basenames exactly as recorded in
`data/disassembly-review.json`, write the O3 and O3-LTO linked artifacts to the
paths above, and verify their SHA-256 values are respectively
`9c3ea8fed87fec1f3f8cf84d6a941a135daf276174e4f52c78174c66cfe600f9`
and `884069f79c782ef2196e5462c73448f178a22686472b68abd100730a13987f97`.

- [ ] **Step 4: Implement canonical path and Git blob sealing**

```python
def load_sealed_bundle(manifest_path: str | Path) -> SealedBundle:
    manifest = _require_committed_regular_file(Path(manifest_path), MAX_MANIFEST_BYTES)
    parsed = _parse_exact_json(manifest.bytes, "e055-raw-bundle/v1")
    artifacts = tuple(_seal_declared_artifact(item, manifest.repo) for item in _entries(parsed))
    _reject_aliases_duplicates_and_unmanifested_files(manifest, artifacts)
    return _parse_and_derive(manifest, parsed, artifacts)
```

Use `lstat`, `st_nlink == 1`, role-specific size limits, `git rev-parse HEAD`,
`git rev-parse HEAD^{tree}`, `git ls-tree`, `git ls-files --stage`,
`git status --porcelain=v2`, and `git cat-file blob`. Require worktree,
index, declared SHA/size/blob, and committed blob bytes to agree.

- [ ] **Step 5: Run loader tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tooling/e055_raw_bundle.py tests/test_e055_raw_bundle.py \
  experiments/E055-q1-hot-cold/artifacts
git commit -m "feat: seal E055 raw artifacts to Git"
```

### Task 3: Parse raw captures and derive qualified samples

**Files:**
- Modify: `tooling/e055_raw_bundle.py`
- Modify: `tests/test_e055_raw_bundle.py`

**Interfaces:**
- `SealedBundle.samples` contains only loader-created frozen `DerivedSample` values.
- Produces `canonical_harness_argv(cell, executable_path, iterations) -> tuple[str, ...]`.

- [ ] **Step 1: Add PMU/thermal/timing/hash/swap RED tests**

```python
def test_recomputed_public_hash_cannot_hide_raw_pmu_thermal_or_timing_mutation(self):
    for field in ("events.0.value", "thermal.max_observed_c", "measured_elapsed_ns"):
        forged = self.fixture.mutate_public_row_and_rehash_only(field)
        with self.subTest(field=field), self.assertRaises(ValueError):
            load_sealed_bundle(forged.manifest)

def test_raw_files_cannot_be_swapped_between_pairs_or_builds(self):
    for forged in self.fixture.swapped_role_bundles():
        with self.subTest(forged=forged.name), self.assertRaises(ValueError):
            load_sealed_bundle(forged.manifest)
```

Add separate cases for truncate, append, duplicate role, unknown environment,
secret-shaped environment, Boolean CPU/count/thermal, golden other than 18,
sync other than exact start/ack/end, event name/config mismatch, and stderr
payload that is not empty.

- [ ] **Step 2: Add and verify the exact ratio RED test**

```python
def test_launcher_floor_does_not_relax_qualification_ratio(self):
    raw = self.fixture.valid_run()
    raw.e049c["min_running_ratio"] = 0.95
    raw.e049c["events"][0]["running_ratio"] = 0.95
    with self.assertRaisesRegex(ValueError, "exactly 1.0"):
        load_sealed_bundle(raw.commit())
```

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_raw_bundle.E055RawParserTest -v`

Expected: FAIL because sealed bytes are not parsed/cross-checked yet.

- [ ] **Step 3: Implement exact stream, runner, harness, and E049c parsers**

Decode every stream envelope and recompute payload hash/size. Parse exactly one
stdout JSON object, require empty successful stderr streams, validate exact
runner keys/environment/argv/affinity/provenance, and validate all E049c
timing/config/count/status/sample-valid/sync/thermal/exit fields. Cross-check
all run/pair/cell/build identities and role hashes before constructing a frozen
derived sample with containing commit/tree/manifest/raw evidence.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tooling/e055_raw_bundle.py tests/test_e055_raw_bundle.py
git commit -m "feat: derive E055 samples from sealed raw captures"
```

### Task 4: Make promotion raw-bundle-only

**Files:**
- Modify: `tooling/e055_q1_hotcold.py`
- Modify: `tests/test_e055_q1_hotcold.py`
- Modify: `tests/test_e055_raw_bundle.py`

**Interfaces:**
- Changes `infer_bottleneck(source)` so `source` is a committed bundle manifest
  path or a sequence of committed bundle manifest paths, never mappings.
- Keeps `ns_per_traversal`, `cache_penalty_ratio`, and
  `paired_speed_delta` as non-promotion mathematical helpers.

- [ ] **Step 1: Add fabricated-matrix and full-bundle RED tests**

```python
def test_fabricated_1260_row_mapping_is_never_promotion_evidence(self):
    with self.assertRaisesRegex(TypeError, "committed raw-bundle manifest"):
        infer_bottleneck(qualified_matrix())

def test_full_sealed_matrix_preserves_pair_median_and_four_percent_gate(self):
    result = infer_bottleneck(self.fixture.committed_full_matrix().manifest)
    self.assertTrue(result["matrix_complete"])
    self.assertEqual(result["matrix_qualified_rows"], 1260)
    self.assertTrue(result["promotion"]["eligible"])
```

- [ ] **Step 2: Add the 16 MiB scalability RED test**

```python
def test_full_1260_run_manifest_stays_below_publication_cap(self):
    built = self.fixture.render_full_matrix_manifest()
    self.assertEqual(len(built["runs"]), 1260)
    self.assertLess(len(canonical_json(built)), 16 * 1024 * 1024)
```

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_q1_hotcold tests.test_e055_raw_bundle.E055PromotionTest -v`

Expected: FAIL because the analyzer still accepts mappings and cannot consume a
sealed manifest.

- [ ] **Step 3: Isolate the existing component model behind private derived rows**

Move current strict pair/matrix/statistics code into
`_infer_derived_samples(samples: Sequence[DerivedSample])`. The public function
loads every manifest through `load_sealed_bundle`, verifies one publication
identity and non-overlapping run IDs, and passes only loader-produced frozen
rows to the private function. Remove `validate_sample` as an evidence API or
make it unconditionally reject mappings with a migration error.

- [ ] **Step 4: Run promotion tests and verify GREEN**

Run the Step 2 command. Expected: PASS, including ratio-of-medians adversarial
data and the exact 4% threshold.

- [ ] **Step 5: Commit**

```bash
git add tooling/e055_q1_hotcold.py tests/test_e055_q1_hotcold.py \
  tests/test_e055_raw_bundle.py
git commit -m "fix: require sealed raw bundles for E055 promotion"
```

### Task 5: Add the non-executing exclusive capture scaffold

**Files:**
- Create: `tooling/e055_capture_scaffold.py`
- Create: `tests/test_e055_capture_scaffold.py`

**Interfaces:**
- Produces `reserve_phase(phase_dir: Path, planned_runs: Sequence[RunPlan]) -> tuple[Path, ...]`.
- The CLI accepts `--phase-dir` and `--plan-json`; it reserves files only and
  prints `target_workload_executed=false`.

- [ ] **Step 1: Add O_EXCL/symlink/partial-failure RED tests**

```python
def test_reservation_refuses_existing_file_or_symlink_before_execution(self):
    for collision in ("regular", "symlink"):
        with self.subTest(collision=collision), self.assertRaises(FileExistsError):
            reserve_phase(self.fixture.with_collision(collision), self.plans)
        self.assertFalse(self.fixture.workload_marker.exists())
```

- [ ] **Step 2: Run the scaffold tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_capture_scaffold -v`

Expected: ERROR importing `tooling.e055_capture_scaffold`.

- [ ] **Step 3: Implement reservation-only behavior**

Use `os.open` with `os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
os.O_CLOEXEC`, immediately `fstat` for regular file and link count one, and
close all reserved descriptors on success or failure. Validate canonical run
IDs and refuse a pre-existing phase directory. Do not call `subprocess`, SSH,
or the harness.

- [ ] **Step 4: Run scaffold tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tooling/e055_capture_scaffold.py tests/test_e055_capture_scaffold.py
git commit -m "feat: reserve E055 capture files exclusively"
```

### Task 6: Publish, document, and verify revised Stage 1

**Files:**
- Modify: `experiments/E055-q1-hot-cold/README.md`
- Modify: `experiments/E055-q1-hot-cold/data/sample.schema.json`
- Modify: `experiments/E055-q1-hot-cold/data/disassembly-review.json`
- Modify: `tooling/generate_e055_manifest.py`
- Modify: `tooling/verify_e055_publication.py`
- Modify: `tests/test_e055_publication.py`
- Modify: `tests/test_e055_aarch64_gate.py`
- Regenerate: `experiments/E055-q1-hot-cold/data/branch-preflight.json`
- Regenerate: `experiments/E055-q1-hot-cold/data/manifest.json`

**Interfaces:**
- Publication manifest binds the new schemas, loader, scaffold, tests,
  rejection evidence, and published deterministic binaries.
- `sample.schema.json` becomes a derived-output schema whose evidence section
  requires containing commit/tree/manifest/raw paths and hashes; it is not an
  accepted analyzer input schema.

- [ ] **Step 1: Add publication RED tests**

Require every new file in `PUBLISHED`, exact binary hashes, exact schema roles,
the rejection evidence, `target_workload_executed=false`, and documentation of
the first accepted hardware phase: CPU0+CPU6, 64 KiB, `full_dotprod`, `core`,
five alternating pairs only.

- [ ] **Step 2: Run publication tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_e055_publication tests.test_e055_harness_contract -v`

Expected: FAIL because publication metadata does not include the raw-bundle subsystem.

- [ ] **Step 3: Update English documentation and publication generators**

Document the Git trust boundary, raw terminology, capture procedure, exact
ratio distinction, no-target status, and fourth-review gate. Regenerate the
all-local/all-remote preflight only after all non-self-referential files are
staged, then regenerate the manifest using the E047 staged-tree pattern.

- [ ] **Step 4: Run the complete fresh verification suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e055_q1_hotcold tests.test_e055_raw_bundle \
  tests.test_e055_capture_scaffold tests.test_e055_harness_contract \
  tests.test_e055_aarch64_gate tests.test_e055_publication -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e053_branch_preflight tests.test_e053_experiment_validator \
  tests.test_a733_pmu_exec tests.test_e049d_pmu_analyze \
  tests.test_e049d_steady_pmu -v
```

Also run two clean O3/O3-LTO builds, QEMU 18-case golden, disassembly checks,
manifest hash/size verification, `git diff --check`, English scan, secret scan,
and the post-push publication verifier.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "fix: require Git-sealed evidence for E055"
git push origin codex/e055-q1-hot-cold
python3 tooling/verify_e055_publication.py
```

Expected: local branch, `HEAD`, and remote-tracking ref are identical; verifier
status is `PASS`; no target/model workload exists in the manifest.
