# Result schema hardening report

## Scope

Hardened `tooling/record_model_result.py`, its focused tests, and the synthetic
schema example. The recorder now requires typed `software`, `workload`, and
`performance.statistics` objects for every status, with stricter provenance,
determinism, statistics, and CPU-reference matching rules for `qualified`.

## RED evidence

After updating `valid_result` and adding focused contract tests, the focused
command was:

```text
python3 -m unittest tests.test_record_model_result -v
Ran 21 tests in 13.670s
FAILED (failures=18)
```

The failures covered unresolved/invalid provenance, missing or nondeterministic
workload data, missing/inconsistent statistics, missing required objects on
nonqualified records, and mismatched reference workload/runtime.

## GREEN evidence

Focused recorder verification:

```text
python3 -m unittest tests.test_record_model_result -v
Ran 22 tests in 14.196s
OK
```

Full repository verification:

```text
python3 -m unittest discover -s tests -v
Ran 38 tests ...
OK
```

Additional checks passed:

```text
python3 -m py_compile tooling/record_model_result.py
```

The synthetic example also passes direct recorder validation. Its software
provenance is explicitly `unresolved`, its workload remains typed, and its
nullable statistics retain empty sample lists for the unqualified status.

## Implemented contract

- Requires all eight software strings and all seven typed workload fields.
- Requires all three statistic objects and all summary/sample keys for every
  status; only summary values may be null, while samples are finite,
  nonnegative numbers.
- For `qualified`, rejects unresolved provenance, invalid repository commits,
  incomplete/non-deterministic workloads, inconsistent ordered summaries,
  repetition/sample-count mismatches, and top-level/median disagreements.
- CPU references must additionally match tokenizer, prompt suite, token counts,
  seed, deterministic setting, runtime, and runtime commit.

## Concerns

The recorder intentionally keeps the model-card rendering compact; complete
provenance and statistics remain in the canonical JSONL record and staged raw
artifacts.

## Qualified raw-evidence gate

The follow-up TDD cycle added `--repo-root` (default `.`) and qualified-only
evidence checks. Before implementation, the focused run showed the expected
unrecognized CLI option and missing validator behavior:

```text
python3 -m unittest tests.test_record_model_result -v
Ran 27 tests ...
FAILED (failures=52, errors=1)
```

After the implementation, focused verification passed:

```text
python3 -m unittest tests.test_record_model_result -v
Ran 27 tests ...
OK
```

Qualified records now require the raw JSON plus `metadata.json`, `stdout.log`,
`stderr.log`, and `telemetry.jsonl` below the resolved repository root.
Metadata must identify the same run, report exit code zero, have null launch and
profiler errors, and point `metadata.files.phases` to an existing safe relative
path within the run directory. Nested/custom phase paths are supported.
