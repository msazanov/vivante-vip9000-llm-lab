# Recorder hardening report

## Scope

Hardened `tooling/record_model_result.py` and its focused tests in
`tests/test_record_model_result.py`. No schema, documentation, cards, or other
tooling files were changed.

## RED evidence

Command:

```text
python3 -m unittest tests.test_record_model_result -v
```

The newly added regression tests failed before the implementation changes:

- card replacement fault left the newly appended ledger bytes in place;
- same ledger/card path was not rejected;
- qualified records with null speed values or zero repetitions were accepted;
- missing, nonexistent, and mismatched references were accepted;
- unsafe raw-result components and a wrong run directory were accepted.

The RED run completed with 15 tests and 16 failures, matching the missing
behaviors rather than test-collection or syntax errors.

## GREEN evidence

Commands:

```text
python3 -m unittest tests.test_record_model_result -v
python3 -m unittest discover -s tests -v
python3 -m py_compile tooling/record_model_result.py
```

Results:

- Focused recorder suite: 16 tests, 0 failures.
- Full repository suite: 32 tests, 0 failures.
- Python compilation: exit 0.

## Implemented contract

- Exclusive flock on a stable ledger-sibling lock file covers ledger/card
  reads, validation, and both replacements; canonical paths resolving to the
  same file are rejected.
- Both temporary artifacts are written and fsynced. The ledger is replaced
  first; a card replacement failure restores the original ledger (or removes a
  newly created one), fsyncs the parent, and cleans temporary/backup files.
- Qualified records require measured performance, positive repetitions,
  meaningful quality fields, and `passed=true`. External references must be
  existing qualified CPU records with matching model SHA and context/batch/
  ubatch; a first CPU reference may self-reference only from a CPU partition.
- Raw-result paths are staged-safe POSIX paths under
  `benchmarks/results/<run_id>/`, with safe components (including explicit
  rejection of `.` and `..`) and a `.json` filename; local existence is
  intentionally not required.
- Failed, unqualified, and rejected records continue to accept nullable
  performance values.

## Concerns

The lock and publication protocol are POSIX-oriented (`fcntl.flock`) as
required. The recorder intentionally does not verify that staged raw-result
files already exist, because connector/staged publication is part of the
contract.
