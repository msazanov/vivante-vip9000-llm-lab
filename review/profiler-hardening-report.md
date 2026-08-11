# Profiler hardening review

## Scope

Hardened `tooling/profile_command.py` and its focused regression tests in
`tests/test_profile_command.py`. No benchmark repetition/statistics behavior was
added to the generic wrapper.

## RED evidence

After adding the run-ID, phase-file, missing-launch, signal, and collector-failure
tests, the focused suite failed because the profiler did not recognize
`--run-id`, did not create phase artifacts, did not retain raw child status, and
did not clean up a child after a snapshot exception.

## GREEN evidence

The focused suite passes:

```text
python3 -m unittest -v tests.test_profile_command
Ran 9 tests ... OK
```

The full repository suite passes:

```text
python3 -m unittest discover -s tests -v
Ran 32 tests ... OK
```

## Implemented contract

- Requires and validates `--run-id`, stores it in metadata, and requires the
  output directory basename to match it.
- Creates an append-only byte phase sink, passes its resolved path as
  `VIP9000_PHASE_FILE`, rejects absolute/parent-traversing paths, and records
  the relative path and byte size without parsing or fabricating events.
- Starts the child in a new session and terminates its process group with a
  bounded TERM→KILL cleanup path on profiler/collector failures.
- Preserves launch failures as return code 127 with metadata and all raw files.
- Maps signal exits to `128 + signal` while retaining the raw negative child
  return code in `child_return_code`.
- Isolates optional sensor-read failures as missing values.

## Residual concern

The profiler attempts atomic metadata publication after cleanup, but no program
can guarantee `metadata.json` creation when the output filesystem is unavailable
or has become unwritable.
