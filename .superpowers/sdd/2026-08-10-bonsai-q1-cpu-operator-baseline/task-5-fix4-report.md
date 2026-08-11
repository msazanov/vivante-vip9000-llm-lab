# Task 5 fix round 4 report

Aligned the summarizer with the real thermal guard emitter and closed the
remaining review findings F7--F10.

Implemented checks include:

- concise `rc=2` handling for invalid UTF-8 in `--check` summaries;
- absolute, lexically normalized thermal paths and canonical decimal
  `st_dev:st_ino` identities, with uniqueness across all inventory categories;
- emitter-compatible signed integer telemetry for temperatures, CPU frequency,
  and cooling states, retaining only strict types and the real temperature
  ceiling invariant;
- non-empty executable validation for `start.command` while allowing empty
  later argv arguments and rejecting non-string argv members;
- adversarial tests for path/identity forgery, cross-category collisions,
  non-canonical identities, malformed argv, invalid UTF-8, and emitter-valid
  signed/range values.

Evidence:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
Ran 19 tests ... OK
python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json
PASS q1-cpu-operator-baseline/v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_thermal_exec_guard -v
Ran 11 tests ... OK
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tooling/summarize_q1_cpu_operator.py tests/test_summarize_q1_cpu_operator.py
git diff --check
```
