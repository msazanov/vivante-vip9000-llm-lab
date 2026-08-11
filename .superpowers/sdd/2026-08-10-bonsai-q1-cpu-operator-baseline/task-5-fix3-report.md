# Task 5 fix round 3 report

Authenticated nested thermal evidence and lifecycle timing, while rejecting duplicate JSON members.

Implemented checks include:

- exact nested zone/policy/cooling/process schemas from `tooling/thermal_exec_guard.py`;
- non-empty inventory/sample arrays, unique inventory path/identity, exact sample cardinality/order and static correlation;
- process PID correlation with `child_started`, numeric ranges, guard-limit temperature rejection;
- nondecreasing lifecycle timestamps, deadline spacing, first-deadline ordering, and lateness equality;
- duplicate-key rejection for runner JSONL, telemetry JSON, and summary `--check` JSON.

Evidence:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
Ran 16 tests ... OK
python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json
PASS q1-cpu-operator-baseline/v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_thermal_exec_guard -v
Ran 11 tests ... OK
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tooling/summarize_q1_cpu_operator.py tests/test_summarize_q1_cpu_operator.py
git diff --check
```
