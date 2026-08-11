# Task 5 fix round 2 report

Hardened the Q1 operator evidence parser against forged thermal lifecycle and malformed JSON shapes.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
Ran 13 tests ... OK
python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json
PASS q1-cpu-operator-baseline/v1
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tooling/summarize_q1_cpu_operator.py tests/test_summarize_q1_cpu_operator.py
git diff --check
```

Thermal records now use the exact fields emitted by `tooling/thermal_exec_guard.py`, with strict integer/boolean checks and contiguous lifecycle/sample ordering. Runner and summary malformed objects are normalized to concise exit-2 errors. Summary telemetry validates strict integer swap delta and strict booleans for every fault/thermal flag. Existing wire-format, gate/down round-trips, packed-byte, DDR, correctness, atomic-output, and p90 tests remain green.
