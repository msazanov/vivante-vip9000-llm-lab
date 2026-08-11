# Task 5 fix round 1 report

Fixed the Task 6/7 contract and fail-closed evidence checks.

- Canonical runner JSONL is now exactly `identity → memory → warmup → 50 measured → result`.
- Identity/executor and memory/fallback fields are top-level wire fields; result is exactly `schema_version`, `record`, `output_f32_bytes`.
- Summaries normalize record envelopes and round-trip through `--check` for both pinned gate and down tensor tuples.
- Thermal guard grammar is strict (`start`, `inventory`, `child_started`, contiguous `kind:sample`, `child_exit=0`, `exit=0`); telemetry and measured DDR counters are type/gate checked.
- Correctness emits and validates finite/cosine/max-error decisions and observed thresholds before performance qualification.
- Output publication uses same-directory fsynced temp + exclusive hard-link publication and cleans partial files.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
Ran 10 tests ... OK
python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json
PASS q1-cpu-operator-baseline/v1
```
