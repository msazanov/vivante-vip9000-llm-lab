# Task 5 report — Q1 CPU operator evidence

Implemented `q1-cpu-operator-baseline/v1` validation and deterministic summarization.

## Evidence

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v` initially failed because `tooling/summarize_q1_cpu_operator.py` was absent.
- Focused GREEN: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v` — `Ran 3 tests ... OK`.
- Schema check: `python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json` — `PASS q1-cpu-operator-baseline/v1`.
- Full suite: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v` — 196 tests, one pre-existing dirty-worktree failure in `test_q1_golden_summary` because `tooling/q1_vip_golden.py` does not match its recorded source hash. The Task 5 focused suite remains green.

The validator is fail-closed: it requires ordered identity/memory/warmup/measured/result records, exactly 50 measured samples, finite F32 artifacts, independent hashes and correctness thresholds before timing qualification, exact population statistics, pinned production tensor tuples, CPU_REPACK-only execution, zero expansion/fallback/swap/faults, null unmeasured DDR counters, and successful thermal-guard lifecycle sidecar evidence. `--check` validates every summary section and refuses mixed output/check arguments.
