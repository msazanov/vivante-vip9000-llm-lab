# Task 6 fix1 report — hardened Prism CPU_REPACK runner

## Delivered

- Added strict `--run-id` validation and preserved the Task 5 wire shape; the
  self-test gets a deterministic per-process ID while normal mode requires an
  explicit safe ID.
- Replaced whole-text fixture matching with strict structured parsing of
  `q1-cpu-operator-fixture/v1`. Pinned model metadata, tensor type/shape/size,
  activation metadata, canonical filenames, and payload SHA-256 values are
  validated before identity emission. The model bytes are not present in the
  fixture and are therefore checked as pinned metadata rather than hashed.
- Made phase rows profiler-compatible and durable: each runner row has exactly
  `event`, nonnegative `CLOCK_MONOTONIC` `monotonic_ns`, and sequential `step`;
  each append is written and `fdatasync`-ed. Compute samples still use
  `CLOCK_MONOTONIC_RAW`.
- Added RAII ownership for backend, contexts, and buffers, released the
  canonical input vector before warmup, and retained one CPU_REPACK tensor set
  with ordinary CPU activation/output buffers.
- Added exclusive same-directory temporary outputs, fsync, link-based
  no-overwrite publication, rollback/temporary cleanup, and support for bare
  relative output filenames.
- Native CMake mode now verifies `CMAKE_HOME_DIRECTORY` and `ggml_SOURCE_DIR`
  bind to `PRISM_SOURCE`, in addition to the exact pinned Prism SHA.
- The executable behavior suite now fails on runtime errors when qualification
  environment variables are present, tests both literal and normal production
  paths, checks Task 5 parsing and phase consumption, and verifies rerun and
  invalid-fixture cleanup behavior.

## Verification

Host x86 source mode (compile/help only):

```text
/usr/bin/cmake -S experiments/E004-q1-cpu-operator-baseline \
  -B /tmp/q1-cpu-operator-build-final \
  -DPRISM_SOURCE=/home/random/src/llama-prismml \
  -DBUILD_PINNED_PRISM_FROM_SOURCE=ON -DGGML_CCACHE=OFF
/usr/bin/cmake --build /tmp/q1-cpu-operator-build-final --parallel 1
/tmp/q1-cpu-operator-build-final/q1_cpu_operator_runner --help
```

Configure reported the exact pinned SHA
`38c66ad0241da4f9fcce541cda8edc219086cec5`; the binary built successfully and
help lists `--run-id`. With qualification variables unset:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_cpu_operator_runner -v
Ran 2 tests in 0.001s
OK (skipped=2)
```

`py_compile` of the runner test and summarizers passed, and `git diff --check`
was clean. The native CMake mismatch guard was also exercised with a build
whose `CMAKE_HOME_DIRECTORY` pointed at the runner project; configure rejected
it with the expected source/build mismatch error.

Target A733 accepted build:

```text
source: /home/orangepi/vip9000-lab/src/llama-prismml-shallow-38c66
build:  /home/orangepi/vip9000-lab/build/cpu-38c66-native-git
HEAD:   38c66ad0241da4f9fcce541cda8edc219086cec5
```

After rebuilding the runner against that native build, the complete qualified
suite passed with no skips:

```text
Ran 2 tests in 1.440s
OK
```

Fresh normal-mode production gate evidence (50 measured iterations) reported:

```text
RUN_ID= bonsai-q1-final-a55-002
BUFFER= CPU_REPACK KERNEL= q1_0_4x4_q8_0 STRICT= True
RECORDS= 54 SAMPLES= 51
PHASES= 10 [setup_begin, repack_begin, repack_end, warmup_begin, warmup_end,
            compute_begin, compute_end, output_read_begin, output_read_end,
            teardown]
OUTPUT_BYTES= 69632 Q8_BYTES= 5440
```

The literal K=128, M=16 test independently checked all 16 expected F32 values
`[-56,-48,-40,-32,-24,-16,-8,0,8,16,24,32,40,48,56,64]`, exact phase fields
and order, and `_consume_phases` acceptance with profiler lifecycle rows.

## Limitations

The host x86 build is intentionally compile/help-only; qualified Q1_0
CPU_REPACK runtime evidence requires the accepted A733 native build and pinned
production fixture. No fan, governor, sudo, or system-policy operation was
used.
