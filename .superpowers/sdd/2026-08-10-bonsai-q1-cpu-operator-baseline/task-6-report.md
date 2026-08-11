# Task 6 report — Prism CPU_REPACK runner

## Delivered

- `experiments/E004-q1-cpu-operator-baseline/q1_cpu_operator_runner.cpp`
  implements a real `GGML_OP_MUL_MAT` graph, exact `CPU_REPACK` discovery through
  `ggml_backend_reg_get_proc_address(..., "ggml_backend_dev_get_extra_bufts")`,
  one canonical Q1 upload, ordinary CPU activation/output buffers, public Q8_0
  fixture emission, resident `CLOCK_MONOTONIC_RAW` samples, phase JSONL, and the
  Task 5 wire shape.
- `experiments/E004-q1-cpu-operator-baseline/CMakeLists.txt` supports source and
  accepted-native-build modes and refuses any Prism SHA other than
  `38c66ad0241da4f9fcce541cda8edc219086cec5` unless explicitly unpinned. The
  unpinned target is marked `strict_mode=false` without changing the wire shape.
- `tests/test_q1_cpu_operator_runner.py` is an executable behavior test with a
  literal K=128, M=16 Q1_0 payload, independent hand-derived expected values,
  record-order/memory/phase assertions, and explicit environment/host skips.

## Verification

Host source mode (x86 compile and help only):

```text
/usr/bin/cmake -S experiments/E004-q1-cpu-operator-baseline \
  -B /tmp/q1-cpu-operator-build \
  -DPRISM_SOURCE=/home/random/src/llama-prismml \
  -DBUILD_PINNED_PRISM_FROM_SOURCE=ON -DGGML_CCACHE=OFF
-- Prism source SHA: 38c66ad0241da4f9fcce541cda8edc219086cec5
[100%] Built target q1_cpu_operator_runner
/tmp/q1-cpu-operator-build/q1_cpu_operator_runner --help
q1_cpu_operator_runner --weights weights.q1_0.bin --activation activation.f32.bin
```

The host behavior test reports `OK (skipped=1)` because x86 has no qualified
Q1_0 CPU_REPACK trait; it never produces a qualified runtime result.

Target native accepted-build mode:

```text
PRISM_SOURCE=/home/orangepi/vip9000-lab/src/llama-prismml-shallow-38c66
PRISM_BUILD=/home/orangepi/vip9000-lab/build/cpu-38c66-native-git
git rev-parse HEAD = 38c66ad0241da4f9fcce541cda8edc219086cec5
[100%] Built target q1_cpu_operator_runner
test_literal_q1_repack_graph_and_wire_contract ... ok
Ran 1 test in 0.029s
OK
```

The target evidence emitted `weight_buffer_type=CPU_REPACK`,
`cpu_fallback_count=0`, three records for the two-iteration self-test (one
warmup plus two measured), a 136-byte Q8_0 fixture, and exactly the phase
sequence `setup_begin, repack_begin, repack_end, warmup_begin, warmup_end,
compute_begin, compute_end, output_read_begin, output_read_end, teardown`.
The independent expected output was `[-56,-48,-40,-32,-24,-16,-8,0,8,16,24,32,40,48,56,64]`
within the 0.01 tolerance required for the CPU's F32→Q8 conversion rounding.

`git diff --check` completed with no output. No fan, governor, sudo, or system
policy operation was used.

## Limitations

The x86 host exposes the `CPU_REPACK` buffer name but not the Q1_0 repack trait;
the runner fails closed with an explicit message in that case. Production
qualification therefore requires the accepted A733 native build and target
fixture set.
