# Experiments

Every experiment must be reproducible, attributable to an exact software/hardware state, and comparable with a CPU reference.

## Directory convention

```text
experiments/
  E000-name/
    README.md
    run.sh
    collect.sh
    expected/
    results/
```

Do not commit proprietary runtime binaries, SDK files, model weights, or private device credentials.

## Experiment template

```markdown
# E000 — Title

## Question

What single question does this experiment answer?

## Hypothesis

What result is expected, and why?

## Target

- board revision:
- SoC/device ID:
- RAM:
- cooling:
- OS/kernel:
- NPU driver/runtime:
- SDK/toolchain:
- repository commit:
- upstream llama.cpp commit:

## Inputs

- model/file hashes:
- graph shapes:
- data types:
- prompt/context/batch:

## Procedure

Exact commands and environment variables.

## Metrics

- cold-start time:
- compile/load time:
- warm execution:
- transfer time/bytes:
- CPU utilization/time:
- memory:
- temperature/clocks:
- power/energy:
- numerical error:

## Results

Raw data plus summary statistics.

## Conclusion

Was the hypothesis supported?

## Limitations

What does this experiment not prove?

## Next action

One concrete follow-up.
```

## Initial experiment queue

- `E001-target-fingerprint`: identify exact board, BSP, driver, runtime, and NPU product ID.
- `E002-cpu-llama-baseline`: upstream `llama.cpp` build and `llama-bench` baseline.
- `E003-vendor-sample`: reproduce the smallest official VIPLite/OpenVX sample.
- `E004-runtime-overheads`: initialization, graph load, empty/small invocation, synchronization.
- `E005-matmul-sweep`: supported dtypes and matrix shapes, including decode-like GEMV and prefill-like GEMM.
- `E006-transfer-bandwidth`: host↔NPU copies and zero-copy/import experiments.
- `E007-persistent-graph`: compare one-shot and persistent runtime execution.
- `E008-ggml-single-op`: dispatch one `ggml` operation through an experimental backend.
- `E009-transformer-mlp`: MLP subgraph correctness and end-to-end cost.
- `E010-prefill-decode`: compare candidate partitions for prompt processing and token generation.
