# E058 — GigaAM Multilingual CTC int8: host screen + VIP9000 compile probe

**Status:** `diagnostic`. **Evidence classes:** `Verified upstream` (x86_64 host),
`Verified in supplied tooling` (AcuityLite 6.51.0 docker). **Nothing here ran on
the A733 target; no target or NPU throughput is claimed.**

## Why this model

Two prior candidates were rejected for the VIP9000 path:

- `bzikst/faster-whisper-large-v3-russian-int8` — CTranslate2 `model.bin`
  (1.55 GB), no ONNX, autoregressive Transformer. Nothing for ACUITY to ingest.
- `vigneshlabs/cohere-transcribe-03-2026-int8-onnx` — ONNX yes, but 2.7 GB,
  2B Conformer + 256-step Python decode loop. Not an NBG shape.

`i2z1/gigaam-multilingual-ctc-onnx-int8` (MIT, 220M Conformer, 215 MB,
single `model.int8.onnx`, charwise CTC + greedy CPU decode) is the first
candidate with an NPU-compatible *structure*: one static encoder graph, no
autoregressive loop. This experiment screens it on the host and probes the
compiler gate.

## Host decode matrix — `benchmarks/results/gigaam-ctc-host-matrix-001`

`sherpa-onnx` 1.13.7, CPU provider, fresh process per cell, 1 warmup + N reps.
Host: 12-core x86_64, 31 GB RAM (workstation, not target).

| threads | clip (dur) | load s | decode s | RTF | peak RSS |
|---|---|---|---|---|---|
| 1 | ru-short 3.2s | 2.57 | 0.74 | 0.23 | 398 MB |
| 1 | ru-med 12.6s | 2.36 | 3.02 | 0.24 | 406 MB |
| 1 | ru-long 37.6s | 2.44 | 10.23 | 0.27 | 639 MB |
| 2 | ru-short | 2.67 | 0.57 | 0.18 | 394 MB |
| 2 | ru-med | 2.30 | 1.81 | **0.14** | 414 MB |
| 2 | ru-long | 2.81 | 8.45 | 0.22 | 636 MB |
| 4 | ru-med | 2.73 | 1.85 | 0.15 | 419 MB |
| 4 | ru-long | 3.23 | 7.22 | 0.19 | 635 MB |
| 8 | ru-short | 3.74 | 1.75 | 0.54 | 388 MB |
| 8 | ru-med | 4.28 | 3.74 | 0.30 | 412 MB |

Findings: 2–4 threads optimal; 8 threads regress (oversubscription, worst on
short clips). Load 2.3–4.3 s grows slightly with threads. RSS 377–419 MB
typical, ~636 MB on 37.6 s audio (frame buffers scale with length — chunked
streaming will be required on the 6 GB service cap regardless of backend).

## Fixture lesson (diagnostic, keep)

`espeak-ng` Russian/English fixtures decode to garbage (norm WER 0.83–1.0)
while natural speech decodes near-perfectly. espeak formants are
out-of-distribution for this model. espeak numbers above are valid for
RTF/RAM scaling only and must never be quoted as model quality.

## Quality on natural speech — `benchmarks/results/gigaam-ctc-fleurs-quality-001`

FLEURS test split, streaming, greedy CTC, norm WER (lowercase, no punct):

- `ru_ru` n=5: **mean 0.015** (4 exact, 1 single-word substitution).
  Consistent with the card's FLEURS 4.4 (unnormalized) claim.
- `kk_kz` n=3: mean 0.144 (digit verbalization, hyphen artifacts).
- `en_us` n=3: mean 0.13 (consistent with "English is weaker, ~12").
- Local 5 s neural-TTS Russian sample (`silero_test.wav`): near-perfect
  transcript at all thread counts, RTF 0.18–0.26.

## Compiler probe — `benchmarks/results/gigaam-ctc-acuity-import-001`

Graph: ir 8 / opset 17, `features(batch,64,seq_len)` → `log_probs(batch,seq_len,71)`,
3662 nodes / 34 op types. Quant layer: `DynamicQuantizeLinear` x163,
`MatMulInteger` x128, `ConvInteger` x51. Risk ops: `LayerNormalization` x96,
`Softmax` x16.

| probe | result |
|---|---|
| P1 import int8 ONNX, symbolic dims | FAIL, bare `AssertionError` in smart-node shape inference |
| P2 import with dims frozen (1,64,1000) | FAIL, same assert → symbolic dims not the blocker |
| P3 control: float mobilenetv2-12, same harness | OK → harness works, failure isolated to this file |
| P4 `check-model` on the file | exit 0 → valid ONNX, rejection is importer-internal |

Conclusion: AcuityLite 6.51 does not ingest ORT dynamic-quant graphs. The NPU
path needs an FP32 re-export (`export-multilingual-ctc.py` ships in the model
repo: `gigaam.load_model("multilingual_ctc")` → `to_onnx`), a fixed input
window, and ACUITY static INT8 calibration on Russian-speech mel features.

## Custom kernels

Yes — unsupported Conformer ops can ride custom kernels: custom EVIS/VXC
kernels inside NBG are `Verified on target` (E003/E019/E020 paths execute and
pass independent goldens), and the supplied SDK headers declare custom-op /
matrixmul / fullconnect / rmsnorm interfaces
(`docs/npu/vip9000-capabilities.md`, `docs/toolchain/inventory.md`).
Each custom op still needs its own golden + H2D/run/D2H phase gate; none is
claimed for LayerNorm/Softmax yet.

## INT8-only + 32-bit bus strategy

- NPU makes sense only at INT8: the published file's *dynamic* INT8 is an
  ORT-CPU quantization and is not reusable — plan ACUITY static INT8
  (`ASYMMETRIC_AFFINE`, uint8/int8) with a mel-feature calibration set.
- Against the 32-bit interface (510 MHz controller readback; 19.2 GB/s is a
  ceiling, not a measurement): keep the network resident, run the full encoder
  in one NBG invocation per window, leave CTC greedy (`argmax` + blank strip)
  on CPU. Split plan if the full 220M graph does not fit: Conformer blocks 0–N
  on NPU as one partition, remainder + head as second, overlap H2D/D2H with
  compute via windowed streaming (10–15 s windows, 2 s hop). Every variant gets
  the phase profile from `docs/profiling/profiling-contract.md` (H2D, run,
  device, D2H, total, traffic) — device time alone is never a result.

## Next

1. FP32 re-export (`GIGAAM_BUILD_DIR`, needs torch + `gigaam` package).
2. ACUITY import of FP32 with fixed window; record unsupported-op list.
3. Calibration set: ~100 Russian clips → mel features (64 bins, same frontend).
4. Static INT8 NBG; on-target VIPLite run with phase profile vs CPU golden.
5. Custom EVIS kernels only for ops that actually fail import/quality gates.
