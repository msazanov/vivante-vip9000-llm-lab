# E059 — Raw-waveform tiny wake net (variant B): full-NPU loop, negative quality/parity result

**Status:** `diagnostic`

**Claim class:** `diagnostic`

**Evidence class:** *Verified on target* (latency, NBG open/parity numbers,
container forensics) and *Verified in supplied tooling* (training curves, ORT
eval, compile logs). No quality or optimization claim is made — both the task
gate (FA/hour) and the int8 parity gate FAILED. The mechanical path
train→compile→deploy→run is proven; the numbers below say why v2 must not ship.

Context: openWakeWord-style chain on Orange Pi Zero 3W (A733,
2×A76+6×A55, 12 GB LPDDR5, VIP9000 CID 0x1000003B, VIPLite 2.0.3.2-AW-2024-08-30).
Production today is a hybrid: mel+head on CPU, distilled student embedding
(592K, Conv/LeakyReLU only) on NPU. Variant B replaces mel+emb+head with one
resident raw-waveform NBG: `[1,12800] int16-normalized samples (800 ms) →
[1,1,80,160] grid (numpy reshape, outside graph) → 426,073-param
Conv/LeakyReLU stack → [1,1,2,3] logits → sigmoid+max on CPU`.

## Result 1 — Verified on target: per-frame cost breakdown (performance governor)

ORT single-thread CPU vs NPU, per 80 ms frame:

| stage | latency |
|---|---|
| mel CPU | 2.53 ms mean |
| emb teacher CPU | 5.11 ms |
| head int8 CPU | 0.13 ms |
| CPU chain total | ~7.8 ms |
| student-emb NPU | 0.19 ms mean |
| hybrid chain (mel CPU + emb NPU + head CPU) | ~2.9 ms (mel = 91%) |
| raw_b NBG (either quantizer, full chain in one graph) | 0.19–0.35 ms mean, p95 ~0.9 cold / ~0.26 steady |

The NBG runs. The speedup is real. Everything below is why it still cannot ship.

## Result 2 — Verified in supplied tooling: B-raw trains but fails the task gate

Training: RTX 2070, 80 epochs, pos_pool 330 (33 raw clips + pitch/tempo/noise
augments with tempo-scaled timings), hard negatives adjacent to the word,
1:1 balance, LR 1.5e-3, BCE pos_weight 3. Best probe checkpoint:
thr 0.3, rec 1.0, FA 0.0625 (tiny probe: 8 train + 8 eval clips + 16 windows).
v1 (33 raw clips, 3:1 neg ratio) collapsed to all-negative (loss 0.03, rec 0.0)
and was killed at ep45.

Honest eval of `raw_b.onnx` (host ORT, 33 train / 8 eval clips, 124 neg files):

- @0.3: train_rec 0.576, eval_rec 1.000 (n=8); FA 1.000 on
  background/generic/adversarial/street (clip-max).
- 30-min synthetic background stream (patience=3, refractory=2 s):
  1020 FA/hour @0.3, 590 @0.5, 306 @0.7, 236 @0.89.

Reference under identical protocol with correct 2.5 s stream warmup
(warmup matters: an empty-buffer probe scores 0 on 0.9–2.7 s clips because the
16-embedding buffer never fills — recorded pitfall): teacher
(student-emb + student head) reaches train rec@0.3 0.64, eval rec@0.3 0.38 on
the same new telegram-codec clips. The new distribution is harder than the old
training data for BOTH pipelines (lab reproduction of the 1/5 production
problem); distillation ceiling is low until owner-mic data exists.

## Result 3 — Verified on target: int8 PTQ parity gate FAILED

16 calibration windows, NPU (TF_ASYMM dequant via wrapper) vs ORT CPU:

- per-tensor symmetric_affine: cos_mean 0.56, max_abs 9.7
- perchannel_symmetric_affine + kl_divergence: cos_mean 0.04, max_abs 15.7
- gate: cos ≥ 0.99 (student-emb holds 0.998). Raw net needs QAT; PTQ fails.

## Result 4 — Toolchain forensics (all verified by reproduction tonight)

1. `pegasus.py import onnx` in `ubuntu-npu:v2.0.10.2` needs
   `--size-with-batch True`; without it the same model fails.
2. CLI shape inference (`smart_toolkit._conv_shape`, IndexError) crashes on
   torch-origin NCHW graphs (`[1,1,80,160]`, `[1,1,76,32]` — even a 2-conv
   stem), while the TF-origin teacher graph (`[1,76,32,1]`) imports fine.
   Working hypothesis: NHWC layout heuristic (H=1 → negative conv dim).
   Direct `OnnxLoader` API imports the same graphs without errors.
3. Direct API export emits a v1 NBG container (header `VPMN 20 00 01 00`,
   ovxlib 1.1.83) that the target rejects (`vip_create_network status=-4`,
   `VIPDRV_SET_TASK_PROPERTY status=-3`). Neither byte-4 patch (0x20→0x00)
   nor full versionbytes patch (`20 00 01 00`→`00 00 02 00`) fixes it —
   the container differs structurally, not just in version bytes.
4. The GLaDOS recipe (`glados_ru/compile_vip9000_int8.py`: pegasus CLI
   import → quantize → `export ovxlib` with `--optimize TARGET --dtype
   quantized --pack-nbg-unify --viv-sdk …/cmdtools --build-platform make
   --target-ide-project linux64`) emits a v2 container (`VPMN 00 00 02 00`)
   that OPENS on the target. Same image, same model, different path.
5. `cmp -l` proves the production `student-emb.nb` (565 KB, opens OK) is
   byte-identical to the Sep-9 `emb_stud` direct-API output except byte 4
   (`0x05`→`0x00`) — i.e. the documented E058 byte-4 patch, applied to a
   1.1.30-style container. Patch applies to `05 02` containers, not to `01 xx`.
6. All three `ubuntu-npu` images carry acuitylib 6.30.22; `acuitylite:ready`
   carries 6.51.0. Every direct-API export observed tonight says 1.1.83
   (the Sep-9 1.1.30 string was the truncated tail of a multi-stage export
   log — tonight's teacher control prints 1.1.53 then 1.1.30 then the final
   stage).

## Artifacts (host paths, outside git)

- `models/raw-b/raw_b.{pt,onnx,json}` — 426K fp32, Conv+LeakyReLU only (asserted).
- `models/raw-b/raw_b-pertensor.nb` (335 KB, cos 0.56), `raw_b-perchannel.nb`
  (397 KB, cos 0.04) — both open on target, both fail parity.
- `bin/train_raw_b.py`, `bin/prepare_rawb_compile.py` — training + calibration.
- Compile recipe that works: pegasus CLI flow with `--size-with-batch True`
  and the GLaDOS export flags (see Result 4).

## Decision

Variant B v2 does not ship (quality + parity red). Production stays on the
hybrid student chain. Next: owner-mic clips (kitchen mic, not telegram codec)
plus B-raw v3 with QAT from scratch on that data. No credentials, passwords,
or device secrets are stored in this record.
