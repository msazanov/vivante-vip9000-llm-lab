# E061 — Supertonic 3 TTS VIP9000 NPU offload, compiler workarounds, and target benchmarks

**Status:** `diagnostic`

**Claim class:** `diagnostic`

**Evidence class:** *Verified on target* (per-stage and end-to-end CPU vs VIP9000 NPU latencies, CPU time, memory pool sizes, and multi-step parity on Orange Pi Zero 3W / Allwinner A733) and *Verified in supplied tooling* (Acuity 6.30.22 `pegasus.py` importer/quantizer/exporter failure modes, `libOpenVX.so` multibranch pass crash dissection, and graph rewrite solutions). This is a TTS offload and NPU compiler study; it does not replace the Bonsai LLM objective.

## 1. Objective and baseline CPU bottleneck

Supertonic 3 (`Supertone/supertonic-3`, OpenRAIL; int8 Sherpa-ONNX export `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`) is a 44.1 kHz multilingual flow-matching TTS pipeline split into four ONNX graphs:

1. `duration_predictor` (`[B, T]`, `[B, 8, 16]`, `[B, 1, T]` -> `[B]`)
2. `text_encoder` (`[B, T]`, `[B, 50, 256]`, `[B, 1, T]` -> `[B, 256, T]`)
3. `vector_estimator` (executed iteratively for `num_steps = 8` or `4` Euler flow-matching steps; `[B, 144, L]`, `[B, 256, T]`, `[B, 50, 256]`, masks, step counters -> `[B, 144, L]`)
4. `vocoder` (`[B, 144, L]` -> `[B, 3072 * L]` waveform at 44.1 kHz)

**Verified on target — CPU baseline (`taskset -c 6,7`, 2x Cortex-A76, ONNX Runtime 1.29.0):**

- Full synthesis takes **2.12-2.27 s wall time** (`4.21-4.34 s` CPU time) for ~2.0-2.15 s of Russian speech (`RTF ~ 1.06-1.13`), which is **~6.8x slower** in wall time and **~9.4x higher** in CPU time than the existing Silero TTS service (`0.27-0.32 s` wall, `0.38-0.46 s` CPU).
- Per-stage CPU profile (`T = 28, L = 31`, 2x Cortex-A76):
  - `duration_predictor`: **2.7 ms**
  - `text_encoder`: **17.5 ms**
  - `vector_estimator`: **245.0 ms per step** (`x8 = 1,960 ms`, **90.7%** of total runtime)
  - `vocoder`: **176.0 ms** (**8.1%** of total runtime)

Offloading `duration_predictor` + `text_encoder` (`~20 ms` combined) is unnecessary; accelerating Supertonic requires offloading `vector_estimator` and `vocoder` to the A733 VeriSilicon VIP9000 NPU (`CID 0x1000003B`, VIPLite `2.0.3.2-AW-2024-08-30`).

---

## 2. What was tried and DID NOT work (negative results and traps)

Every failure mode below was reproduced and isolated during this experiment:

### 2.1. Importing pre-quantized ONNX Runtime INT8 graphs into Acuity (Failed)
- **What was tried:** Using `duration_predictor.int8.onnx`, `vector_estimator.int8.onnx`, and `vocoder.int8.onnx` from the Sherpa-ONNX INT8 package as input to Acuity `pegasus.py`.
- **Why it failed (*Verified in supplied tooling*):** ONNX Runtime dynamic/static INT8 graphs use `DynamicQuantizeLinear`, `QuantizeLinear`, `DequantizeLinear`, and `MatMulInteger` (plus MS domain ops `com.microsoft.nchwc`). Acuity requires float32 source graphs (`Supertone/supertonic-3` FP32 ONNX) and performs its own post-training quantization.

### 2.2. Running `onnx.version_converter` (opset 19 -> 17) or container `onnxsim` (Failed)
- **What was tried:** Converting opset 19 FP32 graphs to opset 17 via `onnx.version_converter.convert_version(model, 17)`, and running `onnxsim` inside `ubuntu-npu:v2.0.10.2`.
- **Why it failed (*Verified in supplied tooling*):**
  - `onnx.version_converter` aborts with `Assertion false failed: No Adapter To Version $17 for Pad`.
  - Inside `ubuntu-npu:v2.0.10.2`, Python 3.8 has `onnx 1.12.0` and an older `onnxruntime` that rejects opset > 15 (`ValidateOpsetForDomain: Current official support for domain ai.onnx is till opset 15`).

### 2.3. Running `pegasus.py import onnx` without explicit `--inputs`, `--outputs`, and per-input `--size-with-batch` (Failed)
- **What was tried:**
  1. Running `pegasus.py import onnx --model vocoder.sim.onnx --input-size-list "1,144,32"` without `--inputs` and `--outputs`.
  2. Running `pegasus.py import onnx` on 7-input `vector_estimator.sim.onnx` without `--size-with-batch`, or with a single `--size-with-batch True`.
- **Why it failed (*Verified in supplied tooling*):**
  1. Without explicit `--inputs` and `--outputs`, `Importonnx.run` crashes at line 103 with `TypeError: object of type 'NoneType' has no len()`.
  2. When `--size-with-batch` is omitted (default `False`), Acuity silently prepends a batch dimension `1` to every entry in `--input-size-list`, turning 3D `[1, 144, 32]` into 4D `[1, 1, 144, 32]` and crashing `vector_estimator` at `/vector_estimator/Concat_7` (`ShapeComputation.concat_shape: AssertionError`) where input `style_ttl` (`[1, 1, 50, 256]`) is concatenated with 3D constant `style_value_special_token` (`[1, 50, 256]`).
  3. Passing a single `--size-with-batch True` for a 7-input model fails with `AcuityError: length of size_with_batch is not equal to input_size_list`.

### 2.4. Exporting 1D `Conv` (`CONV1D`) with `symmetric_affine int8` (Failed)
- **What was tried:** Quantizing `vocoder` with `--quantizer symmetric_affine --qtype int8` (the standard GLaDOS 2D decoder recipe) and running `pegasus.py export ovxlib --pack-nbg-unify`.
- **Why it failed (*Verified in supplied tooling*):** In Acuity 6.30.22 (`ovxlib 1.1.70`), `symmetric_affine int8` assigns `ASYM INT8` to activations and `SYM INT8` to weights. While `vsi_nn_op_conv2d.c` supports this combination, `vsi_nn_op_conv1d.c:op_check:260` rejects it with:
  `E [ops/vsi_nn_op_conv1d.c:op_check:260] Inputs/Outputs data type not support: ASYM UINT8, SYM INT8, SYM INT32, ASYM UINT8` -> `Check node[7] CONV1D fail`.

### 2.5. Exporting `Where` (`VSI_NN_OP_SELECT`) in `vector_estimator` attention masks (Failed)
- **What was tried:** Exporting unmodified attention mask `Equal` + `Where` nodes in `vector_estimator` with `dynamic_fixed_point int8`.
- **Why it failed (*Verified in supplied tooling*):** `vsi_nn_op_select.c:op_check:178` fails at `Check node[270] SELECT fail` (`Inputs/Outputs data type not support: BOOL8, FLOAT16, FLOAT16, DFP INT8`).
- **Secondary trap:** Replacing `Where(text_mask == 0, -inf, scores)` with `Add(scores, (1 - text_mask) * (-10000.0))` works in FP32, but in 8-bit quantization `-10000.0` stretches the `Add` tensor range to `[-10000, +5]`, making the 8-bit step size `~39.2` and crushing all valid attention scores in `[-4, +4]` to zero.

### 2.6. Native `libOpenVX.so.1` crash (`SIGSEGV`, exit 35584) in `transform_merging_multibranch_merge_no_joint` on `vector_estimator` 8-bit export (Failed without workaround)
- **What was tried:** Exporting `vector_estimator` (`ve_nowhere.onnx`) with `dynamic_fixed_point int8` or `asymmetric_affine uint8` via `pegasus.py export ovxlib --pack-nbg-unify`.
- **Why it failed (*Verified in supplied tooling*):** `gen_nbg` segfaults (`Fatal model generation error: 35584`). `catchsegv` and `objdump -d` isolated the crash to `libOpenVX.so.1`:
  `cr_op_get_inputs(NULL)` called from `transform_merging_multibranch_set_cr_graph_for_no_joint` <- `transform_merging_multibranch_merge_no_joint` <- `transform_merging_multibranch_impl`.
  In `vector_estimator`, reshaped `style_ttl`, `text_emb`, and `time_encoder` outputs each fan out into 4-8 parallel `Gemm` (`VSI_NN_OP_FCL`) branches across different transformer blocks that do not directly rejoin (`no_joint`). The 8-bit `merging_multibranch` pass attempts to fuse them and dereferences a null pointer. Setting `VIV_VX_HANDLE_BRANCH=0` or `VIV_VX_ENABLE_GRAPH_TRANSFORM=0` does not disable this pass.

### 2.7. Calibrating `vector_estimator` on synthetic Gaussian `style_ttl` or single `total_step` (Failed quality gate)
- **What was tried:**
  1. Calibrating `vector_estimator` with random `style_ttl ~ N(0, 0.25)` instead of real embeddings from `voice.bin`.
  2. Keeping `vocoder`'s `convnext.9 -> BatchNormalization` unfused during quantization.
  3. Calibrating `ve_dfp16` only on `total_step = 8` and running inference with `num_steps = 4`.
- **Why it failed (*Verified on target*):**
  1. Real `voice.bin` `style_ttl` has `std = 0.0625` (`[-0.74, +0.81]`). With `N(0, 0.25)`, the 8-step ODE trajectory diverges to `[-12, +12]` (`std = 4.96`) instead of converging to `[-3.5, +3.5]` (`std = 0.34`), inflating internal quantizer scales 10-30x and causing 8-step target `latent_cos` to collapse to `0.03-0.09`.
  2. In `vocoder.onnx`, one outlier channel in `convnext.9/Add_output_0` reaches `[-2276.2, +1334.6]` (`std = 29.28`) before `final_norm/BatchNormalization` (`running_var = 3263.4`) scales it back down by ~200x to `std = 0.1733`. Quantizing `convnext.9` before `BatchNormalization` forces `scale = 6.806`, zeroing out the other 511 channels (`std ~ 0.15`).
  3. Calibrating only with `total_step = 8` sets `@Divide_9:out0` (`1 / total_step = 0.125`) to DFP16 `fl = 18` (`max = 0.124996`). At `num_steps = 4`, `1 / 4 = 0.25` clips to `0.125`, halving the Euler step size and dropping 4-step `latent_cos` from `0.992` to `0.346`.

---

## 3. What WORKED (verified solutions and compiler workarounds)

### 3.1. Host ONNX preparation pipeline (*Verified in supplied tooling*)
1. Fix static input/output dimensions (`L = 32, T = 32`) on the FP32 source graphs using `onnx.tools.update_model_dims`.
2. Relabel `opset_import = 17` and `ir_version = 8` (none of the `Pad` nodes use opset-19 `axes` or `wrap`), then run `onnxsim.simplify` in a modern host venv (`onnx 1.23.2`, `onnxruntime 1.31.1`, `onnxsim 0.8.1`).
   - `vocoder`: `401 -> 147` nodes (and `134` nodes after folding `BatchNormalization`), FP32 parity `max_abs = 0.0`.
   - `vector_estimator`: `1004 -> 822` nodes, FP32 parity `max_abs = 5.25e-6`.
3. Import into Acuity (`ubuntu-npu:v2.0.10.2`) with explicit `--inputs`, `--outputs`, and `#`-separated `--size-with-batch`:
   - `vocoder`: `--inputs latent --outputs wav_tts --input-size-list "1,144,32" --size-with-batch True` -> `Error(0), Warning(0)`.
   - `vector_estimator`: `--inputs "noisy_latent text_emb style_ttl latent_mask text_mask current_step total_step" --outputs denoised_latent --input-size-list "1,144,32#1,256,32#1,50,256#1,1,32#1,1,32#1#1" --size-with-batch "True#True#True#True#True#True#True"` -> `Error(0), Warning(0)`.

### 3.2. Eliminating `Where`/`Equal` and bypassing the `libOpenVX.so` multibranch crash (*Verified in supplied tooling*)
1. **Mask arithmetic rewrite:** Replace the 8 post-softmax `Where(latent_mask == 0, 0.0, probs)` nodes with `Mul(probs, latent_mask_4d)` (`max_abs = 0.0` in ORT). Pad shorter text embeddings by repeating the final `</ru>` token embedding so `text_mask` is identically `1.0` across the 32-token window.
2. **`LD_PRELOAD` PLT intercept for `gen_nbg`:** Because `transform_merging_multibranch_impl` in `libOpenVX.so.1` calls `transform_merging_multibranch_merge_no_joint@plt` via the ELF PLT, preloading a one-line shared library during `pegasus.py export ovxlib` cleanly skips the buggy no-joint multibranch fusion without modifying any SDK binary:

```c
/* skip_mb.c: gcc -shared -fPIC -O2 skip_mb.c -o libskip_mb.so */
int transform_merging_multibranch_merge_no_joint(void *a, void *b) { return 0; }
```

With `LD_PRELOAD=/output/libskip_mb.so`, `pegasus.py export ovxlib --pack-nbg-unify` succeeds with **`Error(0), Warning(0)`** for both `dynamic_fixed_point int8` (`50,322,976 B`) and `asymmetric_affine uint8` (`56,708,888 B`), as well as `dynamic_fixed_point int16` (`146,100,368 B`).

### 3.3. Backward residual-stream folding of `BatchNormalization` in `vocoder` (*Verified in supplied tooling & on target*)
Because `convnext.k/dwconv/Conv` (`k = 0..9`) is depthwise (`groups = 512`), the per-channel scale `alpha = weight / sqrt(running_var + eps)` and shift `beta = bias - running_mean * alpha` of `decoder/final_norm/BatchNormalization` can be pushed backward through the entire 10-block residual stream:
- Multiply output channels of `decoder/embed/net/Conv` (`weight` and `bias`) by `alpha`.
- Divide `convnext.k/dwconv/Conv` `weight` by `alpha` (so `dwconv` output into `LayerNormalization` is unchanged).
- Multiply output channels of `convnext.k/pwconv2/Conv` (`weight` and `bias`) by `alpha`, and add `beta` to `convnext.9/pwconv2/Conv` `bias`.
- Remove `decoder/final_norm/BatchNormalization`.

**Result:**
- FP32 parity against original `vocoder.onnx`: **`cos = 0.9999999`, `max_abs = 6.22e-6`**.
- `convnext.0..8` residual range shrinks from `[-35.6, +23.8]` to `[-0.16, +0.29]`, and `convnext.9` shrinks from **`[-2276.2, +1334.6]` (`std = 29.28`) to `[-7.86, +18.94]` (`std = 0.1733`)**.
- On target VIP9000 NPU, `voc_nobn_dfp16.nb` waveform cosine similarity vs stock CPU INT8 jumps from **`0.7155` to `0.8523`** (for comparison, stock CPU INT8 vs CPU FP32 is `0.7322`).

### 3.4. Real-trajectory calibration and `@Divide_9` fixed-point adjustment (*Verified on target*)
- Calibrating `vector_estimator` on 32 real ODE steps (4 Russian utterances x 8 steps using real `voice.bin` `style_ttl` and `style_dp`) and widening `@Divide_9:out0` / `@Reshape_1_output_0_5:out0` from `fl = 18` to `fl = 15` (`max = 0.5`, supporting both 8-step `1/8 = 0.125` and 4-step `1/4 = 0.25`) restores multi-step target convergence:
  - **8-step NPU (`ve_dfp16.nb`) vs 8-step CPU:** `latent_cos = 0.99793` (`SNR = +23.84 dB`) on phrase 1 and `latent_cos = 0.99700` (`SNR = +22.21 dB`, `wav_cos = 0.7993`, `wav_snr = +4.15 dB`) on phrase 2.
  - **4-step NPU (`ve_dfp16.nb`) vs 4-step CPU:** `latent_cos = 0.99241` (`SNR = +18.20 dB`, `wav_cos = 0.7241`) on phrase 1 and `latent_cos = 0.97584` (`SNR = +13.15 dB`, `wav_cos = 0.7274`) on phrase 2.

---

## 4. Target VIP9000 benchmark results (Orange Pi Zero 3W, Allwinner A733)

All target runs were executed on `/dev/vipcore` (NPU devfreq `852 MHz`, CPU pinned `taskset -c 6,7` on 2x Cortex-A76, thermal zone 3 at `61.9-65.2 C`, zero crashes or driver faults).

### 4.1. Per-stage latency: CPU (2x Cortex-A76) vs VIP9000 NPU (`st3_vip_runner`, 10 warm iterations)

| Stage & Variant | NBG size (bytes) | NPU pool (bytes) | Mean wall (ms) | Min wall (ms) | Device `prof_us` | VIP cycles | Speedup vs CPU INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vector_estimator` (1 step) — CPU INT8 ONNX | 78,400,833 | — | 245.00 | 237.50 | — | — | 1.00x |
| `vector_estimator` (1 step) — **NPU DFP8** (`ve_dfp8.nb`) | 50,322,976 | 693,056 | **35.90** | **35.75** | 35,796 | 30,426,804 | **6.82x** |
| `vector_estimator` (1 step) — **NPU UINT8** (`ve_uint8.nb`) | 56,708,888 | 790,592 | **37.00** | **36.87** | 36,915 | 31,384,196 | **6.62x** |
| `vector_estimator` (1 step) — **NPU DFP16** (`ve_dfp16.nb`) | 146,100,368 | 1,514,048 | **70.68** | **70.52** | 70,335 | 59,904,101 | **3.47x** |
| `vocoder` (`L = 32`, 2.23 s audio) — CPU INT8 ONNX | 25,991,073 | — | 176.00 | 175.30 | — | — | 1.00x |
| `vocoder` (`L = 32`, 2.23 s audio) — **NPU DFP8** (`voc_nobn_dfp8.nb`) | 13,801,320 | 2,163,968 | **29.18** | **29.04** | 28,981 | 24,660,189 | **6.03x** |
| `vocoder` (`L = 32`, 2.23 s audio) — **NPU UINT8** (`voc_nobn_uint8.nb`) | 15,499,600 | 2,163,968 | **29.15** | **29.03** | 29,189 | 24,844,505 | **6.04x** |
| `vocoder` (`L = 32`, 2.23 s audio) — **NPU DFP16** (`voc_nobn_dfp16.nb`) | 38,917,784 | 4,326,656 | **47.11** | **46.98** | 46,874 | 39,882,395 | **3.74x** |

### 4.2. End-to-end synthesis comparison on target (CPU frontend + resident VIP9000 NPU `ve` + `vocoder`)

Test utterances (Russian, `sid = 6`, 44.1 kHz output for Supertonic, 24 kHz for Silero):
- **Utterance 1:** `"Privet! Ya gotova pomoch."` (~2.20-2.23 s audio)
- **Utterance 2:** `"Na kukhne vsyo v poryadke."` (~1.74-2.01 s audio)

| Pipeline configuration | Steps | Utterance | Wall time (s) | CPU time (s) | FE CPU (ms) | VE NPU (ms) | Voc NPU (ms) | Latent `cos` / `SNR` | Wav `cos` vs CPU | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| **Silero TTS** (CPU baseline, 24 kHz) | — | Utt 1 | 0.315 | 0.460 | — | — | — | — | — | Production baseline |
| **Silero TTS** (CPU baseline, 24 kHz) | — | Utt 2 | 0.274 | 0.380 | — | — | — | — | — | Production baseline |
| **Supertonic 3 CPU INT8** (Sherpa-ONNX) | 8 | Utt 1 | 2.162 | 4.330 | — | — | — | 1.0000 | 1.0000 | Too slow / high CPU load |
| **Supertonic 3 CPU INT8** (Sherpa-ONNX) | 8 | Utt 2 | 2.119 | 4.212 | — | — | — | 1.0000 | 1.0000 | Too slow / high CPU load |
| **Supertonic 3 NPU (`ve_dfp16` + `voc_nobn_dfp16`)** | **8** | Utt 1 | **0.659** | **0.123** | 45.3 | 566.8 | 47.7 | **0.9979 (+23.8 dB)** | **0.4591** | **3.3x faster, 35x less CPU, high parity** |
| **Supertonic 3 NPU (`ve_dfp16` + `voc_nobn_dfp16`)** | **8** | Utt 2 | **0.658** | **0.122** | 44.1 | 566.3 | 47.7 | **0.9970 (+22.2 dB)** | **0.7993** | **3.2x faster, 34x less CPU, high parity** |
| **Supertonic 3 NPU (`ve_dfp16` + `voc_nobn_dfp16`)** | **4** | Utt 1 | **0.366** | **0.117** | 34.8 | 282.8 | 47.8 | **0.9924 (+18.2 dB)** | **0.7241** | **5.9x faster, 37x less CPU, matches Silero wall** |
| **Supertonic 3 NPU (`ve_dfp16` + `voc_nobn_dfp16`)** | **4** | Utt 2 | **0.376** | **0.121** | 44.5 | 282.9 | 47.6 | **0.9758 (+13.2 dB)** | **0.7274** | **5.6x faster, 35x less CPU, matches Silero wall** |
| **Supertonic 3 NPU (`ve_uint8` + `voc_nobn_dfp16`)** | 8 | Utt 1 | 0.385 | 0.137 | 39.2 | 297.3 | 47.7 | 0.4700 (+0.7 dB) | -0.0285 | Fast (297 ms VE), 8-step PTQ drift |
| **Supertonic 3 NPU (`ve_uint8` + `voc_nobn_dfp16`)** | 8 | Utt 2 | 0.390 | 0.136 | 44.9 | 296.8 | 47.7 | 0.7269 (+1.9 dB) | 0.1036 | Fast (297 ms VE), 8-step PTQ drift |
| **Supertonic 3 NPU (`ve_uint8` + `voc_nobn_uint8`)** | 4 | Utt 2 | 0.213 | 0.114 | 32.2 | 148.8 | 31.9 | 0.2309 (-5.6 dB) | 0.0185 | Fastest (213 ms E2E), 8-bit vocoder noisy |

## 5. Key takeaways for VIP9000 NPU users

1. **DFP16 (`dynamic_fixed_point int16`) on VIP9000 is both fast and numerically faithful for iterative flow-matching and neural vocoders:**
   Unlike E058's float16 export (which ran on the shader at CPU speed), `dynamic_fixed_point int16` executes on the VIP9000 hardware engine at **3.5-3.7x speedup over 2x Cortex-A76 INT8**, while maintaining **0.997-0.998 cosine similarity across 8 recurrent ODE steps** and **35x lower CPU utilization** (`0.12 s` CPU time per utterance vs `4.33 s` on CPU, and `3.1-3.8x` less CPU time than Silero).
2. **8-bit (`uint8`/`int8`) runs 6.0-6.8x faster than CPU (`36 ms/step` VE, `29 ms` vocoder, `213-385 ms` end-to-end),** and single-step `ve_uint8` achieves `cos = 0.9946` (`cos(v) = 0.9496`), but over 8 unrolled Euler steps 8-bit quantization error accumulates (`latent_cos = 0.47-0.73`). Using 8-bit for `vector_estimator` or `vocoder` requires either hybrid quantization (keeping sensitive layers in `int16`) or QAT/distillation.
3. **Three reusable Acuity 6.30.22 workarounds discovered:**
   - Always pass explicit `--inputs`, `--outputs`, and `#`-matched `--size-with-batch "True#..."` for multi-input ONNX models.
   - Preload `int transform_merging_multibranch_merge_no_joint(void *a, void *b) { return 0; }` (`LD_PRELOAD`) when `gen_nbg` segfaults in `libOpenVX.so.1` on multi-branch `FCL`/`Gemm` fan-out graphs.
   - Fold `BatchNormalization` backward through depthwise-separable ConvNeXt residual streams before quantization to eliminate massive pre-BN channel spikes (`[-2276, +1335]`).

---

## 6. Reusable source code and graph-rewrite scripts

### 6.1. Backward residual-stream `BatchNormalization` folding (`fold_vocoder_bn.py`)

```python
import numpy as np
import onnx
from onnx import numpy_helper
from onnxsim import simplify

m = onnx.load("vocoder.sim.onnx")
for n in list(m.graph.node):
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                t = onnx.TensorProto()
                t.CopyFrom(a.t)
                t.name = n.output[0]
                m.graph.initializer.append(t)
        m.graph.node.remove(n)
init_map = {i.name: i for i in m.graph.initializer}

def get_arr(name):
    return numpy_helper.to_array(init_map[name]).copy()

def set_arr(name, arr):
    init_map[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name=name))

bn_w = get_arr("tts.ae.decoder.final_norm.norm.weight")
bn_b = get_arr("tts.ae.decoder.final_norm.norm.bias")
bn_mean = get_arr("tts.ae.decoder.final_norm.norm.running_mean")
bn_var = get_arr("tts.ae.decoder.final_norm.norm.running_var")
alpha = bn_w / np.sqrt(bn_var + 1e-5)
beta = bn_b - bn_mean * alpha

for n in m.graph.node:
    if n.name == "/decoder/embed/net/Conv":
        w_name, b_name = n.input[1], n.input[2]
        set_arr(w_name, get_arr(w_name) * alpha[:, None, None])
        set_arr(b_name, get_arr(b_name) * alpha)
    elif n.op_type == "Conv" and "/convnext." in n.name and "/dwconv/" in n.name:
        w_name = n.input[1]
        set_arr(w_name, get_arr(w_name) / alpha[:, None, None])
    elif n.op_type == "Conv" and "/convnext." in n.name and "/pwconv2/" in n.name:
        w_name, b_name = n.input[1], n.input[2]
        b_new = get_arr(b_name) * alpha
        if "/convnext.9/" in n.name:
            b_new = b_new + beta
        set_arr(w_name, get_arr(w_name) * alpha[:, None, None])
        set_arr(b_name, b_new)

new_nodes = []
for n in m.graph.node:
    if n.name == "/decoder/final_norm/BatchNormalization":
        continue
    if "/decoder/final_norm/BatchNormalization_output_0" in n.input:
        n.input[:] = [
            "/decoder/convnext.9/Add_output_0"
            if x == "/decoder/final_norm/BatchNormalization_output_0" else x
            for x in n.input
        ]
    new_nodes.append(n)
del m.graph.node[:]
m.graph.node.extend(new_nodes)
m_sim, _ = simplify(m)
onnx.save(m_sim, "vocoder_nobn.onnx")
```

### 6.2. Attention `Where` / `Equal` elimination (`remove_where_ve.py`)

```python
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxsim import simplify

m = onnx.load("vector_estimator.sim.onnx")
one_init = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="mask_one_const")
neg_init = numpy_helper.from_array(np.array(-10000.0, dtype=np.float32), name="mask_neg_inf_const")
m.graph.initializer.extend([one_init, neg_init])

new_nodes = []
bias_added = False
for n in m.graph.node:
    if n.op_type == "Equal":
        continue
    if n.op_type == "Where":
        cond, _, x_false = n.input
        out = n.output[0]
        if cond.endswith("Cast_3_output_0"):
            new_nodes.append(helper.make_node(
                "Mul",
                inputs=[x_false, "/vector_estimator/vector_field/main_blocks.3/attn/Unsqueeze_27_output_0"],
                outputs=[out],
                name=n.name + "_mul_mask",
            ))
        elif cond.endswith("Cast_2_output_0"):
            if not bias_added:
                new_nodes.append(helper.make_node(
                    "Sub",
                    inputs=["mask_one_const", "/vector_estimator/vector_field/main_blocks.3/attn/Transpose_1_output_0"],
                    outputs=["text_mask_inv"],
                    name="text_mask_inv_sub",
                ))
                new_nodes.append(helper.make_node(
                    "Mul",
                    inputs=["text_mask_inv", "mask_neg_inf_const"],
                    outputs=["text_attn_bias"],
                    name="text_attn_bias_mul",
                ))
                bias_added = True
            new_nodes.append(helper.make_node(
                "Add",
                inputs=[x_false, "text_attn_bias"],
                outputs=[out],
                name=n.name + "_add_bias",
            ))
    else:
        new_nodes.append(n)

del m.graph.node[:]
m.graph.node.extend(new_nodes)
m_sim, _ = simplify(m)
onnx.save(m_sim, "ve_nowhere.onnx")
```

### 6.3. Resident VIPLite 2.x C wrapper for multi-step `vector_estimator` and `vocoder` (`libst3_vip.c`)

Supports `VIP_BUFFER_FORMAT_FP16`, `VIP_BUFFER_QUANTIZE_TF_ASYMM` (`UINT8`/`INT8`), and `VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT` (`INT8`/`INT16`), keeping both `ve` and `vocoder` networks resident on `/dev/vipcore` and executing all `num_steps` ODE iterations inside C without returning to Python between steps.

