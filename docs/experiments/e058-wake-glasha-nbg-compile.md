# E058 — GLaDOS wake-word head NBG compile screen and A55 cost breakdown

**Status:** `diagnostic`

**Claim class:** `diagnostic`

**Evidence class:** *Verified on target* (cost breakdown; NBG runtime failure +
bisection, **corrected 2026-09-09**: post-fix runtime + quantization parity +
streaming A/B) and *Verified in supplied tooling* (NBG compile; scanner failure
reproduction). No quality, correctness, or optimization claim is made.

This record captures two bounded results from the GLaDOS wake-word assistant
work (2026-09-09), which are relevant to the lab's VIP9000 capability map:

1. A per-operator cost breakdown of the openWakeWord-style wake chain on the
   pinned A55 cores of the target board.
2. A successful NBG compile of an oww-style conv graph through the
   `pegasus.py export ovxlib` path, contrasted with a failing import of an
   STFT-style graph in the same toolchain image.

The wake chain is: `melspectrogram.onnx` (STFT-like frontend, [1, 14000] →
[T, 32]) → `embedding_model.onnx` ([1, 76, 32, 1] → [1, 1, 1, 96]) → a custom
36 KB int8 wake head ([1, 16, 96] → [1, 16]). The models are the stock
openWakeWord feature models plus a locally trained head; the OWW models are
Apache-2.0. This is a 12.5 fps streaming workload with resident graphs, not a
Bonsai decode task; the results are recorded here because they are A733/VIP9000
facts, not because the wake task replaces any lab objective.

## Result 1 — Verified on target: per-frame cost breakdown

Orange Pi Zero 3W, Allwinner A733 (cpu0–5 Cortex-A55 capacity 385, cpu6–7
Cortex-A76 capacity 1024, kernel 6.6.98-sun60iw2). onnxruntime CPU sessions
with `intra_op_num_threads = 1`, pinned `taskset -c 0,1`, measured 2026-09-09
on 50 repetitions per stage over a real captured wake sample. Cadence target
is one chain pass per 80 ms (12.5 fps) with resident sessions.

| Stage | ms per 80 ms frame | Share of chain |
|---|---:|---:|
| `melspectrogram` (14000-sample input) | 10.70 | 34% |
| `embedding_model` (one run per frame) | 20.58 | 65% |
| wake head int8 | 0.21 | 0.7% |
| **Chain total** | **31.5** | **≈ 39% of one A55 core** |

Service-level context: the deployed wake hybrid (oww prefilter → Vosk
confirmation) measures ≈ 53% of the pinned-core CPU quota, versus ≈ 29.3%
for the previous Vosk-only detector, consistent with the breakdown above plus
callback overhead. Measurement pitfall recorded for future chains: an
end-to-end loop benchmark divided by all frames under-reports the chain cost
when early warm-up frames skip stages (an earlier 14000-sample-buffer loop
measured a misleading 12.8 ms/frame; the per-stage table above is the honest
number).

## Result 2 — Verified in supplied tooling: embedding NBG compile succeeded

Host compile in `ubuntu-npu:v2.0.10.2` (same image as the GLaDOS NBG
pipeline), target `VIP9000NANODI_PID0X1000003B`:

1. Static-shape onnx: batch dim fixed to 1, input `[1, 76, 32, 1]`.
2. `pegasus.py import onnx --input-size-list 1,76,32,1` — succeeded; Acuity
   graph input lid `input_1_68`, output shape `(1, 1, 1, 96)` confirmed.
3. `pegasus.py quantize --quantizer symmetric_affine --qtype int8
   --algorithm normal`, 50 iterations calibrated on real wake-sample mel
   contexts — 0 errors, 0 warnings.
4. `pegasus.py export ovxlib --pack-nbg-unify` — 0 errors.
5. Artifact: `network_binary.nb`, **319,776 bytes**, SHA-256
   `05ab27a034087c7f50f21df6e25d8124a0398a5c7a138bb17528784d9ae36564`.

Contrast with E018: the *native* OpenVX `vxGenerateNBG` path SIGSEGVs even for
a minimal MatrixMultiply graph, while the *pegasus ovxlib* export path
completed a 20-Conv/19-LeakyRelu/5-MaxPool/2-Reshape graph without errors on
the same image. This is evidence that the ovxlib export path is the reliable
compiler route for conv graphs on this toolchain build; it does not establish
that the NBG executes on the target (no wrapper, no golden yet).

## Result 3 — Verified in supplied tooling: STFT-style graph fails the onnx scanner

`melspectrogram.onnx` (Conv/Pow/MatMul/Log/ReduceMax/Transpose/Clip mix,
opset 13) fails `pegasus.py import onnx`:

```text
File "acuitylib/onnx_ir/scanner/onnx_util.py", line 63, in OnnxNode.__init__
AttributeError: op_type
```

Reproduced on both the original graph (29 nodes) and an onnx-simplifier output
(18 nodes, `Constant`/`Cast` constant-folded). The failure is in the scanner's
numpy-backend node walk, not in operator support. **Resolution (2026-09-09):**
the identical crash reproduces at the same scanner line in all three available
toolchains — pegasus 2.0.10.2 (`ubuntu-npu:v2.0.10.2`), AcuityLite 6.51.0
(`acuitylite:ready`), and AcuityLite 6.57.1 (newest PyPI). All involved
modules are cythonized (`.so`, no source on the image or in the
VeriSilicon/acuitylite repo) and `OnnxLoader.load` exposes no flag to skip
pattern matching / value inference. The bug is unfixable from our side; the
mel frontend stays on CPU.

## Result 4 — Verified on target: embedding NBG prepare fails; bisection isolates MaxPool

The phase-2 embedding NBG (int8, 319,776 bytes) **crashes on the device**:
`vip_prepare_network` raises SIGSEGV (VIPLite driver 2.0.3.2-AW-2024-08-30)
after `vip_create_network`, the io query, and buffer creation all succeed. A
VIPLite wrapper (`edge_wake_viplite.c` → `libwake_vip.so`; int8 TF_ASYMM
scale/zero plumbing) was built on the device to drive the tests.

Bisection over graph substitutions (host edits → docker compile → device open
test, each in a segfault-isolated subprocess):

| Variant | Compile | Device | Conclusion |
|---|---|---|---|
| original, pegasus int8 (319,776 B) | ok | prepare SIGSEGV | baseline |
| v1: 19× `Max(x, −0.4)` → `LeakyRelu(alpha=1)` (identity) | ok (385,464 B) | prepare SIGSEGV | Max clamp exonerated (fires only at x < −40) |
| v4: v1 + input/output `Reshape` dropped, IO `[1,1,76,32] → [1,96,1,1]` | ok (385,464 B) | prepare SIGSEGV | Reshape exonerated → **MaxPool guilty** |
| float export of the original graph (846,376 B, quantize skipped) | ok | prepare SIGSEGV | **MaxPool broken in float too** |
| AcuityLite 6.51.0 compile of the original graph (321,808 B, ovxlib 1.1.30) | ok | `vip_create_network` fails at ioctl `VIPDRV_SET_TASK_PROPERTY` (container header byte 4 = 0x05 v5 vs pegasus 0x00); byte-4 patch (0x05→0x00, section table identical) → create_network ok → prepare SIGSEGV | crash is **runtime-level, toolchain-independent** |

Early probe failures (not needed for the verdict): `MaxPool → Conv(stride-2)`
and `MaxPool → Slice` substitutions die at the pegasus quantize stage
(TF-backend layout conventions; lessons recorded: an onnx 1×1 conv weight must
be `[c_out, c_in, 1, 1]`, and Slice ends must be positive when steps ≠ 1).

**Verdict:** ~~MaxPool crashes `vip_prepare_network` on this VIPLite build in
int8 and float.~~ **Corrected 2026-09-09 (phase 4):** the phase-3 verdict is
superseded — see the correction below. The prepare crashes were wrapper
artifacts (reversed io-query sizes) plus poisoned driver state, not graph
topology.

## Correction (2026-09-09, phase 4) — bisection verdict invalidated; the embedding NBG runs

Two independent defects invalidated the phase-3 bisection:

1. **Wrapper io-query bug.** `vip_query_input`/`vip_query_output` return sizes
   head-aligned **reversed** (the production decoder queries `[1,192,1,32]` as
   `[32,1,192,1,0,0]`). `edge_wake_viplite.c` trusted the returned order and
   built mis-shaped buffers, so every phase-3 prepare SIGSEGV was caused by the
   wrapper, not by graph topology. Fixed by reversing the query sizes per
   `num_of_dims` (env switches `WAKE_VIP_REV`/`WAKE_VIP_MEM` kept for sweeps).
2. **Driver-state confound.** After ~20 SIGSEGVs, even the production GLaDOS
   decoder NBG segfaulted at prepare in any standalone process (both wrappers)
   while serving fine inside `edge_server`. The vipcore module retains poisoned
   state until reboot (positive control post-reboot: decoder prepare OK).

Post-fix results (Verified on target):

| Probe | Result |
|---|---|
| decoder NBG, positive control | prepare OK |
| original pegasus int8 embedding NBG (319,776 B) | prepare OK |
| int8 `wake_vip_run` latency | **1.33–1.41 ms/frame** vs 20.58 ms CPU |
| fp16 (float-export) NBG | 20.77 ms/frame — equal to CPU, no speedup |
| int8 window parity (200 val windows) | cos 0.9869 (p10 0.9771), max_abs 17.25 |
| output int8 range | [-51.3, 70.9] — clips teacher extremes (±100) |
| recalibration (1200 mixed samples, perchannel + KL, 368,144 B NBG) | **identical** IO scales and parity numbers |
| streaming A/B (18 clips, fresh state) | CPU 6/8 pos, 3/10 prefilter FA; int8 NPU: 4 pass→FA, 1 pass→miss |

**Instrumentation retraction.** The AcuityLite recalibrated NBGs were never
actually measured: the device parity script used a hardcoded NBG path, and an
argv-argument fix was lost with a `/tmp` wipe (a concurrent agent reboot), so
every post-fix parity run silently re-opened the original teacher NBG. The
"identical to 16 digits" numbers were the original NBG measured four times.
`nbg_meta.json` proves the ranges DID change under recalibration: input scale
0.468131 -> 0.354047, output range [-51.3, +70.86] -> [-54.18, +86.03]. The
stock teacher graph int8 noise (cos 0.987) stands; the structural-invariance
claim is retracted. A head retrained on device-dumped NPU
embeddings (78,974 windows) becomes self-consistent but learns the
quantization noise as positive signal — the pass→FA flips persist.

**Final verdict (student, 2026-09-09 night):** the stock teacher graph int8
is decision-unsafe (5/18 clip flips). The remedy is the **distilled student
with a bounded output range** (592K params, Conv/LeakyRelu only, standardized
targets ~[-4.5, +3.9]): int8 NBG 578,320 B with **self-parity cos 0.9980**
(p10 0.9962, max_abs 0.49, no clipping possible) at **0.20 ms/frame** (103×
vs the 20.58 ms CPU teacher). Head retrained on device-dumped student-NPU
embeddings (78,974 windows, 94 s; hold recall 0.989 / FA 0.008 @0.3).
48-clip A/B: recall 27/32 vs 26/32, prefilter FA 2/16 vs 3/16.
Full-set A/B (358 clips: all 162 positives + 196 negatives): **recall
135/162 (CPU) vs 129/162 (student)** - both chains miss 2 raw recordings
(different sessions; the delta concentrates in augmented variants of
held-out sessions), **prefilter FA 107/196 vs 89/196**; the hard-negative
set (71/72 firing on BOTH chains) carries no differential signal, so the
semantic FA is 36/124 vs 18/124; **chain 7.53 vs 3.56 ms/frame**, embedding
20.58 vs 0.20 ms. Production-viable; the integration sits behind
HUGGINGVOICE_WAKE_EMB_NPU=1 with full CPU fallback, pending operator
enablement.

## Artifacts

NBG and manifest live outside this repository (recorded, not published):

| Artifact | Bytes | SHA-256 | Location |
|---|---:|---|---|
| `emb-network_binary.nb` | 319,776 | `05ab27a0…36564` | `/var/tmp/wake-npu/emb/wksp/emb_nbg_unify/network_binary.nb` and `/home/random/wake-glasha/models/npu/emb-network_binary.nb` |
| `emb-npu.json` (manifest: onnx/graph hashes, lid, shape, calibration) | 963 | — | `/var/tmp/wake-npu/emb/npu.json` and `/home/random/wake-glasha/models/npu/emb-npu.json` |
| v1 NBG (19× Max → LeakyRelu identity; MaxPool+Reshape kept) | 385,464 | `c087eb44…8f00f` | `/var/tmp/wake-npu/bisect/v1/wksp/v1_nbg_unify/network_binary.nb` |
| v4 NBG (v1 + both Reshapes dropped; IO `[1,1,76,32]→[1,96,1,1]`) | 385,464 | `22d2d0f6…dd5c` | `/var/tmp/wake-npu/bisect/v4/wksp/v4_nbg_unify/network_binary.nb` |
| float NBG (original graph, quantize skipped) | 846,376 | `54b84558…d2f4` | `/var/tmp/wake-npu/emb/wksp/emb_float_nbg_unify/network_binary.nb` |
| AcuityLite NBG (original graph, ovxlib 1.1.30) | 321,808 | `4914912a…148a` | `/var/tmp/wake-npu/emb_al/wksp_al/network_binary.nb` |
| `nbg_meta.json` (AcuityLite IO scales/zero-points) | 1,043 | — | `/var/tmp/wake-npu/emb_al/wksp_al/nbg_meta.json` |

Origin chain: openWakeWord `embedding_model.onnx` (Apache-2.0) → static-batch
onnx (`68c0c39d…`) → Acuity graph → int8 quantize → NBG. Adding the NBG to the
public artifact manifest is a follow-up under the publication policy.

## Boundaries and remaining work

- The embedding NBG **does run** on the target post-fix (1.4 ms int8), but the
  int8 output range clips the teacher's extremes (structural, see the
  correction) and streaming wake decisions diverge from CPU on 5/18 eval
  clips. Not production-safe; the production wake chain stays fully CPU.
- fp16 is numerically exact but runs at CPU speed — no acceleration on this
  NPU for this graph.
- int8 quantization error on the wake scores is moot for the stock oww
  embedding; it becomes relevant again only for a pool-free distilled
  embedding with a bounded output range.
- The mel frontend remains on CPU (scanner bug, toolchain-wide).
- Wake-domain quality claims live in the wake project's own report
  (`/home/random/wake-glasha/models/REPORT.md`, outside this repository).

## Next steps

1. ~~Adapt the VIPLite wrapper~~ — done: `edge_wake_viplite.c` built on the
   device; the io-query reversal bug found and fixed (correction above).
2. Quantization-aware distilled student (bounded output range by design):
   the remaining research path to a production-safe NPU embedding. The device
   dump, head-parity, and streaming-A/B infrastructure exist.
3. ~~Probe `acuitylite:ready` for the mel import~~ — done, negative: the
   scanner bug reproduces identically in AcuityLite 6.51.0 and 6.57.1
   (Result 3 resolution).
4. Recompile probe with `minimize_layer_error=True` (in flight at record
   time); if the structural ranges still hold, close the int8 line and keep
   the embedding on CPU.
