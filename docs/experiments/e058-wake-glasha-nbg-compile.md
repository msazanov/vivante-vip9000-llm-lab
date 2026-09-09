# E058 — GLaDOS wake-word head NBG compile screen and A55 cost breakdown

**Status:** `diagnostic`

**Claim class:** `diagnostic`

**Evidence class:** *Verified on target* (cost breakdown) and *Verified in supplied
tooling* (NBG compile). No quality, correctness, or optimization claim is made.

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
numpy-backend node walk, not in operator support. **Hypothesis —** the Acuity
6.x onnx scanner does not handle this op mix or its attribute layout; the
newer `acuitylite:ready` image (AcuityLite 6.51.0, per
[toolchain inventory](../toolchain/inventory.md)) is the untested next probe.
Until then the mel frontend stays on CPU.

## Artifacts

NBG and manifest live outside this repository (recorded, not published):

| Artifact | Bytes | SHA-256 | Location |
|---|---:|---|---|
| `emb-network_binary.nb` | 319,776 | `05ab27a0…36564` | `/var/tmp/wake-npu/emb/wksp/emb_nbg_unify/network_binary.nb` and `/home/random/wake-glasha/models/npu/emb-network_binary.nb` |
| `emb-npu.json` (manifest: onnx/graph hashes, lid, shape, calibration) | 963 | — | `/var/tmp/wake-npu/emb/npu.json` and `/home/random/wake-glasha/models/npu/emb-npu.json` |

Origin chain: openWakeWord `embedding_model.onnx` (Apache-2.0) → static-batch
onnx (`68c0c39d…`) → Acuity graph → int8 quantize → NBG. Adding the NBG to the
public artifact manifest is a follow-up under the publication policy.

## Boundaries and remaining work

- The NBG has **not** been executed on the target: no VIPLite wrapper exists
  for these shapes (the GLaDOS `libglados_vip.so` contract is decoder-
  specific), no golden, no score-parity validation, no end-to-end timing.
- int8 quantization error on the wake scores is unknown; parity on the 106
  captured wake samples is the required gate before any device switch-over.
- The mel frontend remains on CPU; the wake chain remains fully CPU today.
- Wake-domain quality claims live in the wake project's own report
  (`/home/random/wake-glasha/models/REPORT.md`, outside this repository).

## Next steps

1. Adapt the VIPLite wrapper (`edge_viplite.c` GLaDOS pattern; device runtime
   libraries `vip_lite.h`, `libNBGlinker.so` per the toolchain inventory) to
   the embedding NBG; cross-compile aarch64.
2. Wire the wrapped NPU call into the wake runtime behind a fallback flag;
   run score parity (CPU vs NPU) over the 106 wake samples and the adversarial
   set.
3. Probe `acuitylite:ready` (AcuityLite 6.51.0) for the mel import.
