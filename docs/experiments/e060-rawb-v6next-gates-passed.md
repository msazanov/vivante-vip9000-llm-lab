# E060 — B-raw v6-next: full NPU loop closed, gates passed

**Status:** `diagnostic` (quality gate passed on realistic proxy; production
integration proposed, not performed)

**Claim class:** `diagnostic`

**Evidence class:** *Verified on target* (NBG open/parity/latency) and
*Verified in supplied tooling* (training curves, honest eval). E059 recorded
the failed v2 loop (poisoned data + PTQ failure); this record supersedes it
after full rollback (all v1–v5 weights deleted, old synth quarantined).

Context: single-keyword («глаша») raw-waveform tiny net for Orange Pi Zero 3W
(A733, VIP9000 CID 0x1000003B, VIPLite 2.0.3.2-AW-2024-08-30). Data after
rollback: 36 train clips (33 + 031 + 069, all vosk-verified глаша), voice
negatives, 46 confusables + opus mirrors (25% of neg batch), street noise.
Old synthetic negatives (36 generic files with literal «глаша») quarantined —
see E059 for the poisoning forensics.

## Model

RawB width×1.5: 957,925 params, Conv+LeakyReLU only (asserted at export),
input 800 ms raw → [1,1,80,160] grid (numpy reshape outside graph),
DC-centering + RMS-norm in the single `to_grid` choke point (runtime must
replicate bit-for-bit), score = sigmoid of spatial mean on CPU. Loss:
BCE on mean logit, pos_weight=1.0 (3.0 biased the boundary upward in v3c),
50% positives mixed with street noise at 0–10 dB SNR. 80 epochs, RTX 2070.

## Result 1 — Quality, honest protocol (host ORT)

- train_rec 0.944 @0.3–0.7; eval_rec 0.714 (5/7: misses 028 + 029);
- unseen-voice probes: Маша-holdout (never trained) 2/3 (026, 027 hit;
  029 miss — short/quiet pattern); Андрей (never trained) 2/2;
- clip FA@0.5 0.109 overall (confusables 0.02, telegram voice 0.78 on n=9);
- realistic-stream (75% silence, 10% street noise, 15% confusables, 30 min,
  patience=3, refractory=2 s): FA/hour 10.0 @0.3, **0.0 @0.5** —
  gate FA<0.5/h PASSED. Torture-stream (100% speech) still ~1100/h:
  dense-speech weakness is real and recorded.

## Result 2 — NPU: open, parity, latency (performance governor)

Compile: proven GLaDOS pegasus-CLI recipe (import with `--size-with-batch
True`, per-tensor symmetric_affine + normal quantize, export ovxlib with
`--viv-sdk …/cmdtools`), v2 container (`VPMN 00 00 02 00`), 779 KB.

- `vip_open` OK, int8 in/out (12800/6 B, TF_ASYMM scales via wrapper);
- parity vs ORT CPU (16 windows): cos_mean 0.9999 (min 0.9998),
  max_abs 1.78 logit units. RMS-norm + bounded activations fixed PTQ where
  v2 failed (0.56 per-tensor / 0.04 perchannel+KL). Threshold must be
  calibrated on NPU scores (small absolute shift);
- latency 0.45 ms mean (p95 ~1.0 incl. first-run warmup) — one resident
  graph replaces mel (2.53) + emb-teacher (5.11) + head (0.13).

## Decision / next

Technical path to production is unblocked: recommend side-by-side A/B
behind env flag with live FA watch (Vosk second stage stays) before swap.
Remaining risks: dense-speech FA, kitchen-bg validation (user to provide),
threshold calibration on NPU scores. No credentials or device secrets stored.
