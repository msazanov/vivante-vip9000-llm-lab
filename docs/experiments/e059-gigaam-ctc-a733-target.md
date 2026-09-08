# E059 — GigaAM CTC int8 ON TARGET (A733 CPU path)

**Status:** `diagnostic`. **Evidence class:** `Verified on target` for CPU
decode (Orange Pi Zero 3W / A733). No NPU execution is claimed — `/dev/vipcore`
was present but unused; this run used the sherpa-onnx CPU provider.

## Target state

- `orangepizero3w.lan` (192.168.31.117), kernel `6.6.98-sun60iw2`, aarch64,
  8 cores, 11884 MiB RAM.
- sherpa-onnx 1.13.7 aarch64 in `~/gigaam-bench/.venv` (built on-device via
  `uv`; no system pip on the image).
- All runs pinned with `taskset -c 6,7` (A76 cluster) to protect the media
  appliance. Co-resident load during runs: `llama-server` RSS ~3.5 GiB,
  `python` ~1.1 GiB, `kodi.bin` ~378 MiB, `hermes` ~239 MiB.

## Decode matrix (threads x audio) — `benchmarks/results/gigaam-ctc-a733-target-001`

| threads | clip | load s | decode s | RTF | peak RSS MB |
|---|---|---|---|---|---|
| 1 | ru-short 3.2s | 4.21 | 0.63 | 0.20 | 377 |
| 1 | ru-med 12.6s | 3.43 | 2.83 | 0.23 | 450 |
| 1 | en-med 7.2s | 3.04 | 1.47 | 0.21 | 450 |
| 1 | silero 5.0s | 2.95 | 1.01 | 0.20 | 434 |
| 2 | ru-short | 3.03 | 0.39 | **0.12** | 350 |
| 2 | ru-med | 2.89 | 1.74 | **0.14** | 453 |
| 2 | en-med | 2.95 | 0.90 | **0.13** | 397 |
| 2 | silero | 2.90 | 0.61 | **0.12** | 421 |
| 2 | ru-long 37.6s | 5.76 | 7.13 | 0.19 | 737 |
| 2 | fleurs ru 8–19s | 2.7–2.8 | 1.0–3.8 | 0.13–0.19 | 383–572 |
| 4 | ru-short | 3.01 | 1.02 | 0.32 | 370 |
| 4 | ru-med | 2.96 | 3.28 | 0.26 | 475 |
| 4 | en-med | 3.10 | 1.80 | 0.25 | 381 |

Threads=2 on A76 is the operating point: RTF 0.12–0.19 everywhere.
Threads=4 regresses (A55 stragglers/contention), matching the lab's E057
observation that 8-core spread is slower than A76-only on this SoC.

## Quality on target

FLEURS `ru_ru` test clips (n=5), threads=2: norm WER per clip
`0.0, 0.0, 0.0, 0.0, 0.0769`, mean **0.0154** — transcript-identical to the
host E058 run clip-for-clip. The INT8 CPU greedy path is deterministic across
x86_64 and aarch64 here. espeak fixtures again decode to garbage on target
too (fixture OOD, not a model finding).

## Incident: reboot during threads=4 phase

After the t4 `en-med` run and before t4 `silero`, SSH dropped
(`Connection reset by peer`); on reconnect uptime showed a fresh boot.
No OOM signature was recoverable without elevated journal access
(previous-boot err log shows only `lircd` errors). Cause is **unknown** —
correlated with the t4 phase, not proven OOM — and consistent with the lab's
recorded history of spontaneous resets under sustained load (E057 notes).
All subsequent solo runs (ru-long, 5x FLEURS, threads=2) completed cleanly.

Operating guidance: cap on-device ASR at threads=2 pinned to A76, run solo
(no parallel matrix), chunk audio to <=15 s to bound RSS next to the live
3.5 GiB `llama-server`.

## Russian STT shortlist for this board (quality x speed, HF-sourced)

Baseline (tested E058/E059): `i2z1/gigaam-multilingual-ctc-onnx-int8`
(ru norm WER 0.015, RTF 0.12–0.19 @t2 on A733, 215 MB, MIT).

| model | why interesting | NPU fit |
|---|---|---|
| `istupakov/gigaam-v3-onnx` | GigaAM v3 CTC with **both** `v3_ctc.onnx` (FP32) and int8 + yaml. FP32 = direct ACUITY import, no torch re-export. Next probe #1 | best |
| `ai-sage/GigaAM-v3` | newer GigaAM, strongest ru in family (254k dl); FP32 source for export | high (after export) |
| `ai-sage/GigaAM-Multilingual` | FP32 source behind the tested int8 (pytorch_model.bin) | high (after export) |
| `nvidia/stt_ru_conformer_ctc_large` | NeMo-official RU Conformer CTC Large (.nemo → ONNX export is standard) | high (CTC, static) |
| `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | most-downloaded RU STT on HF (3.8M); encoder-only CTC, greedy-capable | medium (1.2 GB FP32, attention-heavy) |
| `onnx-community/wav2vec2-large-xlsr-53-russian-ONNX` | same, pre-exported ONNX | medium |
| `openai/whisper-large-v3-turbo` + RU tunes (`coriollon/…-turbo-russian`, `dvislobokov/faster-whisper-large-v3-turbo-russian`) | max quality on CPU, 4x fewer decode steps than large-v3 (6.9M dl upstream) | none (autoregressive) |
| `antony66/whisper-large-v3-russian` | RU large-v3 fine-tune family base | none |
| `alphacep/vosk-model-small-ru` (off-HF dist) | max speed / min RAM (~50 MB), Apache-2.0 | low (already CPU-cheap) |
| Silero STT ru via torch.hub (off-HF) | compact CTC, ONNX-exportable, very fast | medium-high |
| `7Gluk/wav2vec2-large-mms-1b-russian` | MMS-1B RU adapter, quality option | low (1B, heavy) |

Probe order for NPU: istupakov FP32 → NeMo conformer ONNX export →
wav2vec2-xlsr encoder. Whisper-family stays CPU-only (quality tier).
