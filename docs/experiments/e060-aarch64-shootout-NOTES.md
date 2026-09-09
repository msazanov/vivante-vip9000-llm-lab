# E060 notes (in progress, 2026-09-09): aarch64-only shootout

Scope per operator: aarch64 only, x86 numbers are not of interest.
Models: i2z1/gigaam-multilingual-ctc-onnx-int8 (done E059) vs ai-sage/GigaAM-v3
family vs HF shortlist, same clips, threads=2 pinned A76.

## v3-int8 (istupakov/gigaam-v3-onnx, 215 MB) in sherpa stack

- vocab_size metadata missing -> added copy v3_ctc.int8.meta.onnx + generated
  v3-tokens.txt (33 ru + <blk>=33). Host smoke: silero OK, RTF 0.16.
- On target (t2, A76), 3 runs survived before reboot spiral:
  ru-short RTF 0.1148, ru-med RTF 0.1285, silero RTF 0.1160 (correct transcript).
  Same speed class as i2z1 (0.12-0.14), fractionally faster.
- Remaining 6 runs (ru-long + 5x FLEURS) were computed but lost to sudden
  reboots (unflushed ext4 writes); partial file in
  benchmarks/results/v3int8-partial.jsonl (3 runs).

## Native torch v3 (ai-sage/GigaAM-v3, revision ctc, 422 MB)

- Host (torch 2.8 CPU, t2): FLEURS ru WER 0.0 (5/5 exact) vs i2z1 0.0154.
  Quality crown goes to v3. RTF 0.16-0.23, but RSS 1740 MB.
- On target: threads=1 on A55 -> RTF 1.81, RSS 1665 MB (below realtime).
  threads=2 on A76 -> board rebooted during load, twice. NOT USABLE on the
  live appliance in fp torch form.

## Board stability incident (5 reboots in ~6 h, intervals shrinking)

- t4 sherpa matrix, 2x v3-torch A76 loads, 1x fully idle/light-commands.
- No OOM/panic signature in user-visible journal; thermals 65-72 C.
- Pattern matches sudden death (power or SoC), not gradual OOM; hermes logs
  from Sep 08 show a prior identical unclean death with 3.5 GB still free.
- Device runs PAUSED to avoid SD corruption. Next: stabilize power/thermals,
  fsck, then resume shortlist battery (vosk-small-ru, wav2vec2-xlsr-ru,
  whisper-turbo-ru 1-clip, NeMo conformer export).
