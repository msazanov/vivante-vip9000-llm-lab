# E057 — Vikhr-Qwen-2.5-1.5B CPU Load-Type Screen and Assistant Integration on A733

**Status:** `unqualified`

**Claim class:** `screen`

**Evidence class:** **Verified on target** (timings, thermals, hashes); reboot
cause remains **Hypothesis**; LFM2.5 comparison figures are **Verified on
target** via E056 on this repository.

A CPU-only llama.cpp characterization of the Russian instruct model
`Vikhr-Qwen-2.5-1.5B-Instruct` on the Orange Pi Zero 3W (Allwinner A733),
recorded on 2026-09-06. It maps decode throughput across CPU load types
(thread count, core-cluster affinity, quantization), measures prompt
processing with a lean assistant profile, and records the end-to-end chain
Hermes (agent) → llama-server (Pi) → GLaDOS TTS (x86 host). It is not a
quality result, not an NPU result, and not an optimization claim.

The machine-readable record is
[`summary.json`](../../benchmarks/results/e057-vikhr-a733-cpu-loadtypes/summary.json).

## Results — decode across load types

Single-sample screens; each row is one chat completion.

| Load type | Quant | Decode tok/s | ms/token | Generated tokens |
|---|---|---:|---:|---:|
| all 8 cores, `-t 8`, no affinity, `ondemand` | Q6_K | 1.480 | 677.3 | 42 |
| `taskset -c 4-7`, `-t 4` | Q6_K | 3.012 | 332.4 | 16 |
| `taskset -c 6-7`, `-t 2` (A76 only) | Q6_K | 4.940 | 202.5 | 15 |
| `taskset -c 6-7`, `-t 2` (A76 only) | Q4_K_M | 6.938 | 144.1 | 16 |
| `taskset -c 6-7`, `-t 2`, governor `performance` | Q4_K_M | 6.863 | 145.8 | 34 |

Findings, **Verified on target**:

- The A733 cluster split is 2× Cortex-A76 (`cpu6-7`, `cpu_capacity` 1024) +
  6× Cortex-A55 (`cpu0-5`, `cpu_capacity` 385), all cores 1,794,000 kHz max.
- Decode on all 8 cores is ~4.7× slower than on the 2 A76 cores alone
  (1.48 vs 6.94 tok/s). The A55 stragglers dominate the thread pool; this is
  a scheduling artifact, not a core-count benefit. It directly refutes the
  intuition "more threads = faster decode" on this SoC.
- Q4_K_M is ~1.4× faster than Q6_K on the same partition (144.1 vs 202.5
  ms/token), consistent with the lower bytes-per-token read.
- Governor `performance` vs `schedutil` on the big cores: no material decode
  difference (6.86 vs 6.94 tok/s). `performance` was still pinned via the
  root unit `cpufreq-performance.service` to remove frequency ramp-up jitter
  on short requests and to stabilize the little cluster under Kodi load.

## Results — prompt processing (first-message prefill)

| Prompt | Approx. tokens | Prompt rate | Source |
|---|---:|---:|---|
| Minimal chat question | 47 | 16.8 tok/s (59.4 ms/tok) | server `print_timing` |
| Lean Hermes profile (no tools/skills/memory) | ~605 | ~19.3 tok/s | 35.67 s total incl. 30-token generation at 6.69 tok/s |
| Full Hermes default profile (41 tool schemas) | ~11,400 | 17.7 tok/s | 2,048 tokens at 18% progress after 115.96 s |

Findings, **Verified on target**:

- Prompt processing on 2 A76 cores is ~17–19 tok/s and scales neither with
  the fat tool-schema profile nor with quantization; it is compute-bound on
  the A76 NEON path (`libggml-cpu-armv8.2_2.so` selected at runtime).
- The full Hermes default profile costs ~11 minutes of prefill before the
  first token; it is unusable on this target. The lean profile (system
  prompt ≈ 10 KB, zero tool schemas) cuts first response to ≈ 40 s
  end-to-end including Hermes overhead.

## Assistant chain end to end

**Verified on target**: `vikhr -z "<question>"` (Hermes profile `vikhr`,
provider `orangepi-vikhr`) returned a correct concise Russian answer; the
same text rendered through the native-RU GLaDOS Style-Bert-VITS2 profile
`v3-1000` produced valid non-silent WAV audio (5.49 s, RMS 0.064, style
`Deep`; second style `Standard` also verified). The GLaDOS renderer requires
the verified `espeak-ng` build (`GLADOS_ESPEAK_NG` env override); the system
`espeak-ng` mis-stresses Russian and aborts.

Hermes integration facts, **Verified in supplied tooling**: the agent floor
`MINIMUM_CONTEXT_LENGTH = 64_000` (`agent/model_metadata.py`) rejects models
below 64K context; server runs with `-c 64000` (KV + weights ≈ 3 GB under
the 4 GB `MemoryMax`). Per-provider `models.<model>.context_length` in
`config.yaml` is the supported override path.

## Incidents

- **Hypothesis (cause unverified):** two spontaneous reboots under
  llama-server load (~21:44 and ~22:00 local). PSU brownout vs thermal
  protection cannot be distinguished from user-space evidence; peak observed
  SoC temperature was 86–89 °C before the incidents. Load testing should
  keep runs short until the PSU/thermal question is resolved.

## Identity and configuration

Runtime: `llama-server` build `10827` (v0.4.0-dev, commit `d03efa5d5`),
prebuilt `llama-b10827-bin-ubuntu-arm64.tar.gz`, backend
`libggml-cpu-armv8.2_2.so`. Server flags: `--host 0.0.0.0 --port 8081
-c 64000 -t 2` under `taskset -c 6,7`, `MemoryMax=4G`, user systemd unit
`vikhr-llama.service`.

| Model file | File bytes | SHA-256 | Runtime type / parameters |
|---|---:|---|---|
| `Vikhr-Qwen-2.5-1.5b-Instruct-Q4_K_M.gguf` | 985,700,992 | `d82b16244a33dbbde4742e38e42b2b3d74dd883f4b8ff4ae53eca29aabf60662` | `qwen2 1.5B Q4_K - Medium` |
| `Vikhr-Qwen-2.5-1.5b-Instruct-Q6_K.gguf` | 1,272,392,320 | `48733fdb79be6aa0e5032818d802643d0cb5898d0056f4d228eae262d71bcf5f` | `qwen2 1.5B Q6_K` |

Sources: [Vikhrmodels GGUF mirror](https://huggingface.co/Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct-GGUF).

LFM2.5 cross-reference, **Verified on target** via E056 (same board, build
9594, `-t 8`, all cores, r=5): `LFM2.5-8B-A1B-Q4_K_M.gguf` (5,155,564,768 B,
SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`)
decoded at a 5-repetition mean of 5.203 tok/s — slower than this 1.5B dense
screen on the A76-only partition, before any A76-only MoE test. A SHA-matched
copy of that exact file is staged on the target for a server-mode
prefill/TTFT follow-up (planned E058).

## Boundaries

- Decode rows are single-sample screens, not r=5 means; E056 remains the
  methodological reference for repeated measurement.
- No deterministic output comparison, prompt rubric, or quality threshold
  was run; the record is `unqualified` for that reason.
- CPU only; no NPU or GPU backend executed.
- SoC temperature comes from `thermal_zone0` only.
- The reboot incidents are recorded as observations; no root-cause evidence.
- `file bytes` is the complete GGUF size; runtime `model_size` was not
  separately captured for this screen.

E047 remains the required path for prompt-based compatibility and quality;
this screen does not modify `current-best`.
