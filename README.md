# Vivante VIP9000 LLM Lab

Research and engineering workspace for using the Vivante VIP9000 NPU together with the Arm CPU cores of the Allwinner A733 SoC for local large-language-model inference.

The primary target device is an Orange Pi Zero 3W with an Allwinner A733 and 12 GB of system memory. The intended long-term result is a maintainable `ggml`/`llama.cpp` backend or companion execution path that can offload useful transformer workloads to the NPU while retaining reliable CPU fallback.

> Status: **NPU and packed-ternary capability research**. No performance, compatibility, or licensing claim should be treated as final until it is reproduced on the target board and recorded in this repository.

## Mission

1. Inventory the exact A733/VIP9000 hardware, drivers, runtime libraries, SDK, and legal terms available on the target board.
2. Establish reproducible CPU-only `llama.cpp` baselines.
3. Characterize the NPU: supported data types, operators, tensor layouts, shape restrictions, compilation flow, memory-transfer cost, and concurrency behavior.
4. Determine whether Ternary Bonsai packed weights can be consumed directly by custom OpenCL/EVIS kernels without model-wide INT4 expansion.
5. Prototype a `ggml` accelerator backend with transparent CPU fallback.
6. Determine the most effective partition between CPU, programmable PPU/EVIS and native NN cores for prompt processing and token decoding.
7. Publish reproducible benchmarks, correctness checks, patches, and integration notes.

## Optimization objective

The primary metric is steady-state decode throughput in tokens per second. Prompt-processing throughput and time to first token are measured and optimized separately. A speed result is eligible for promotion only when it is compared with a pinned CPU reference and passes a declared quality guardrail; faster output with missing quality data remains `unqualified`, and a quality regression remains recorded as `rejected`.

Profiling is part of every experiment from its first run. Raw command output, process and board telemetry, runtime phase timings, copies, memory, clocks, thermals, repetition statistics and quality evidence are retained according to [`docs/profiling/profiling-contract.md`](docs/profiling/profiling-contract.md).

## Preferred technical direction

The working hypothesis is a heterogeneous backend:

- `ggml` remains responsible for model loading, graph construction, scheduling, tokenization, sampling, and CPU fallback.
- Optimized AArch64 CPU kernels remain the mandatory reference and may remain the best token-at-a-time ternary decode path.
- The programmable VIP PPU/EVIS path handles custom packed low-bit kernels, packing/layout work, RoPE and normalization candidates.
- TIM-VX, VIPLite, or the board-vendor Vivante runtime compiles supported large fixed-shape operations or subgraphs for the eight native NN cores.
- Compiled graphs and kernels are cached and reused; model execution must not invoke an ONNX conversion pipeline per token.
- Unsupported or inefficient nodes remain on the CPU.
- Data movement is measured explicitly. Offload is accepted only when end-to-end latency, throughput, energy, memory footprint, or CPU availability improves.

This is a hypothesis, not a commitment. Operator-level offload, subgraph offload, precompiled NBG dispatch, direct packed-ternary PPU kernels and model-specific execution will all be evaluated.

## Repository map

```text
AGENTS.md                                             Project rules and research protocol
benchmarks/README.md                                  Benchmark and quality contract
benchmarks/models/                                   Per-model append-only test tables
docs/architecture/backend-plan.md                     Backend design and phased implementation
docs/evidence/orange-rag-prior-tests.md               Sanitized prior local evidence boundary
docs/evidence/a733-fan-policy-2026-08-09.md            Installed 30 °C fan policy evidence
docs/evidence/a733-extra-cooler-thermal-baseline-2026-08-09.md Clean sustained CPU thermal baseline
docs/evidence/prism-cpu-reference-build-2026-08-09.md Pinned native runtime and first model execution
docs/evidence/a733-cpu-optimization-matrix-2026-08-09.md Source-audited CPU A/B matrix
docs/evidence/vip9000-next-capability-probe-2026-08-09.md Verified SDK assets and next NPU probe
docs/evidence/bonsai-q1-npu-partition-audit-2026-08-10.md Exact Bonsai Q1 tensor/offload inventory
docs/evidence/q1-uint8-nbg-packed-carrier-design-2026-08-10.md Packed Q1 over proven UINT8 NBG design
docs/evidence/vip9000-phase-profile-2026-08-10.md Real H2D/run/D2H target profile and terminology
docs/evidence/q1-vip9000-evis-bonsai-2026-08-11.md Real packed Q1×Q8 EVIS and Bonsai scaling result
docs/evidence/a733-ddr-secure-dfs-2026-08-12.md Read-only DDR secure-DFS/SMC boundary and hashes
docs/superpowers/specs/2026-08-10-q1-vip9000-backend-design.md Approved-scope Q1 backend research specification
docs/hardware/a733.md                                 A733 hardware facts and validation checklist
docs/hardware/orange-pi-zero-3w.md                    Observed target-board fingerprint
docs/legal/licensing.md                               Licensing and redistribution matrix
docs/npu/vip9000-stack-and-capabilities.md            SDK/runtime layers and preliminary capability map
docs/npu/ternary-bonsai-custom-kernel-feasibility.md  Direct packed-ternary feasibility assessment
docs/npu/a733-vip9000-strengths-for-bonsai.md          CPU/PPU/NN strengths and likely partitioning
docs/profiling/profiling-contract.md                   Required metrics and promotion gates
docs/research/llm-operation-compatibility.md          Transformer/ggml operation mapping
docs/research/related-work.md                         Relevant projects and prior art
docs/toolchain/inventory.md                           Sanitized compiler/runtime/tool inventory
experiments/E001-vip9000-capability-probe/            Target-side SDK/operator experiment
experiments/E002-packed-ternary-kernel/               Packed Q2_0 OpenCL/EVIS and native-NN study
experiments/E039-q1-pair-wholek/                      Target golden and rejected whole-K packed-Q1 CPU pair
experiments/E040-q1-register-lut/                     Target golden and rejected register-only Q1 nibble-LUT
experiments/E044-cluster-prfm/                        Short PRFM ranking and sustained-load reset evidence
experiments/E045-dspark-stock-oom/                    DSpark memory/functional preflight sequence and graphs
experiments/README.md                                 Experiment template and reproducibility rules
research/decisions/002-packed-ternary-research-direction.md Research direction record
research/open-questions.md                            Unresolved technical and legal questions
references/sources.md                                 Curated primary and secondary sources
tooling/README.md                                     Inventory, profiling and result-recording commands
```

## Initial milestones

### M0 — Evidence and legal inventory

- Obtain the exact board SDK package, release version, license/EULA, headers, libraries, compiler tools, and sample projects.
- Identify which materials may be committed, linked, redistributed, or only referenced.
- Record the exact kernel, BSP, NPU driver, firmware, runtime, and device identifier.

### M1 — Reproducible baselines

- Build upstream `llama.cpp` on the target board.
- Benchmark representative Ternary Bonsai GGUF files on optimized AArch64 CPU.
- Capture CPU topology, clocks, thermals, memory bandwidth, power mode, and build flags.

### M2 — Minimal NPU execution

- Compile and execute a tiny known-correct graph on VIP9000.
- Compile and execute a custom OpenCL operation.
- Determine whether the VXC/EVIS compiler path is available in the exact SDK.
- Measure cold compilation, warm execution, host↔NPU transfer, and numerical error.

### M3 — Packed ternary proof of concept

- Consume Q2_0 group-64 and Bonsai group-128 packed bytes directly.
- Implement portable OpenCL and EVIS bit-unpack/dot-product paths.
- Compare direct PPU GEMV/GEMM, PPU tile-unpack plus native NN GEMM, CPU NEON, native INT8 and optional INT4.
- Preserve long-lived packed model storage.

### M4 — `ggml` proof of concept

- Register an accelerator device through the `ggml` backend API.
- Implement buffer management, capability checks, graph execution, synchronization, compiled-kernel caching and CPU fallback.
- Offload one useful packed operation or repeatable subgraph with correctness tests.

### M5 — Transformer partition

- Evaluate MLP, projections, RoPE, normalization, attention and full-block partitioning separately.
- Compare prefill, decode, batched decode and speculative decode; they have different shapes and bottlenecks.
- Add cost-aware CPU/PPU/NN scheduling.

### M6 — Real model benchmark

- Run a selected Bonsai GGUF or model-specific graph end to end.
- Report prompt-processing throughput, decode throughput, time-to-first-token, memory, power, CPU load, NPU utilization, and output-quality checks.

## Safety and licensing

Do not commit private SDK archives, vendor runtime binaries, firmware, model weights, generated NPU binaries, or documentation unless their license explicitly permits repository storage and redistribution. Keep hashes, version strings, acquisition instructions, and local path conventions instead.

See [`docs/legal/licensing.md`](docs/legal/licensing.md).

## Current evidence snapshot

Confirmed from official, BSP and upstream source:

- The exact A733 product ID `0x1000003b` reports eight active NN cores, one programmable shader/PPU core, EVIS, 256 shader threads and 512 KiB VIP SRAM.
- The target reports NN GEMM, FP16 ALU, FP32 I/O, a stream processor and an early 4-bit feature, but no packed 4-bit coefficient mode, no group quantization, no dynamic shapes, no tensor DMA and no high-performance decode feature.
- VeriSilicon describes the VIP9000 PPU as a 128-bit vector engine with OpenCL and EVIS, with programmable instructions and possible parallel execution with NN accelerators.
- The A733 SDK tree includes OpenVX, CLC, VSC and NNVXC compiler/runtime libraries, while TIM-VX demonstrates both public custom OpenCL operations and internal VXC/EVIS source kernels.
- Current `llama.cpp` Q2_0 stores four 2-bit values per byte plus an FP16 group scale; Prism Ternary Bonsai uses the same core idea with group-128 release files.
- The target VIPLite runtime is now execution-verified: driver 2.0.3.2-AW,
  CID `0x1000003b`, one runtime-visible device/logical core, and a recovered
  ShuffleNetV2 UINT8 NBG completed 99 measured resident loops at 2.846 ms
  median host run time and 2.803 ms median device profile time. The fixture has
  no golden output, so this is compatibility/performance evidence rather than
  a correctness qualification.
- A phase-aware resident runner completed 1000/1000 target iterations with no
  repeat mismatch: steady median H2D 66.250 µs, host run 2918.750 µs, device
  2815 µs, D2H 7.834 µs, end-to-end 3025.668 µs. NPU stayed at 1008 MHz and
  peaked at 42.16 °C; no thermal-throttling evidence was observed. This remains
  unqualified because repeat equality is not an independent CPU golden.
- A direct packed-ternary PPU kernel is technically plausible, but it should not be assumed to use all eight NN cores.
- A tile-unpack-to-native-NN path may exploit the eight cores but risks losing the memory advantage through intermediate traffic.
- Our packed Q1_0×Q8_0 EVIS kernel now executes on the target without an
  expanded DDR weight tensor. On the same 16×128 golden workload it reduces
  device cycles from 133472 to 13331 (10.01×), passes golden and remains stable
  for 100/100 runs.
- Real Bonsai `blk.0.ffn_gate` slices also pass golden, but a measured 1024×5120
  EVIS tile takes 24.137 ms end-to-end while the CPU completes the full
  17408×5120 layer in 3.850 ms. The programmable EVIS path is therefore not a
  production decode offload; the next gate is EVIS unpack fused with a native
  NN tensor operation whose intermediate tile stays on-chip.
- Community A733 work has executed complete transformer-body graphs through VIPLite, demonstrating feasibility but not yet optimal autoregressive LLM decode.
- E039 whole-K packed-Q1 pair passes 18/18 bit-exact target golden cases, but
  its assembly microgate is 5.88–14.41% slower than two native SIMD calls;
  it is therefore rejected for llama.cpp integration.
- E040 register-only Q1 nibble-LUT passes the target exhaustive/canary golden
  gate, but at K=5120 it takes 4507.1 ns versus 1741.1 ns for one native SIMD
  group whole-K call (2.58865x latency, 61.37% lower speed). It is rejected;
  additional TBL/index work does not remove the full-GEMV memory bottleneck.
- E038 recovers the built-in DDR clock provider and SMC FID `0xc0000096`, but
  the secure DFS/training sequence remains opaque; runtime SMC/MMIO/raw DDR
  writes remain prohibited. See the [Russian DDR evidence](docs/evidence/a733-ddr-secure-dfs-2026-08-12.md).
- E044 `big-only PRFM` достиг 1.018629 токенов/с только в коротком ranking
  screen `n=8/r=1`. Три sustained-load попытки перезагрузили плату до появления
  throughput или quality record; CPU оставался ниже 65.1 °C при лимите guard
  85 °C. Production-статус — `UNQUALIFIED`, а не подтверждённый рекорд.
- E045a загрузил официальный Bonsai-27B DSpark с `n=0`, но kernel завершил
  процесс по OOM. В команде не было `-c`, поэтому эта сборка выбрала полный
  контекст модели 262144 токена. Это `CONFIGURATION_FAIL`, а не доказательство
  того, что DSpark не помещается при ограниченном контексте.
- E045b с явным `-c 512` загрузил target и DSpark без OOM и сохранил около
  6.39 GiB `MemAvailable`; затем example отклонил неподдержанный путь `n=0`.
  E045c повторил bounded load, но однотокенный prompt оставил capture-prefill
  пустым и выявил robustness-дефект driver до генерации.
- E045e завершил первый functional smoke официального DSpark: 2 фактических
  токена при 0.263 ток/с, 8 предложено и 0 принято. E045f target-only измерил
  один внутренний eval-шаг при 1.12 ток/с. Области timer различаются, поэтому
  направленный разрыв 4.26× является только диагностикой; ещё нужны точные
  token IDs/golden и повторный benchmark с одинаковыми timers.

The next hard gates are:

- [`E001`](experiments/E001-vip9000-capability-probe/README.md): target-verified SDK, operation, shape, type, memory and overhead measurements.
- [`E002`](experiments/E002-packed-ternary-kernel/README.md): direct Q2_0 packed execution, EVIS capability and CPU/PPU/NN performance comparison.
- [`E003`](experiments/E003-q1-packed-carrier/README.md): target-verified
  canonical Q1_0×Q8_0 EVIS kernel. Синтетическое микроядро прошло golden и
  ускорилось в 10,01× по cycles относительно скалярного NPU варианта. Реальное
  масштабирование Bonsai также прошло golden, но отклонено по скорости полного
  слоя. Следующая ступень — fused EVIS unpack → native NN FC/Conv внутри NBG.
- [`E039`](experiments/E039-q1-pair-wholek/README.md): target golden для
  whole-K packed-Q1 pair. Корректность принята, локальное ускорение отвергнуто.
- [`E040`](experiments/E040-q1-register-lut/README.md): target exhaustive
  golden для register-only Q1 nibble-LUT. Baseline — один native SIMD group;
  корректность принята, timing отвергнут, интеграции нет.
- [`E044`](experiments/E044-cluster-prfm/README.md): короткий cluster-aware
  PRFM screen и три full-reset timeline. Графики отделяют promising ranking от
  отсутствующего production throughput/quality.
- [`E045a`](experiments/E045-dspark-stock-oom/README.md): официальный
  Bonsai-27B DSpark, kernel OOM и доказанный конфаундер полного контекста.
  Следующий корректный gate использует явный `-c 512`.

## Текущие графики

Основная [XY-карта экспериментов](benchmarks/charts/experiment-xy-overview.svg)
разделяет две XY-системы и correctness-панель: latency/tokens/s полной Bonsai
27B, latency/inferences/s служебного ShuffleNet NPU и CPU golden Q1.
Failed-запуски без метрики перечисляются, но не получают выдуманную координату
скорости. Подробная таблица запусков находится в
[карточке модели Bonsai 27B](benchmarks/models/bonsai-27b.md).

Старый [сводный график бенчмарков](benchmarks/charts/benchmark-overview.svg)
остаётся вторичным обзором сохранённых CPU/NPU свидетельств. Неквалифицированные
строки показывают наблюдение производительности, но не подтверждают качество.

Отдельный [график фаз VIPLite](benchmarks/charts/vip9000-shufflenet-phase-profile.svg)
является диагностикой драйвера, а не графиком токенов: он показывает подготовку
входа, выполнение NPU и чтение результата для первого и 999 steady запусков.
Термины, методика, raw hashes и границы golden-проверки описаны в
[русском отчёте](docs/evidence/vip9000-phase-profile-2026-08-10.md).

Новый [график packed Q1×Q8](benchmarks/charts/q1-vip9000-evis-bonsai-20260811.svg)
показывает отдельно честное A/B микроядра на одинаковой форме и XY-масштабирование
реального `blk.0.ffn_gate`. Полая оранжевая точка полного NPU-слоя — явно
подписанная линейная оценка, а не измерение. Методика и выводы находятся в
[русском отчёте E003](docs/evidence/q1-vip9000-evis-bonsai-2026-08-11.md).
