# Разбиение Bonsai-27B Q1_0 между CPU и VIP9000

## Итог

Первый активный кандидат — packed Q1 GEMV для трёх FFN-проекций
`ffn_gate`, `ffn_up`, `ffn_down`. Они составляют 2,406,481,920 байт, или
63.632% всех Q1-весов. Это максимальная однородная часть весового потока и она
использует две уже выбранные production shapes: `5120×17408` и
`17408×5120`.

Второй кандидат — Q1-проекции 48 recurrent GDN-слоёв. Full attention и LM head
исследуются после FFN. Embedding lookup, рекуррентное состояние, nonlinearities,
KV attention и sampling на первом этапе остаются на CPU.

Это не утверждение, что Q1 kernel уже запускается на VIP9000. На target доказан
VIPLite/NBG путь для заранее скомпилированной UINT8-сети; совместимый custom
PPU/OpenCL/EVIS Q1 kernel пока является проверяемой гипотезой.

## Закреплённый артефакт

- Файл: `Bonsai-27B-Q1_0.gguf`.
- Размер: `3,803,452,480` байт.
- SHA-256:
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Архитектура: `qwen35`, 64 блока, embedding 5120, FFN 17408.
- Tensor inventory: 851 tensor, из них 498 `Q1_0` и 353 `F32`.
- Q1 payload: `3,781,877,760` байт.
- Q1 block: 18 байт на 128 весов, 1.125 бит/вес.

Inventory получен read-only командой:

```bash
PYTHONPATH=/home/random/src/llama-prismml/gguf-py \
python3 -m gguf.scripts.gguf_dump --json \
  /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf
```

## Точные семейства Q1-тензоров

| Семейство | Форма GGUF | Count | Q1 bytes | Доля Q1 |
|---|---:|---:|---:|---:|
| `ffn_down` | `17408×5120` | 64 | 802,160,640 | 21.211% |
| `ffn_gate` | `5120×17408` | 64 | 802,160,640 | 21.211% |
| `ffn_up` | `5120×17408` | 64 | 802,160,640 | 21.211% |
| recurrent `attn_qkv` | `5120×10240` | 48 | 353,894,400 | 9.358% |
| recurrent `attn_gate` | `5120×6144` | 48 | 212,336,640 | 5.615% |
| recurrent `ssm_out` | `6144×5120` | 48 | 212,336,640 | 5.615% |
| `output.weight` | `5120×248320` | 1 | 178,790,400 | 4.728% |
| `token_embd.weight` | `5120×248320` | 1 | 178,790,400 | 4.728% |
| full-attn `attn_q` | `5120×12288` | 16 | 141,557,760 | 3.743% |
| full-attn `attn_output` | `6144×5120` | 16 | 70,778,880 | 1.872% |
| full-attn `attn_k` | `5120×1024` | 16 | 11,796,480 | 0.312% |
| full-attn `attn_v` | `5120×1024` | 16 | 11,796,480 | 0.312% |
| recurrent `ssm_alpha` | `5120×48` | 48 | 1,658,880 | 0.044% |
| recurrent `ssm_beta` | `5120×48` | 48 | 1,658,880 | 0.044% |
| **Итого** | | **498** | **3,781,877,760** | **100%** |

Строки таблицы исчерпывают все Q1-тензоры; категории «прочие Q1» нет.
`token_embd.weight` хранит 4.728% Q1 payload, но decode читает одну embedding
строку, а не всю матрицу. Остальные 497 Q1-тензоров соответствуют полным
проекциям, включая LM head.

Нижняя оценка полного Q1-потока на decode без полной embedding-матрицы:

```text
3,781,877,760 - 178,790,400 = 3,603,087,360 байт/токен
```

Это 95.272% Q1 payload. При измеренных `0.726175 ток/с` получается минимум
`2.616 GB/s`; F32 weights, KV/state, activation и служебный трафик сюда не
входят.

## Реальный qwen35-граф

По `src/models/qwen35.cpp` модель чередует:

- 48 recurrent GDN-слоёв, где `(layer + 1) % 4 != 0`;
- 16 full-attention слоёв: 3, 7, 11, …, 63.

### Recurrent слой

1. QKV projection `5120→10240`.
2. Gate projection `5120→6144`.
3. Alpha/beta projections `5120→48`.
4. Conv1d и SiLU.
5. GDN state update.
6. Normalization/gating.
7. Output projection `6144→5120`.
8. FFN gate/up/down `5120↔17408`.

### Full-attention слой

1. Q projection `5120→12288`.
2. K/V projections `5120→1024`.
3. Q/K norm, RoPE, KV attention.
4. Gate и output projection `6144→5120`.
5. FFN gate/up/down `5120↔17408`.

Получается 384 Q1 projections в recurrent слоях, 112 в full-attention слоях и
один LM head: 497 полных GEMV на decode token. Embedding lookup считается
отдельно.

## Сопоставление с реальными возможностями

| Этап | Статус | Первый маршрут |
|---|---|---|
| ARM Q1_0×Q8_0 | Доказано | CPU DOTPROD + текущий 4×4 repack |
| VIPLite device/runtime/NBG | Доказано | Resident precompiled NBG |
| Native NN UINT8 graph | Доказано только на ShuffleNet fixture | Не считать Q1 support |
| Packed Q1 PPU/OpenCL/EVIS | Правдоподобно, не доказано | C0 compiler/runtime handshake |
| Native NN packed Q1 | Не доказано | Не использовать в первом milestone |
| SRAM-only Q1→INT8→NN | Неизвестно | Только после измерения DDR bytes |
| Восемь NN engines | Неизвестно | Runtime показывает один logical core |
| PowerVR GPU | Отдельный accelerator | Не является VIP9000 backend |

Доказанный resident NBG benchmark дал median host/device `2.846/2.803 ms` и
разницу около `42 µs` на вызов. Переносить эту разницу напрямую на custom Q1
kernel нельзя. Как ориентир, `497 × 42 µs ≈ 20.9 ms/token`; реальный overhead
обязан измеряться на Q1 fixture.

## Ранжирование переноса

1. **FFN packed GEMV.** 63.632% Q1 bytes, одинаковые shapes во всех 64 слоях,
   максимальная потенциальная отдача от одного семейства kernel.
2. **Recurrent QKV/gate/out.** 20.675% Q1 bytes вместе с alpha/beta. Малые
   alpha/beta не запускаются отдельными VIP jobs до появления fusion.
3. **Full-attention Q/K/V/O projections.** 6.238% Q1 bytes. Сам attention и KV
   cache пока остаются CPU.
4. **LM head.** 4.728% Q1 bytes на каждый token и около 1 MiB F32 logits;
   нужен отдельный wide-output benchmark.
5. **Embedding.** Не переносить: это lookup одной строки, а не полный GEMV.
6. **Native NN через SRAM-only unpack.** Отдельный поздний prefill эксперимент,
   только если profiler доказывает отсутствие expanded DDR traffic.

## Первое практическое разбиение

### Загрузка

- Проверить exact GGUF hash.
- Один раз создать/проверить `Q1_VIP_16x128/v1` sidecar.
- Сохранить packed weights и kernel/network handles resident.
- Не создавать network и не перепаковывать веса на token.
- Не держать одновременно ненужные полные CPU и VIP repack всех весов.

### Prefill

Первая реализация оставляет prefill на CPU. Batched FFN Q1 GEMM исследуется
после decode C1/C2: он лучше амортизирует запуск, но требует отдельно доказать
поддержанные batch/shape и отсутствие expanded DDR traffic.

### Decode

CPU сохраняет управление графом и состоянием. VIP candidate выполняет только
поддержанный Q1 GEMV:

1. F32 activation квантуется в Q8_0.
2. Q8_0 input и Q1 packed weights находятся в resident/mapped buffers.
3. Один flush, trigger/wait, output invalidate/read.
4. CPU продолжает nonlinear/state/attention часть.

Для слоя существуют четыре разных activation groups: две разные
`K=5120`, одна `K=6144` и одна `K=17408`. Их суммарный Q8_0 объём:

```text
2×5,440 + 6,528 + 18,496 = 35,904 байт/слой
64 слоя = 2,297,856 байт/токен
```

Промежуточные F32 outputs всех 497 GEMV дают ориентировочно 15–16 MiB/token,
плюс около 1 MiB logits. По объёму это намного меньше 3.603 GB Q1 weight stream,
но сотни synchronization points могут быть существенны. Поэтому layer-level
fusion и переиспользование resident activation желательны, но их выгода должна
быть измерена, а не объявлена заранее.

## Что оставить на CPU сначала

- embedding lookup;
- RMSNorm и остальные normalization;
- SiLU, sigmoid, softplus и elementwise gating;
- conv1d и GDN recurrent state update;
- RoPE, attention и KV cache;
- residual add/mul, view/permute/concat;
- logits sampling и tokenizer;
- alpha/beta как отдельные VIP calls;
- любой unsupported `MUL_MAT` с обязательной причиной fallback.

## Минимальные доказательные gates

1. **C0:** identity/add custom NBG, exact target execution и CPU golden.
2. **C1:** Q1 pack/unpack и K=128/512/1024 fixtures со всеми sign patterns.
3. **C2:** `5120×17408` и `17408×5120`, не менее 50 resident repetitions,
   correctness, CV ≤2%, host-total минимум на 10% быстрее CPU A55 того же GEMV.
4. **C3:** llama.cpp backend/loader, exact sidecar hash, resident lifetime,
   strict counters и запрет silent CPU fallback.
5. **C4:** exact model, deterministic token agreement, prefill/decode/TTFT,
   no swap/throttle/faults и полная thermal/frequency telemetry.

## Главные blockers

- Не доказан A733-compatible custom OpenCL/VXC/EVIS compiler path.
- Native NBG не доказал поддержку Q1 или group-128 packed coefficients.
- Доступ к нижележащим NN engines через public runtime неизвестен.
- Цена 497 submit/sync boundaries на реальном Q1 kernel ещё не измерена.
- Нельзя предполагать SRAM-only unpack без byte accounting.

## Evidence paths

- `docs/evidence/bonsai-q1-cpu-baselines-2026-08-09.md`;
- `docs/evidence/vip9000-next-capability-probe-2026-08-09.md`;
- `docs/superpowers/specs/2026-08-10-q1-vip9000-backend-design.md`;
- `/home/random/src/llama-prismml/src/models/qwen35.cpp`;
- `/home/random/src/llama-prismml/src/models/delta-net-base.cpp`;
- `/home/random/src/llama-prismml/ggml/src/ggml-cpu/arch/arm/quants.c`;
- `/home/random/src/llama-prismml/ggml/src/ggml-cpu/repack.cpp`.
