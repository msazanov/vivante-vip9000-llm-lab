# Memory-first исполнение Bonsai-27B на A733 и VIP9000

Дата: 2026-08-10

Статус: архитектура одобрена владельцем проекта 2026-08-10

Основной workload: одна интерактивная генерация, `batch=1`

## Цель

Увеличить устойчивую скорость decode точной модели
`Bonsai-27B-Q1_0.gguf` относительно CPU reference `0.726175 ток/с` без
изменения логических Q1-весов и без потери качества генерации.

Оптимизация строится вокруг количества байтов DDR на один принятый токен, а
не вокруг номинальных TOPS. A733 CPU, GPU и VIP9000 используют общую память,
поэтому параллельная конкуренция за DDR не считается полезным overlap без
отдельного измерительного доказательства.

## Закреплённый артефакт

- Файл: `Bonsai-27B-Q1_0.gguf`.
- Размер: `3,803,452,480` байт.
- SHA-256:
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Архитектура GGUF: `qwen35`.
- Слоёв: 64, из них 48 recurrent GDN и 16 full-attention.
- Full-attention слои: `3, 7, 11, ..., 63`.
- Hidden size: 5120; FFN: 17408.
- Attention heads / KV heads: 24 / 4; K/V head dimension: 256.
- SSM inner/state/groups/dt-rank: `6144 / 128 / 16 / 48`.
- Vocabulary: 248320.
- Тензоров: 851, из них 498 `Q1_0` и 353 `F32`.
- Q1 payload: `3,781,877,760` байт.
- Q1 block: FP16 scale и 128 sign bits, всего 18 байт на 128 весов,
  или 1.125 бит/вес.
- FFN является dense SwiGLU; MoE и router отсутствуют.

Факты получены прямым read-only разбором закреплённого GGUF и сопоставлением
с `src/models/qwen35.cpp` закреплённой PrismML-ветки `llama.cpp`. Подробный
инвентарь сохранён в
[`docs/evidence/bonsai-q1-npu-partition-audit-2026-08-10.md`](../../evidence/bonsai-q1-npu-partition-audit-2026-08-10.md).

## Бюджет памяти одного decode-токена

Embedding lookup читает одну строку, поэтому полная embedding-матрица не
входит в поток полных decode GEMV. Остальные 497 Q1-проекций требуют не менее
`3,603,087,360` packed Q1 bytes на токен.

| Семейство | Q1 bytes/token | Доля decode Q1 stream | Logical GEMV |
|---|---:|---:|---:|
| FFN `gate/up/down` | 2,406,481,920 | 66.789% | 192 |
| Recurrent projections | 781,885,440 | 21.700% | 240 |
| Full-attention projections | 235,929,600 | 6.548% | 64 |
| LM head | 178,790,400 | 4.962% | 1 |
| **Итого** | **3,603,087,360** | **100%** | **497** |

Суммарная работа этих проекций равна `25,621,954,560 MAC/token`. На текущем
A55 reference packed Q1 lower bound соответствует `2.616 GB/s`. Это сильное
свидетельство memory-bound decode, но не замена operator-level bandwidth
profile.

Дополнительные state-потоки:

- recurrent convolution state: 30720 F32 elements, или 122880 байт на
  recurrent слой;
- GDN state: 786432 F32 elements, или 3145728 байт на recurrent слой;
- full-attention KV append: 2048 F16 elements, или 4096 байт на
  full-attention слой и токен;
- полный KV scan растёт вместе с context length и при длинном контексте может
  стать важнее weight stream.

## Непереговорные memory-first инварианты

1. В steady-state существует одна физическая packed-копия каждого Q1-веса.
2. Полная матрица весов никогда не материализуется в INT8, FP16 или FP32 DDR.
3. Распаковка sign bits допустима только в регистрах, PPU/EVIS execution tile
   или доказанном внутреннем VIP SRAM.
4. Weight buffer создаётся и синхронизируется один раз при загрузке модели.
   На токен разрешены только малые activation/state flush и output invalidate.
5. Network creation, compilation, prepare и weight packing отсутствуют в
   measured steady-state.
6. CPU и NPU не стримят независимые большие weight tensors одновременно без
   A/B evidence, показывающего выигрыш wall-clock при том же качестве.
7. Нельзя считать NPU-ускорением device-only время, скрытый CPU fallback или
   run с неизвестным количеством скопированных/расширенных байтов.

Обычный UINT8 weight tensor нарушает второй инвариант: 18 packed bytes на 128
Q1-весов превращаются примерно в 128 INT8 bytes плюс scales, то есть поток
возрастает более чем в семь раз — ориентировочно до 26 GB на decode-токен.

## Физический layout

Канонический логический тип остаётся `GGML_TYPE_Q1_0`. Physical backend layout
— `Q1_VIP_16x128/v1`:

```text
tile M=16, K=128

packed sign bits:  16 × 16 B = 256 B
FP16 row scales:   16 ×  2 B =  32 B
total:                          288 B
```

Для одного K-блока activation Q8_0 переиспользуется сразу шестнадцатью
выходными строками. Для нескольких независимых проекций с одинаковым input
один NBG получает один activation buffer и несколько packed weight regions.
Carrier имеет datatype `UINT8` только как транспорт байтов; это не семантика
UINT8-весов.

Исследовательский sidecar обязан содержать source model hash, tensor names,
offsets, shapes, layout version и payload hashes. Production loader не держит
одновременно канонический mmap, CPU repack и VIP sidecar всей модели. После
прохождения hardware gates допускается отдельный offline-packed контейнер с
каноническим CPU reconstruction/fallback contract.

## Гетерогенное распределение

### Быстрые A76, CPU 6-7

- graph orchestration и NPU dispatch/wait;
- Q8_0 quantization небольших activation vectors;
- RMSNorm, residuals и короткие elementwise operations;
- SiLU, sigmoid, softplus и gating до включения их в fused NBG;
- RoPE, sampling и tokenization;
- short-context full attention;
- prompt/prefill reference: сохранённый A76 result быстрее A55.

### Шесть A55, CPU 0-5

- доказанный CPU reference для steady decode Q1 GEMV;
- explicit fallback неподдержанных Q1 shapes;
- большие memory-bound GDN state scans после отдельного affinity A/B;
- long-context KV scan после отдельного context-length A/B;
- CPU golden path для каждого VIP operator.

Сохранённый decode A55×6 быстрее A76×2 на 11.747%. Поэтому A55 называются
efficiency cores, но не считаются второстепенными для memory-bound работы.
Backend предусматривает отдельные CPU pools или эквивалентную устойчивую
affinity; он не меняет affinity процесса сотни раз на токен.

### VIP9000

Приоритет offload соответствует весовому потоку:

1. полный FFN одного слоя;
2. recurrent QKV/gate/alpha/beta bundle и отдельный `ssm_out`;
3. full-attention Q/K/V bundle и отдельный output projection;
4. LM head;
5. GDN state и multi-layer recurrent islands только после variable-state gate.

Embedding остаётся CPU lookup. Отдельные alpha/beta GEMV, normalization,
reshape или activation op не запускаются отдельными NPU jobs.

## Рассмотренные архитектуры

### A. Per-op offload — отклонён

Каждый из 497 GEMV создаёт собственные quantize/flush/submit/wait/invalidate
границы. Даже если ориентировочный runtime overhead мал, этот вариант пишет
промежуточные activation в DDR и усложняет доказательство отсутствия hidden
copies. Он остаётся только диагностическим operator harness, но не production
scheduler.

### B. Fused layer subgraphs — первый production milestone

FFN выполняется как один NBG:

```text
input 5120
  ├─ Q1 gate ─┐
  └─ Q1 up   ─┴─ SwiGLU ─ Q1 down ─ output 5120
```

Gate/up и nonlinear/down связаны transient tensors. Host видит только layer
input и final output. Если полный fused FFN не компилируется, допустим
промежуточный gate+up multi-output NBG и отдельный down NBG. Это evidence
milestone, а не конечная архитектура.

Recurrent layer:

```text
attn_norm on CPU/A76
  → NBG [QKV + gate + alpha + beta]
  → CPU GDN conv/state/norm/gating
  → NBG ssm_out
  → CPU residual/post-norm
  → fused FFN NBG
  → CPU residual
```

Full-attention layer:

```text
attn_norm on CPU/A76
  → NBG [Q + K + V]
  → CPU Q/K norm, RoPE, KV append/scan and attention
  → NBG attention output
  → CPU residual/post-norm
  → fused FFN NBG
  → CPU residual
```

Conservative bundling даёт ориентир `497 logical GEMV → 257 NBG submits`.
Полный fused FFN снижает ориентир до 193 submits. Эти числа являются
scheduler design targets, а не измеренной производительностью.

### C. Persistent recurrent islands — целевая архитектура

Модель повторяет последовательность из трёх recurrent слоёв и одного
full-attention слоя. После доказательства custom GDN и `VARIABLE` tensors три
recurrent слоя объединяются в один resident NBG island:

```text
CPU full-attention layer
  → NPU recurrent layer
  → NPU recurrent layer
  → NPU recurrent layer
  → CPU full-attention layer
```

Weights являются `CONSTANT`, GDN state — `VARIABLE`, а межслойные activation —
`TRANSIENT`. На host возвращается только island output. Шестнадцать таких
islands уменьшают число CPU/NPU границ, но state остаётся в shared DDR, пока
profiler не докажет другое.

Target VIPLite пока доказал только precompiled resident UINT8 NBG. Поддержка
large custom graph, variable state, dynamic context, packed Q1 и доступ к
нижним NN engines не предполагается заранее.

## TIM-VX/NBG модель выполнения

TIM-VX upstream предоставляет graph-level custom OpenCL operations и типы
тензоров `INPUT`, `OUTPUT`, `CONSTANT`, `TRANSIENT`, `VARIABLE`. Custom op
layout inference upstream помечен как неподдержанный, поэтому все dimensions,
strides, bit order и layout transforms фиксируются приложением.

На целевой плате исполняется только заранее скомпилированный NBG через
VIPLite. Исходник kernel не загружается target runtime. Поэтому каждый custom
path сначала проходит точный host compiler/runtime handshake для
`CID=0x1000003b`; наличие upstream API само по себе не является target
capability.

## Single-stream scheduling

Autoregressive dependency запрещает pipeline следующего токена до logits и
sampling текущего. Поэтому для `batch=1`:

- CPU не выполняет фиктивную фоновую работу во время зависимого NPU job;
- asynchronous trigger используется только если существует реальная
  независимая работа без дополнительного DDR pressure;
- основная метрика — wall-clock latency одного принятого токена;
- throughput нескольких запросов и batch-amortization не входят в первый gate.

После стабильного packed backend отдельно исследуется lossless speculative
decoding. Это единственный выбранный путь, способный амортизировать один target
weight pass по нескольким принятым позициям без изменения распределения
target-модели. DSpark не включается до измерения его собственного weight
traffic, memory peak, acceptance length и verify cost на A733.

## Профилирование

Каждый operator и model run сохраняет:
- packed weight bytes read or mapped;
- expanded weight DDR bytes, ожидаемое значение — 0;
- activation/state bytes mapped, copied, flushed и invalidated;
- network create/prepare отдельно от warm/steady execution;
- quantize, H2D/cache flush, submit, device, wait, D2H/invalidate и host-total;
- logical GEMV count, NBG submit count и CPU fallback count;
- RSS, mapped model/sidecar bytes, swap delta;
- CPU/NPU frequencies, temperatures, cooling states и kernel faults;
- individual samples, median, p10, p90, CV;
- CPU golden error и model-level deterministic tokens.

`H2D` и `D2H` на shared-DDR SoC означают host write/read вместе с cache
maintenance, а не PCIe transfer. Repeat equality не заменяет independent CPU
golden.

## Correctness и fallback

- Pack/unpack восстанавливает каждый canonical Q1 scale и sign bit побайтово.
- Operator output сравнивается с закреплённым CPU Q1_0×Q8_0 golden.
- Основной FP32 accumulation candidate проходит cosine ≥ `0.999999` и
  max absolute error ≤ `1e-4 * max(1, max(abs(reference)))`.
- Полная модель проходит deterministic token agreement с CPU reference при
  одинаковых model hash, prompt, tokenizer, context, seed и sampling.
- Benchmark `strict-vip` запрещает silent CPU fallback для eligible op.
- Обычный runtime может fallback-нуть только с явной причиной и counters.
- Wrong hash/layout, unsupported shape, compiler/runtime mismatch, allocation
  failure, timeout или wrong output завершают strict run до публикации speed.

## Performance gates

### Operator gate

- точные production shapes `5120×17408` и `17408×5120`;
- минимум 50 resident repetitions после warmup;
- CV ≤ 2%;
- host-total median минимум на 10% быстрее A55 CPU того же GEMV;
- zero expanded-weight DDR traffic;
- zero silent CPU fallback;
- ненулевые VIP device cycles/time;
- correctness gate пройден до speed qualification.

### Model gate

- exact model SHA;
- одна warmup и минимум пять measured repetitions;
- decode p10 строго выше `0.7263472 ток/с`;
- decode median выше `0.726175 ток/с`;
- deterministic token agreement;
- swap delta 0;
- отсутствие throttle, OOM и accelerator faults;
- сохранённые phase, affinity, memory, thermal и frequency evidence.

## Последовательность доказательств

1. Исправить несовместимый host compiler bundle и пройти UINT8 identity/add
   NBG против CPU golden.
2. Выполнить packed Q1 `16×128`, затем `64×128` и `128×128` fixtures.
3. Измерить production Q1 GEMV обеих обязательных FFN shapes.
4. Проверить одну activation с двумя independent matrices.
5. Собрать `gate+up`, затем полный fused FFN с transient intermediates.
6. Подключить FFN-only `ggml` backend в strict mode и измерить exact Bonsai.
7. Добавить recurrent и full-attention projection bundles.
8. Доказать custom GDN и variable state, затем проверить persistent islands.
9. Перенести LM head только после wide-output benchmark.
10. После стабильного single-token backend проверить DSpark speculative decode.

Каждый шаг может закончиться отрицательным результатом. Отрицательный
operator gate сохраняется как evidence и не маскируется изменением workload,
числа токенов или quality threshold.

## Вне текущего scope

- Ternary Bonsai и TQ layouts;
- throughput нескольких одновременных запросов;
- обязательный GPU backend;
- model-wide UINT8/FP16 expansion;
- маркетинговое предположение о восьми NN engines без runtime evidence;
- изменение качества модели ради прохождения speed gate.

## Критерий завершения исследования

Исследование считается успешным только когда точный Bonsai Q1 backend
показывает wall-clock single-stream decode выше CPU reference, читает packed
веса без полной DDR-распаковки, фактически исполняет eligible operations на
VIP9000 и проходит deterministic token agreement. Если VIP path проигрывает,
результатом становится воспроизводимое объяснение по фазам и байтам, а не
утверждение об ускорении.
