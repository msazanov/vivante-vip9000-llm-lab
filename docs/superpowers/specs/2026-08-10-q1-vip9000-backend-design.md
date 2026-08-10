# Проект бинарного Q1 backend llama.cpp для A733/VIP9000

## Цель

Оптимизировать ветку llama.cpp/ggml для бинарной модели Bonsai-27B Q1_0 именно
под Orange Pi с Allwinner A733 и Vivante VIP9000, а не построить абстрактный
NPU-микробенчмарк. Основной критерий — увеличить устойчивую скорость decode
относительно текущего CPU-результата `0.726175 ток/с` без изменения
математических значений весов и без потери качества генерации.

Физический тип — `Q1_VIP_16x128/v1`: перестановка исходных Q1_0-блоков для
прямого packed GEMV на 128-битном PPU/EVIS. Канонический GGUF и логический
`GGML_TYPE_Q1_0` остаются неизменными. Layout сначала существует как
проверяемый sidecar и получает интеграцию с `ggml` только после правильного и
более быстрого выполнения производственных форм матриц на целевой плате.

Конечный deliverable — backend/device `VIP9000` внутри ggml. Он поддерживает
существующий `GGML_OP_MUL_MAT`, распознаёт исходный тип весов и передаёт
вычисление соответствующему packed kernel. Новый логический op не вводится,
пока стандартный контракт `MUL_MAT` достаточен. В steady-state веса не
расширяются в INT8/FP16 в DDR и не перепаковываются на каждом токене.

TQ1_0/TQ2_0 и Ternary-Bonsai исключены из активного milestone. Уже проверенные
факты сохранены в `docs/evidence/ternary-packed-format-audit-2026-08-10.md`, но
не входят в реализацию, capability checks или performance gates этого проекта.

Параллельный CPU-контроль — `Q1_A733_4x4/v1`, заранее построенный вариант уже
выбираемого PrismML repack. Он нужен для измерения стоимости загрузочного
repack, дополнительной памяти и возможного влияния готового layout на CPU.

## Закреплённая граница доказательств

### Модель

- Файл: `Bonsai-27B-Q1_0.gguf`.
- Размер: `3,803,452,480` байт.
- SHA-256:
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Архитектура GGUF: `qwen35`, 64 блока, embedding 5120, FFN 17408.
- В GGUF находятся 851 тензор: 498 `Q1_0` и 353 `F32`.
- Q1_0-тензоры занимают `3,781,877,760` байт.

Наиболее тяжёлые Q1_0-формы:

| Форма GGML (`ne0 × ne1`) | Тензоров | Q1-байт | Доля Q1 |
|---|---:|---:|---:|
| `5120 × 17408` | 128 | 1,604,321,280 | 42.421% |
| `17408 × 5120` | 64 | 802,160,640 | 21.211% |
| `5120 × 248320` | 2 | 357,580,800 | 9.455% |
| `5120 × 10240` | 48 | 353,894,400 | 9.358% |
| остальные | 256 | 663,920,640 | 17.555% |

Первые две формы создают 63.632% всего Q1-потока и являются обязательными
производственными формами первого performance gate.

Первый model-level offload охватывает FFN `gate/up/down` всех 64 слоёв:
2,406,481,920 Q1-байт, или те же 63.632%. Затем добавляются recurrent
`qkv/gate/out` (20.675%), full-attention Q/K/V/O projections (6.238%) и LM head
(4.728%). `token_embd.weight` не является полным decode GEMV и остаётся CPU
lookup. Полное обоснование разбиения сохранено в
`docs/evidence/bonsai-q1-npu-partition-audit-2026-08-10.md`.

### Точная семантика Q1_0

В закреплённом PrismML runtime Q1_0 имеет group size 128. Один блок содержит:

```text
FP16 d       2 байта
sign[128]   16 байт
итого       18 байт на 128 весов = 1.125 бит/вес
```

`d` равен среднему абсолютному значению исходного блока. Каждый восстановленный
вес равен `+d` или `-d`; sign-bit выбирает знак. Это бинарный Q1, а не base-3
ternary и не стандартный `TQ1_0`. Новый layout не меняет `d`, sign-биты или
соответствие логических индексов весам.

Для ARM DOTPROD текущий runtime уже выбирает специализированный
`q1_0_4x4_q8_0` repack. Поэтому новый CPU layout должен сравниваться с этим
путём, а не с медленным скалярным fallback.

### Измеренная производительность и узкое место

| CPU-раздел | Prompt median | Decode median | Нижняя оценка Q1-потока |
|---|---:|---:|---:|
| A76, CPU 6–7, t2 | 1.582870 ток/с | 0.649836 ток/с | 2.457600 GB/s |
| A55, CPU 0–5, t6 | 1.444640 ток/с | 0.726175 ток/с | 2.746305 GB/s |

Нижняя оценка умножает все Q1-байты на decode tok/s и не включает F32-тензоры,
KV, активации и служебный трафик. Более быстрый decode на шести A55 при более
медленном prompt на тех же весах является сильным признаком bandwidth-bound
режима. Это не заменяет отдельный memory-bandwidth benchmark и operator-level
профилирование.

Пик RSS обоих полноценных CPU-запусков около 7.3 GiB при GGUF около 3.8 GB
согласуется с наличием исходного mapping, repack и рабочих буферов, но сам по
себе не доказывает долю каждого потребителя. Температурный троттлинг, swap и
kernel fault в сохранённых интервалах не наблюдались.

### Доказанные и недоказанные возможности VIP9000

Доказано на целевой плате:

- `/dev/vipcore`, VIPLite 2.0, `libNBGlinker.so` и `libVIPhal.so` работают;
- заранее скомпилированный UINT8 NBG выполняется с резидентной сетью и
  буферами;
- 99 измеряемых resident loops дали 2.846 ms host median и 2.803 ms device
  median;
- median host/API/sync разница около 42 us, или 1.50% device time;
- доступны default, host-handle и DMA-BUF buffer APIs, map, flush/invalidate,
  trigger/wait и profiling query;
- продуктовая конфигурация описывает 128-битный PPU/EVIS, 256 shader threads,
  VIP SRAM 512 KiB и восемь нижележащих NN engines.

Пока не доказано:

- что host OpenCL/VXC/EVIS compiler создаёт совместимый с точным A733 runtime
  custom NBG;
- что public API позволяет custom kernel использовать восемь NN engines;
- что один runtime-visible logical core равен одному физическому NN engine;
- что native NN cores принимают packed Q1, group quantization или динамические
  decode-формы;
- что расширение Q1 в INT8 может остаться в VIP SRAM и не породить
  промежуточный DDR-трафик.

Эти неизвестные являются экспериментальными шлюзами, а не основаниями для
заранее положительного вывода.

## Семейство бинарных весов `Q1_VIP_16x128/v1`

### Логический контракт

Layout хранит те же Q1_0-значения, что исходный GGUF. Для каждого 2D
Q1-тензора `ne0` является K-измерением, а `ne1` — выходным M-измерением. Все
498 текущих Q1-тензоров имеют K, кратный 128, и M, кратный 16, поэтому v1 не
нуждается в padding или tail-коде для этой модели. Другие формы fail closed,
если `ne0 % 128 != 0` или `ne1 % 16 != 0`; tail encoding требует новой версии.

Один tile покрывает 16 выходных строк и 128 последовательных K-значений:

```text
sign row 0    16 B
sign row 1    16 B
...
sign row 15   16 B
scale d[16]   32 B, FP16 little-endian
-----------------
tile total   288 B
```

Tile содержит 2048 весов и сохраняет плотность:

```text
288 / 2048 = 0.140625 байт/вес = 1.125 бит/вес
```

Порядок в тензоре:

```text
for m_tile in range(0, ne1, 16):
    for k_block in range(0, ne0, 128):
        write sign[16][16 bytes]
        write d[16][FP16]
```

Размер tile кратен 32 байтам. Каждая sign-строка равна одному 128-битному
векторному чтению PPU. Один блок Q8_0-активации переиспользуется для 16
выходных строк. Конвертер обязан сохранять little-endian FP16 bytes и точный
порядок sign-битов.

### Sidecar-контейнер Q1 weights

Первый формат не получает глобальный GGUF type ID. Type 41 уже занят PrismML
Q1_0, а ранняя регистрация нового GGUF-типа связала бы проверку kernel с
loader, fallback и сериализацией.

Sidecar состоит из:

```text
<model>.q1vip.json   канонический JSON manifest
<model>.q1vip.bin    выровненные tensor payloads
```

Manifest schema `q1-vip-sidecar/v1` содержит ровно:

- исходные filename, byte size и SHA-256;
- semantic source type `GGML Q1_0 type 41, group 128`;
- physical type `Q1_VIP_16x128/v1`;
- byte order `little`, tensor alignment 64 и tile alignment 32;
- converter repository commit и pinned Prism runtime commit;
- для каждого packed tensor: GGUF index, name, `ne0`, `ne1`, source offset,
  sidecar offset, byte size, tile count и SHA-256 payload;
- SHA-256 всего `.bin`.

JSON сериализуется с сортированными ключами, UTF-8, LF и одним завершающим
newline. Payload-тензоры следуют в GGUF index order, начинаются на 64-байтной
границе и не содержат неописанных данных. Padding заполнен нулями и не входит
в tensor payload hash.

Файлы весов, sidecar и vendor binaries остаются вне Git. В репозиторий входят
только source, fixture, manifest schema/example, hashes, команды и результаты.

### Вычислительная семантика

Первый kernel выполняет GEMV `Q1_VIP_16x128 × Q8_0 → FP32`.

Для каждого Q1-блока и четырёх соответствующих Q8_0 subblocks по 32 элемента
kernel вычисляет знаковый dot, применяет исходный Q1 `d` и Q8_0 scale и
аккумулирует выход. Активация общая для 16 выходов. Нельзя молча заменить
FP32 accumulation на FP16: такой вариант является отдельным quality-risk A/B.

Веса и выходные buffers остаются resident. Активация одного token-step
записывается, flush выполняется один раз, затем trigger/wait. Output invalidate
и read учитываются в host latency. Device profiling не заменяет host total.

## Почему не INT8 в DDR

Полное расширение одного блока в `±1 INT8` с scale требует 130 байт вместо 18,
то есть 8.125 против 1.125 бит/вес. FP16 требует 258 байт, или 16.125 бит/вес.
Это увеличивает весовой поток примерно в 7.22 или 14.33 раза и противоречит
ограниченной DDR bandwidth.

Распаковка в INT8 допустима только как отдельный `Q1_VIP_UNPACK_NN/v1`
эксперимент, если intermediate tile остаётся в 512 KiB VIP SRAM и profiler
доказывает отсутствие записи/чтения expanded weights через DDR. Этот путь
может использовать native NN engines и особенно интересен для prefill, но не
является первым decode-кандидатом.

## CPU-контроль `Q1_A733_4x4/v1`

CPU sidecar сохраняет exact output закреплённого PrismML Q1_0 4×4 repack для
ARM DOTPROD. Manifest обязан указывать runtime commit, потому что это внутренний
layout конкретной реализации, а не переносимый GGUF type.

CPU A/B сравнивает:

1. канонический GGUF с runtime repack;
2. тот же runtime с заранее проверенным sidecar;
3. одинаковые affinity, threads, workload, warmup и repetitions.

Основные метрики — model load/repack time, peak RSS, decode/prompt throughput и
качество. Удаление загрузочного repack не считается steady-state ускорением.
Sidecar принимается только если он не ухудшает decode и уменьшает измеренный
startup или memory overhead.

## Архитектура backend llama.cpp/ggml

### Граница интеграции

Реализация добавляет регистрируемый backend/device `VIP9000` по существующим
интерфейсам `ggml_backend_reg`, `ggml_backend_device` и
`ggml_backend_buffer_type`. Ожидаемый build option — `GGML_VIP9000`; точное имя
фиксируется в implementation plan после проверки текущей CMake-конвенции.
Публичный logical graph не меняется.

Device классифицируется как `GGML_BACKEND_DEVICE_TYPE_ACCEL`. Закреплённая
ветка не включает ACCEL в обычную семантику `--gpu-layers`, поэтому выбор
VIP9000 нельзя оставлять неявному GPU offload: loader/device selection получает
отдельный явный путь и печатает выбранные weight buffers. Выдавать VIP9000 за
GPU/IGPU только ради существующего CLI запрещено.

`supports_op` возвращает true только при одновременном выполнении условий:

- op равен `GGML_OP_MUL_MAT`;
- weight source имеет `Q1_0`;
- activation tensor имеет поддержанный graph type/stride; первый контракт —
  contiguous F32, который backend квантует в рабочий `Q8_0`;
- форма, strides, batch и alignment поддержаны конкретным kernel;
- exact sidecar source SHA совпадает с GGUF и packed payload hash проверен;
- compiler/runtime handshake и нужный kernel доступны на target.

Не вводить новый `GGML_OP_*`, если эту семантику можно выразить
`GGML_OP_MUL_MAT`. Физический layout является свойством backend buffer, а не
новым математическим типом tensor. Исходный `GGML_TYPE_Q1_0` остаётся
каноническим и обеспечивает обычный CPU fallback.

### Loader bridge, buffer и lifetime

Одного backend callback недостаточно для sidecar: стандартный loader различает
только mmap GGUF и `set_tensor` каноническими bytes. Поэтому нужен узкий loader
bridge:

1. manifest и source/payload hashes проверяются до allocation;
2. `.q1vip.bin` map сохраняется на весь lifetime модели;
3. GGUF index/name однозначно сопоставляется packed payload;
4. проектный loader bridge (это не стандартный ggml callback) передаёт backend
   признак `already-packed`, не вызывая обычный canonical `set_tensor` повторно;
5. отсутствие/mismatch sidecar выбирает однократный canonical pack или явный
   CPU путь согласно режиму, но никогда не молчаливую интерпретацию bytes.

Backend buffer type предоставляет host-visible mapped arena для `get_base` и
обычного allocator contract, но объявляет `is_host=false`; generic host copy не
должен прочитать packed bytes как canonical Q1. Device handles/physical
addresses хранятся в backend metadata. `init_tensor` связывает tensor с
physical layout, payload, kernel handle, source type и validated hash. Обычный
`set_tensor` принимает канонические Q1 bytes и делает pack один раз; loader
bridge для sidecar копирует/map-ит уже packed payload без второго pack.

`get_tensor` выполняет обратную перестановку и возвращает канонические GGML
bytes. Это обязательно для правильной scheduler copy на CPU при неподдержанной
форме. `cpy_tensor` реализуется явно; CPU `supports_buft` не объявляет VIP
buffer совместимым, иначе CPU может прочитать packed payload как canonical.

Веса создаются и prepare-ятся один раз, затем остаются resident между token
steps. На каждом decode шаге разрешены только подготовка/квантизация небольшой
activation, один input flush, trigger/wait и output invalidate/read.

Запрещены в measured steady-state:

- повторный pack weights;
- полная копия weight tensor из GGUF в backend;
- материализация полной INT8/FP16 weight matrix;
- скрытый transfer поддержанного op обратно на CPU;
- повторный create/prepare network для каждого токена.

Sidecar не должен одновременно удерживать лишние CPU и VIP repack всех весов,
если они не нужны выбранному scheduler. Peak RSS, mapped bytes и фактическое
число копий входят в gate.

### Dispatch, scheduler и fallback

Первый milestone переносит только decode GEMV поддержанных форм. Prefill,
attention, RoPE, normalization и неподдержанные `MUL_MAT` остаются на CPU.
Prefill можно отдельно исследовать через native NN subgraph после доказательства
decode path; это не должно задерживать packed GEMV.

Обычный режим допускает явный CPU fallback с диагностикой причины. До вызова
`supports_op` loader создаёт prevalidated sidecar state. Benchmark режим
`strict-vip` завершает run до allocation/graph при missing/mismatched sidecar и
считает ошибкой любой eligible `Q1_0 × Q8_0 MUL_MAT`, ушедший на CPU, даже если
backend вернул `supports_op=false`. Он сохраняет counters `eligible_q1_ops`,
`vip_ops`, `cpu_fallback_ops`, transfer bytes, kernel id и fallback reason.
Нельзя считать ускорением run, в котором scheduler незаметно исполнил GEMV на
CPU.

Интеграция следует существующей backend registration/loader архитектуре,
включая динамическую регистрацию, если она доступна в закреплённой ветке.
Минимальный patch не меняет общий allocator или scheduler без доказанной
необходимости; сначала backend реализует собственные device, buffer и graph
compute callbacks.

## Путь компиляции и выполнения

### Gate C0 — точный compiler/runtime handshake

В изолированном host-контейнере фиксируются digest AcuityLite/SDK, compiler
versions, target identifier, custom-op API и hashes использованных материалов.
Минимальный identity/add custom op компилируется в NBG и выполняется на A733 с
CPU-golden. Успех доказывает только этот compiler/runtime path.

OpenCL custom operation на PPU — рекомендуемый первый путь. VXC/EVIS может
быть compiler backend того же пути после доказательства. Если compiler не
создаёт совместимый NBG, не подменять эксперимент CPU OpenCL или GPU fallback;
зафиксировать integer statuses и blocker.

### Gate C1 — малые packed fixtures

Standalone harness для Q1 использует детерминированные K=128/512/1024
fixtures, включая все положительные знаки, все отрицательные, чередующиеся
биты, псевдослучайные данные, граничные FP16 scales и нулевые исходные
значения, которые исходный quantizer относит к положительному знаку.

### Gate C2 — производственные формы

Обязательны полные GEMV формы:

- `ne0=5120, ne1=17408`;
- `ne0=17408, ne1=5120`.

Q1-веса берутся из точных тензоров закреплённого GGUF. Они загружаются один
раз, один запуск прогревается, затем выполняется не менее 50 resident
повторений. Сравнение проводится с текущим A55 t6 и A76 t2 CPU kernel на тех же
данных.

### Gate C3 — `ggml` backend integration

Только после C2 backend получает:

- sidecar discovery по exact model SHA;
- loader bridge, mapped lifetime и защита от двойного pack;
- buffer ownership, `set_tensor/get_tensor/cpy_tensor` и canonical CPU copy;
- capability check для `Q1_VIP × Q8_0`;
- graph scheduling для поддержанных GEMV;
- activation quantization, sync и диагностируемый CPU fallback;
- compiled-kernel cache;
- запрещённый fallback при benchmark, чтобы CPU execution нельзя было принять
  за NPU результат.

### Gate C4 — полная модель

Первым кандидат выполняет точный Q1 Bonsai pp512/tg128 и отдельный
deterministic `llama-completion` quality workload. Prefill и decode измеряются
отдельно. Планировщик может оставить prefill на A76, а decode-supported GEMV
перенести на PPU; переключение фаз и неподдержанные операции входят в
wall-clock.

## Контракт корректности

### Конвертер

- `pack → unpack` побайтово восстанавливает каждый source `d` и `qs`;
- logical dequantized tensor совпадает для каждого индекса;
- повторный pack создаёт byte-identical manifest и payload;
- source model и каждый payload проверяются SHA-256;
- malformed dimensions, duplicate names, overlap, unsafe offsets, wrong hash,
  truncated data и unknown schema fail closed до публикации output.

### Operator golden

CPU reference использует закреплённую реализацию Q1_0×Q8_0. PPU kernel обязан:

- содержать только finite значения;
- быть детерминированным между resident repetitions;
- иметь cosine similarity не ниже `0.999999`;
- иметь max absolute error не выше
  `1e-4 * max(1, max(abs(reference)))`.

Порог объявлен до измерений и не ослабляется после результата. Variant с FP16
accumulation получает собственный заранее объявленный порог и не заменяет
основной FP32 candidate.

### Полная модель

Финальный quality gate exact Q1 model — deterministic token agreement с точным
CPU reference при одинаковом model hash, tokenizer, prompt, context, seed и
sampling. Дополнительно сохраняются logits error/cosine на заранее
выбранных шагах, чтобы отличить изменение порядка accumulation от ошибки
layout. Любое расхождение блокирует статус `qualified`, даже если tok/s выше.

## Контракт производительности

### Operator-level

Для каждого CPU/PPU варианта сохраняются индивидуальные samples, min, p10,
median, p90, max и CV. Отдельно измеряются:

- compiler time вне target throughput;
- VIP init, NBG load/link, prepare и buffer allocation;
- sidecar map и resident weight setup;
- activation quantization/map/write/flush;
- trigger, device execution, wait/sync;
- output invalidate/read;
- teardown;
- bytes mapped, copied, flushed, invalidated и expanded.

Кандидат допускается к `ggml`-интеграции, если на обеих производственных формах
он проходит correctness gate, имеет минимум 50 измерений, CV не выше 2% и его
host-total median минимум на 10% меньше CPU A55 median той же GEMV. Device-only
ускорение при более медленном host total не проходит gate.

### Model-level

Текущий CPU decode reference относится только к точному
`Bonsai-27B-Q1_0.gguf`:

```text
median = 0.726175 ток/с
p90    = 0.7263472 ток/с
```

Минимальное доказательство end-to-end ускорения требует:

- не менее одного warmup и пяти measured repetitions;
- candidate decode p10 строго выше `0.7263472 ток/с`;
- candidate median выше `0.726175 ток/с`;
- deterministic token agreement;
- swap delta 0, отсутствие throttle/OOM/accelerator fault;
- сохранённые RSS, CPU/NPU frequencies, affinity, thermals и profiler phases.

Основная цель — максимальный устойчивый decode tok/s, а не прохождение
минимального порога. Prompt throughput и TTFT остаются отдельными метриками.

## Профилирование и тепловая безопасность

Каждый target run использует существующие `profile_command.py` и
`thermal_exec_guard.py`. Emergency ceiling остаётся 85 °C; fan trip 30 °C не
заменяет guard. Профайлер и guard по возможности закрепляются за ядром вне CPU
reference. All-core контроль получает парное измерение overhead collectors.

Фазы custom path:

```text
profiler_start
vip_init
network_create
network_prepare
weights_resident
activation_prepare
activation_quantize
input_flush
warmup_start / warmup_complete
measured_start / measured_complete
output_invalidate
network_finish
vip_destroy
profiler_complete
```

Для каждого измеренного вызова сохраняются host monotonic latency, device
microseconds/cycles, NPU frequency, температурные samples и integer return
statuses, logical op type, physical layout version, kernel id и fallback reason.
NPD/layer dump — отдельный diagnostic A/B и не смешивается с
неинструментированным throughput.

## Ошибки и честная интерпретация

- Compiler error, incompatible NBG, unsupported property, buffer failure,
  timeout, wrong output и CPU fallback сохраняются как самостоятельные failed
  evidence.
- Наличие enum, symbol или marketing core count не считается выполнением.
- Нельзя сообщать LLM tok/s из ShuffleNet или standalone GEMV.
- Нельзя считать PPU kernel использованием восьми NN engines без profiler/API
  evidence.
- Нельзя учитывать expanded INT8 tile как SRAM-only без измеренных DDR bytes.
- Нельзя называть Q1_0 тернарным: это бинарный sign-bit format.
- Proprietary SDK, NBG, model и sidecar payload не попадают в Git.

## Последовательность реализации

1. Закрепить compiler/runtime handshake C0.
2. Реализовать host-side `Q1_VIP_16x128/v1` pack/unpack и adversarial tests.
3. Создать Q1 CPU golden harness и production-shape fixtures.
4. Реализовать Q1 PPU custom-op path и resident profiler C1/C2.
5. Выполнить CPU A/B `Q1_A733_4x4/v1`.
6. Добавить минимальный `VIP9000` backend для прошедшего C2 Q1 kernel и
   `GGML_OP_MUL_MAT` в strict benchmark режиме.
7. Выполнить полную Q1-модель C4 и обновить канонический ledger/график.
8. Только затем исследовать SRAM-only Q1 unpack → native NN и prefill
   partition.

## Критерии приёмки исследования

- Q1 layout воспроизводим, versioned и математически эквивалентен исходному
  GGML Q1_0.
- Target compiler/runtime path доказан либо blocker зафиксирован с точными
  statuses и hashes.
- Производственные GEMV-формы имеют CPU golden, resident host/device timings,
  byte accounting и thermal evidence.
- Никакой speed claim не основан только на device counter.
- `ggml` fallback не маскирует отсутствие NPU execution.
- Backend llama.cpp выполняет поддержанные `MUL_MAT` на доказанном VIP path,
  сообщает counters/fallback и не гонит распакованные полные веса через DDR.
- Полная Q1 Bonsai-27B либо превышает текущий decode gate без потери токенов,
  либо результат честно показывает, почему PPU/native NN path проиграл CPU.
