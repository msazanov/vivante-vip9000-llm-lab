# E003 — packed Q1 carrier и EVIS Q1_0×Q8_0

Эксперимент начался с переносимого C0-контракта `q1_vip_16x128_c0`, а затем
дошёл до реальной сборки VXC, NBG и выполнения на VIP9000. Базовый кандидат
`q1_vip_q8_evis_1x128.vx` принимает canonical Q1_0 и Q8_0, извлекает знаки
только в EVIS-регистры и не создаёт развёрнутую INT8-копию весов в DDR.
Текущий лучший вариант `q1_vip_q8_evis_4row_vector_accum.vx` дополнительно
переиспользует каждый Q8 chunk для четырёх Q1 rows и векторизует FP32
accumulation без изменения canonical carrier или golden.

Clang-проверка старых `.cl` файлов по-прежнему означает только синтаксис
OpenCL C. Доказательство выполнения относится к отдельно собранному `.vx`
ядру, target NBG и сохранённым логам от 2026-08-11.

## Контракт C0

Kernel получает ровно пять буферов и запускается с global size `16`; каждый
work-item вычисляет одну строку:

1. `uchar packed_signs[256]` — 16 строк по 16 байт, LSB-first; бит `1` означает
   `+d`, бит `0` — `-d`.
2. `float q1_scales[16]` — FP32-плоскость масштаба каждой строки.
3. `uchar q8_values[128]` — четыре группы по 32 байта; байты несут raw signed
   int8 bits и декодируются ровно как `raw <= 127 ? raw : raw - 256`.
4. `float q8_scales[4]` — FP32-масштаб каждой Q8-группы.
5. `float output[16]` — результат `q1_scale[row] * Σ(q8_scale[block] *
   signed_dot(row, block))`.

## Точная конверсия byte planes

`q1_vip.bin` имеет ровно `288 bytes`. Первые `q1_vip[0:256]` байт копируются
без расширения в `packed_signs[256]`. Хвост `q1_vip[256:288]` содержит 16
последовательных little-endian FP16 значений (по 2 bytes); каждое значение
декодируется в FP32 и записывается в `q1_scales[16]`.

`q8.bin` имеет ровно `136 bytes`: это 4 Q8-блоках по 34 байта. В каждом блоке
первые 2 bytes little-endian FP16 декодируются в FP32 `q8_scales[4]`, а следующие 32 bytes
копируются как raw signed-int8 bit patterns в `q8_values[128]`; байт `0x80`
означает `-128`, а не `+128`. `expected_f32.bin` имеет ровно `64 bytes` и
содержит 16 little-endian FP32 результатов.

Итого конверсия scale planes: 16 FP16 → FP32 q1_scales и 4 FP16 → FP32 q8_scales.

`q1_canonical.bin` имеет ровно `288 bytes` (16 Q1_0 blocks по 18 bytes).
Преобразование считается корректным только если
`pack_tensor(q1_canonical.bin, 128, 16) == q1_vip.bin`; CPU golden из
`tooling/q1_vip_golden.py` затем сравнивается с `expected_f32.bin`. В C0 FP32
scale planes — capability probe, а не production FP16-in-payload layout.

Global size равен `16`; local size не фиксируется, и kernel не зависит от
выбранного local size. Запуск с local size `1`, `4` или другим допустимым
значением не меняет ABI или математику.

## Проверка на Orange Pi

NBG собирается или получается только вне Git с Acuity/TIM-VX и не добавляется
в репозиторий. На Orange Pi его загружают через VIPLite. До запуска сохраняют
SHA-256 kernel, compiler/image, NBG, runtime и fixture; каждый VIPLite status
пишут как целое значение вместе с ошибкой. Для каждого запуска проверяют все
64 bytes output против `expected_f32.bin`, а также отсутствие CPU fallback.

Профиль должен отдельно измерять `prepare`, `first`, `steady`, `h2d`, `run`,
`d2h`, включая resident buffers. Сохраняются cold/warm данные и thermals/clocks.
Повторения: `1/10/100/1000`; после warmup для resident-профиля нужны
50 повторов. Нельзя объединять setup с steady-state или выдавать host timing
за device timing.

## EVIS-контракт

На каждый блок `K=128`:

- одна строка Q1_0 занимает ровно `18 bytes`: FP16 scale и 16 packed sign bytes;
- Q8_0 activation занимает ровно `136 bytes`: четыре группы
  `FP16 scale + 32×INT8`;
- `VXC_BitExtract` извлекает 16 последовательных LSB-first знаков в регистр;
- два `VXC_DP16x1` реализуют точное тождество
  `Σ(2b−1)q = 2Σ(bq)−Σq`;
- один work-item считает четыре output-строки и делает один неперекрывающийся
  `float4` write;
- `_viv_uniform block_count` делает то же ядро применимым к `K=128×N`.

Четыре строки на work-item здесь необходимы не только для скорости. В tensor
image ABI `write_imagef` записывает четыре FP32 элемента. Вариант «один
work-item на строку» создавал перекрывающиеся записи `[row..row+3]` и гонку;
после устранения перекрытия 100/100 запусков стали стабильными.

## Что реально прошло на VIP9000

Собраны и запущены NBG для синтетического `16×128` и реального
`blk.0.ffn_gate` Bonsai с `K=5120`, включая tiles на 16 и 1024 выходные строки.

Одинаковая синтетическая задача `16×128`, steady median:

- scalar Q1×F32: 133472 cycles, device 137 µs, run 176,459 µs,
  end-to-end 178,668 µs;
- EVIS packed Q1×Q8: 13331 cycles, device 18 µs, run 67,834 µs,
  end-to-end 73,209 µs;
- ускорение: `10,01×` по циклам, `7,61×` по device time и `2,44×` end-to-end;
- golden совпал, 100/100 запусков стабильны.

Это парный back-to-back A/B на одних Q1 weights и одной Q8 activation. Scalar
получает точную FP32-деквантизацию Q8, EVIS читает исходный packed Q8. Оба
64-byte output побайтно совпали с одним сохранённым независимым golden.

Реальный `blk.0.ffn_gate`, `K=5120`:

- 16 строк: end-to-end 0,599 ms, device 0,535 ms, 100/100 стабильны;
- 1024 строки: end-to-end 24,137 ms, device 23,585 ms, 20/20 стабильны;
- максимальная абсолютная ошибка NPU относительно независимого golden на
  1024 строках — `1,79e−7`; CPU относительно того же golden имеет тот же
  maximum;
- после 1024-row прогона NPU был 35,154 °C при 1008 MHz, признаков троттлинга
  нет.

Это настоящий успех микроядра, но не успех полного offload. Измеренный CPU
выполняет все `17408×5120` за 3,850 ms, тогда как один 1024-row EVIS tile уже
занимает 24,137 ms. Линейная оценка полного EVIS-слоя — около 410,3 ms и явно
не является измерением. Поэтому текущий EVIS shader отклонён как production
decode path; CPU пока остаётся исполнителем Q1 GEMV.

Наглядный [график](../../benchmarks/charts/q1-vip9000-evis-bonsai-20260811.svg),
машинная [сводка](../../benchmarks/results/q1-vip9000-evis-bonsai-20260811/summary.json)
и полный русский [отчёт](../../docs/evidence/q1-vip9000-evis-bonsai-2026-08-11.md)
фиксируют измерения отдельно от проекции.

## E011–E014: переиспользование Q8

Базовое ядро повторно читало один Q8 chunk для каждой из четырёх строк.
E011-v2 переставляет циклы: один Q8 load и одна общая `Σq` используются сразу
четырьмя row dots. На реальном tile `1024×5120` это снизило end-to-end с
`24,137 ms` до `14,150 ms`, то есть дало `1,71×`, с побайтно тем же output.

E014 оставляет целочисленный порядок неизменным, но хранит четыре FP32
accumulators как `float4`. Новый результат — `13,970 ms`, ещё `1,3%` быстрее
E011 и `1,73×` быстрее исходного E003. На микротесте E014 имеет `24 570`
cycles против `25 586` у E011.

Проверены и отклонены три альтернативы:

- direct `0/1 → −1/+1` точен, но vector multiply/add дороже общей `Σq`;
- direct signed Q8 read не даёт измеримого выигрыша: явный `COPY` не является
  текущим bottleneck;
- восемь rows на work-item точны, но дают `42–43 тыс.` cycles против
  `24,6 тыс.` у E014. Вероятная причина — увеличенный live register state;
  без compiler resource/spill report это остаётся объяснением, а не фактом.

Полная русская [сводка](../../docs/evidence/vip9000-q1-q8-row-reuse-2026-08-11.md),
машинный [JSON](../../benchmarks/results/vip9000-q1-q8-row-reuse-20260811-001/summary.json)
и воспроизводимый [график](../../benchmarks/charts/vip9000-q1-q8-row-reuse-20260811-001.svg)
отделяют ускорение NPU kernel от прогноза полного слоя. Новый kernel всё ещё
не проходит production gate: линейная оценка полного NPU-слоя `237,482 ms`
примерно в `61,68×` медленнее измеренного CPU operator `3,850 ms`.

## Сборка graph wrapper

`vxc_nbg_builder.c` загружает собранный `.vxgcSL`, регистрирует kernel с двумя
input tensors и одним output, передаёт официальные I8 DP16 uniforms и выгружает
NBG. Пример для реального 1024×5120 tile:

```bash
./vxc_nbg_builder q1_vip_q8_evis_4row_vector_accum.vxgcSL output.nb \
  q1_vip_q8_evis_4row_vector_accum 1024 5120 q8
```

На плате NBG запускается двумя input carriers:

```bash
/tmp/profiled_viplite_runner --iterations 20 output.nb \
  weights-first1024.q1_0.bin activation.q8_0.bin output.f32.bin
```

Runner не содержит host CPU implementation; ненулевые device cycles и
успешный VIPLite device profiler подтверждают device execution. Возможный
внутренний fallback закрытого vendor runtime этим runner наблюдать не может,
поэтому «нулевой внутренний fallback» отдельно не заявляется.

## Следующий gate: fused native tensor op

Текущая продолженная ветка [E022 fused packed-Q1](../E022-fused-q1/README.md)
проверяет более узкий путь: `BitExtract → DP16x1 → FP16 scale decode →
INT32/FP32 reduce` без expanded weight tensor. На полной форме `M=1024,K=5120`
она дала exact FP32 golden и `13.518 ms` end-to-end, что ниже E014 `13.970 ms`
на той же плате. Это ускорение одного GEMV-тайла; интеграция всего Bonsai
decode ещё не доказана.

EVIS выполняется на программируемом shader/PPU-пути и не загружает native NN
MAC array так, как стандартный UINT8 FullyConnected/Conv. Следующий эксперимент
должен собрать один graph:

1. EVIS распаковывает небольшой Q1 tile в INT8 scratch;
2. native UINT8/I8 FC или Conv считает dot product;
3. промежуточный tile остаётся в on-chip SRAM и не возвращается в DDR;
4. Q8_0 scales по 32 элемента применяются без изменения математики;
5. CPU остаётся эталоном, а offload принимается только при росте полной
   скорости decode без ухудшения golden.

Если NBG planner материализует expanded tile в DDR или native op не сохраняет
Q8_0 scale semantics, кандидат отклоняется независимо от локального ускорения
одного узла.
