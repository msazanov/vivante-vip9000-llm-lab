# Packed Q1 поверх рабочего UINT8 NBG

Дата: 2026-08-10.

## Решение

Рабочий на A733 путь `UINT8 NBG` используется как транспорт и контейнер
графа, но не как разрешение хранить один бинарный вес в одном байте. В DDR
веса остаются в `Q1_VIP_16x128/v1`: один `UINT8` переносит восемь sign-битов,
а исходный `FP16 d` хранится отдельно для каждой строки и группы K=128.

Стандартный UINT8 MAC не получает новой семантики автоматически. Разбор восьми
битов выполняет custom PPU/OpenCL/EVIS operation внутри NBG. Развёрнутые
`+1/-1` могут существовать только в регистрах или VIP SRAM и не должны
материализоваться в DDR.

Это сохраняет точные веса модели:

```text
Q1 block: FP16 d (2 B) + sign[128] (16 B) = 18 B
weight[j] = d * (bit[j] ? +1 : -1)
```

Для одного `Q1_VIP_16x128` tile:

```text
sign[16][16 B] = 256 B
d[16]          =  32 B
tile           = 288 B = 1.125 bit/weight
```

## Точная вычислительная форма

Для одного Q1-блока строки:

```text
for s in 0..3:
    positive_sum = sum(bit[j] ? q8[j] : 0), j = 32*s .. 32*s+31
    signed_dot   = 2 * positive_sum - sum(q8[j])
    output      += d_q1 * d_q8[s] * signed_dot
```

Здесь `q8[j]` — квантованные целые значения активации. Q8_0 делит K=128 на
четыре subblock по 32 элемента, каждый со своей независимой шкалой `d_q8[s]`.
Q1 `d` меняется для каждой выходной строки и каждого K-блока. Поэтому обычная
per-tensor/per-channel UINT8 quantization не заменяет эту математику: нужен
custom group-wise scale/accumulate либо доказанная per-group поддержка.

## Два независимых способа сократить overhead

### Несколько физических tile в одном workgroup

Формат хранения остаётся `16×128`, а execution tile группирует несколько
соседних физических tile по M. Кандидаты для первого autotune:

| Execution tile | Q1 payload | Expanded INT8, только SRAM | Q8_0 K=128 |
|---|---:|---:|---:|
| `16×128` | 288 B | 2,048 B | 136 B |
| `64×128` | 1,152 B | 8,192 B | 136 B |
| `128×128` | 2,304 B | 16,384 B | 136 B |

`64×128` и `128×128` уменьшают повторные чтения одной активации, число
workgroup и управляющий overhead. Все три варианта должны измеряться, потому
что 512 KiB SRAM не доказывают оптимальное occupancy или фактическое
размещение compiler-ом.

### Несколько проекций с одним входом в одном NBG

Bundling не объединяет зависимые операции; он только устраняет повторные
host submit/sync и повторную загрузку общей активации.

| Слой | Четыре последовательных вызова |
|---|---|
| recurrent GDN | `qkv + attn_gate + alpha + beta`; `ssm_out`; `ffn_gate + ffn_up`; `ffn_down` |
| full attention | `q + k + v`; `attn_output`; `ffn_gate + ffn_up`; `ffn_down` |

Между перечисленными вызовами остаются соответствующие state/attention и FFN
нелинейные операции. Они не включаются в projection bundle.

Если runtime допускает один multi-output invocation с внутренним циклом по
output tile, первый безопасный уровень теоретически снижает число Q1/NBG
submissions на decode token:

```text
48 recurrent layers * 4 calls = 192
16 attention layers * 4 calls = 64
LM head                        =   1
total                          = 257 calls
```

Исходный граф содержит 497 полноценных Q1 GEMV на decode token. Поэтому цель
первого graph-bundle gate — проверить достижимость `497 → 257`, без переноса
nonlinear/state/attention операций и без изменения порядка зависимых
вычислений. Если API запускает отдельный NBG на каждый output tile, 257 не
считается достигнутым: тогда bundling оценивается только по повторному
использованию активации и реальному host-total.

Bundled outputs сами по себе помещаются в 512 KiB:

```text
recurrent input bundle: (10240 + 6144 + 48 + 48) FP32 = 65,920 B
attention QKV bundle:   (12288 + 1024 + 1024) FP32     = 57,344 B
FFN gate+up bundle:     (17408 + 17408) FP32           = 139,264 B
```

Это только арифметическая проверка размера. SRAM residency должна быть
подтверждена profiler-ом и byte accounting; compiler может создать другие
рабочие буферы или spill.

Для одного K=128 chunk packed weights, все FP32 accumulators и один 136-byte
Q8_0 input block требуют без учёта alignment/descriptors/scratch примерно
354.20 KiB для recurrent bundle и 308.13 KiB для attention QKV, поэтому
арифметически помещаются. Полный FFN `gate+up` chunk требует около 748.13 KiB
и не помещается; он обязан стримить paired output tiles. Это ещё не доказывает
реальное SRAM-размещение compiler-ом.

## Кандидаты исполнения

### A. Packed carrier + прямой PPU kernel — основной

Custom operation читает sign bytes как UINT8, извлекает биты в регистрах,
вычисляет signed dot и применяет исходные Q1/Q8 scales. Весовой DDR-поток
остаётся 1.125 бит/вес. Этот путь не требует, чтобы native NN engines понимали
Q1, но требует target-compatible custom NBG compiler.

### B. PPU unpack → VIP SRAM → native UINT8/INT8 NN — условный

PPU разворачивает только текущий execution tile, native NN engine считает
integer partial dot, а PPU применяет group scales. Путь принимается только если
один compiled NBG доказывает одновременно:

- native engine принимает transient coefficients, созданные PPU;
- expanded tile не читается и не пишется через DDR;
- разные `d[m, group]` применяются до окончательной редукции;
- host-total быстрее прямого packed PPU и CPU reference.

Наличие `VIP SRAM preload`, UINT8/INT8 enum или 512 KiB SRAM само по себе этого
не доказывает.

### C. Native UINT8/UINT4 weights — диагностический

Материализованный UINT8 требует 130 B вместо 18 B на группу, то есть примерно
7.22× исходного потока. UINT4 требует 66 B, примерно 3.67×. Эти варианты могут
проверить native engines, но не являются decode-кандидатами до доказательства
per-group scale/correction и выигрыша host-total.

### D. Bit-serial/popcount PPU — резервный packed path

Весовой поток остаётся packed, но activation разбивается на восемь bitplane.
Вариант имеет смысл только если target EVIS/OpenCL предоставляет быстрый
popcount или эффективный LUT lowering; иначе восемь проходов могут проиграть
прямому извлечению битов.

## Экспериментальные шлюзы

1. Собрать identity/add custom OpenCL operation в NBG и выполнить на A733 с
   golden output. Существующий ShuffleNet UINT8 NBG этого не доказывает.
2. Выполнить adversarial `16×128`: all-zero/all-one/alternating/random signs,
   предельные INT8 activation и разные scales.
3. Сравнить direct packed kernels для `16×128`, `64×128`, `128×128` не менее
   чем по 50 resident повторениям.
4. Собрать bundled NBG сначала для двух независимых матриц с общим входом,
   затем для производственных FFN `gate+up`.
5. Отдельно проверить custom-op → native NN transient-coefficient graph.
   Expanded-DDR bytes делают вариант B неуспешным независимо от device time.
6. Пропустить производственные формы `5120×17408` и `17408×5120`, затем
   полный deterministic model gate.

Для каждого варианта сохраняются host total, device time, submit/wait,
прочитанные/записанные/flush/invalidate bytes, NPU frequency, температуры,
CV и CPU fallback counters. Успех требует exact token agreement; ускорение
не покупается изменением sign-битов, scales или порядка зависимых операций.

Сохранение NPU-выигрыша доказывается отдельно: target run должен иметь нулевой
CPU fallback для eligible Q1 ops, ненулевые device cycles/time, синхронные
NPU frequency/thermal samples и минимум 10% меньший host-total на обеих
производственных формах относительно A55 CPU. Использование всех восьми
нижлежащих NN engines — отдельная capability-метрика; оно не выводится из
факта NPU execution и не блокирует прямой PPU-кандидат A.

## Что подтверждает открытый SDK, а что ещё нет

TIM-VX документирует graph-level custom operations из built-in и custom
OpenCL 2.0 kernels. Public types содержат INT8/UINT8 и INT4/UINT4, а internal
headers содержат per-group quantization structures под feature macro. Это
подтверждает направления для compiler probe, но не поддержку конкретным
A733 compiler/runtime:

- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/docs/customized_op.md>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/include/tim/vx/types.h>
- <https://github.com/VeriSilicon/TIM-VX/blob/702715d714277376d7ff466b7b72833ff57564c1/src/tim/vx/internal/include/vsi_nn_tensor.h>

Итоговый порядок: **A → B → D → C**. Вариант A сохраняет Q1 traffic и является
первым production-кандидатом; B имеет максимальный потенциальный выигрыш от
native NN engines, но только после строгого SRAM-only доказательства.
