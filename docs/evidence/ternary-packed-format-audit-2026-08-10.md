# Отложенный аудит тернарных форматов TQ1_0/TQ2_0

## Статус

Исследование сохранено, но исключено из активного Q1 milestone по решению
владельца проекта от 2026-08-10. Оно не входит в текущий implementation plan,
performance gates или backend capability contract. Возврат к TQ требует
отдельного решения и закрепления точного GGUF-артефакта.

## Проверенные факты закреплённого PrismML runtime

Runtime уже содержит логические типы:

- `GGML_TYPE_TQ1_0 = 34`;
- `GGML_TYPE_TQ2_0 = 35`.

Источники:

- `ggml/include/ggml.h` — type ids;
- `ggml/src/ggml-common.h` — serialized structs;
- `ggml/src/ggml-quants.c` — quant/dequant semantics;
- `ggml/src/ggml-cpu/arch/arm/quants.c` — ARM dot kernels.

Оба формата используют блок `QK_K=256` и рабочую активацию `Q8_K`, а не Q8_0.
`block_q8_K` содержит FP32 scale, 256 signed INT8 значений и 16 INT16 block
sums.

### TQ1_0

```text
qs[48]     48 байт, пять base-3 trit в байте
qh[4]       4 байта, четыре оставшихся trit в байте
FP16 d      2 байта
итого      54 байта на 256 весов = 1.6875 бит/вес
```

`d` равен абсолютному максимуму блока. Значения: `-d`, `0`, `+d`. Base-3 code
сериализуется через `ceil(code * 256 / 243)`; прямое целочисленное base-3
деление не эквивалентно reference decode.

Точный logical K mapping:

- `qs[0..31]`: `K=n*32+m`, `n=0..4`, `m=0..31`;
- `qs[32..47]`: `K=160+n*16+m`, `n=0..4`, `m=0..15`;
- `qh[0..3]`: `K=240+m*4+j`, `m=0..3`, `j=0..3`.

### TQ2_0

```text
qs[64]     64 байта, четыре 2-bit code в байте
FP16 d      2 байта
итого      66 байт на 256 весов = 2.0625 бит/вес
```

Канонический quantizer создаёт `-d`, `0`, `+d`, но serialized 2-bit format
допускает code 3. Reference dequantizer трактует его как `+2d`, поэтому будущий
lossless backend обязан сохранять и корректно выполнять все четыре кода.

## Проверенные кандидаты physical layout

Это зафиксированные исследовательские кандидаты, не активный design contract:

- `TQ1_VIP_16x256`: 16 строк × 54 байта = 864 байта на 4096 весов;
- `TQ2_VIP_16x256`: 16 строк × 66 байт = 1056 байт на 4096 весов.

Оба tile кратны 32 байтам и сохраняют исходную плотность. Возможный PPU kernel
должен декодировать веса внутри регистров/локального tile и не записывать
полную INT8-матрицу в DDR.

## Что ещё не доказано

- Точный файл из `prism-ml/Ternary-Bonsai-27B-gguf` не закреплён локальным
  filename, размером, SHA-256 и inventory tensor types/shapes.
- Нельзя утверждать, что конкретная модель использует TQ1_0 или TQ2_0 до GGUF
  dump точного файла.
- Q1 baseline и результаты Q1 kernel нельзя переносить на тернарную модель.
- В закреплённой ветке TQ dot найден в CPU implementations, но не доказан в
  других accelerator backends.

Возобновление работы начинается с exact artifact pinning и отдельного CPU
baseline; synthetic fixtures не заменяют model-level evidence.
