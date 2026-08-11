# E023 — безопасный seam GGML Q1_0 → VIP9000

## Цель

Проверить, можно ли подключить доказанное E022 fused-Q1 ядро к реальному
`ggml/llama.cpp` без изменения семантики GGUF и без разворачивания Q1-весов в
полный INT8/FP16 тензор. Критерий успеха — сначала exact tile-golden, затем
измерение полной Bonsai projection и только после этого сравнение decode
токенов/с. Само наличие NPU-ядра не считается ускорением модели.

## Факты, на которых строится дизайн

- Стандартный `GGML_TYPE_Q1_0 = 41` имеет блок 18 байт:
  `[fp16 scale][16 packed LSB-first sign bytes]`.
- E022 target carrier имеет те же 18 байт, но порядок:
  `[16 sign bytes][fp16 scale]`.
- Q8_0 carrier совместим: `[fp16 scale][32 signed INT8 values]`.
- E022 target gate доказан только для `K=5120`, tiles до `M=1024`, с четырьмя
  строками на work-item и диагностическим четырёхслотовым output ABI.
- Bonsai `ffn_gate/up` имеют `K=5120,M=17408`; `ffn_down` имеет
  `K=17408,M=5120`. Поэтому E022 нельзя автоматически применять ко всем
  `MUL_MAT`.

## Архитектура

1. `ggml` продолжает владеть GGUF, графом, scheduler и CPU fallback.
2. Адаптер делает только byte-reorder каждого Q1 блока `[d][qs] → [qs][d]`
   в resident UINT8 buffer. Объём данных не меняется: 18 байт на 128 весов.
3. Activation F32 квантизуется стандартным Q8_0 carrier; его layout напрямую
   читается E022.
4. VIP backend принимает только явно поддержанный `MUL_MAT` tile. Для любого
   другого K/M/stride/type он возвращает `false`, и scheduler оставляет узел
   на CPU.
5. После target exact gate output lane 0 собирается в обычный GGML F32 tensor;
   четыре-float диагностические слоты не считать production ABI.
6. Веса и адаптированный carrier кэшируются. Перестановка не должна
   выполняться на каждом токене, а NBG не должен компилироваться во время
   decode.

## Фазы

### Фаза A — host contract

- Проверить reorder для одного и нескольких блоков, input immutability и
  отсутствие expanded tensor.
- Проверить identity `2*sum(bits*q)-sum(q)` на adversarial Q8 значениях.
- Проверить conservative `supports_tile` contract.

### Фаза B — target tile

- Извлечь настоящий `blk.0.ffn_gate.weight` из pinned GGUF, сделать только
  byte-reorder и подать его в E022 NBG.
- Сравнить output с независимым CPU golden на `M=4`, затем `M=1024`.
- Сохранить H2D/run/device/D2H/cycles, golden, repeat_equal и thermals.

### Фаза C — projection/full-model gate

- Проверить все 17 tiles `M=17408` для `ffn_gate` и `ffn_up`.
- Отдельно разработать kernel для `K=17408` (`ffn_down`); текущий E022
  kernel к нему не применять.
- Интегрировать только если суммарная projection latency и CPU/NPU copies
  улучшают pinned CPU reference.
- После этого выполнить deterministic decode и сравнить token sequence,
  logits tolerance и steady-state tok/s.

## Rejection rules

- Нельзя переименовывать или менять `GGML_TYPE_Q1_0`.
- Нельзя передавать GGUF bytes в E022 без reorder.
- Нельзя выделять expanded `+1/-1`, INT8 или FP16 copy весов в DDR.
- Нельзя объявлять NPU ускорением по одному `device` timestamp: учитываются
  conversion, H2D, launch, D2H и scheduler overhead.
- Если full projection медленнее CPU, E023 остаётся диагностическим backend
  candidate, а production decode остаётся на оптимизированном CPU.

## Проверка качества

Каждая фаза сравнивается с независимым CPU golden. Для full model обязательны
детерминированные sampling parameters и совпадение токенов; любое расхождение
без объяснимой численной нормы классифицируется как `rejected`, даже если
latency лучше.
