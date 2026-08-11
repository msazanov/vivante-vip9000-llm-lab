# E009 — packed Q1 → нативный FullyConnected на VIP9000

## Что проверяет эксперимент

`q1_unpack_u8_evis.vx` читает канонический Q1_0 carrier: 18 байт на строку
`K=128` (FP16 scale и 128 знаковых битов). Внутри одного NBG он превращает
биты в временные UINT8-коды `0/2`. С affine-параметрами `scale=1,
zeroPoint=1` эти коды означают точные веса `−1/+1`.

Временный tensor имеет форму `[128, rows]`, не является graph input/output и
передаётся в стандартный `vxFullyConnectedLayer`. Активация имеет форму
`UINT8 [128,1]`, `zeroPoint=128`; результат capability-теста —
`INT16 [rows,1]`.

```text
packed Q1 (DDR)
    │
    ▼
EVIS/SH: биты → UINT8 0/2
    │ virtual tensor
    ▼
NT: TensorTranspose
    │
    ▼
native GEMM: UINT8 × UINT8 → INT16
```

Здесь:

- **EVIS/SH** — программируемое векторное ядро VIP9000;
- **NT / TensorTranspose** — перестановка осей матрицы перед вычислением;
- **GEMM** — нативное аппаратное умножение матриц;
- **virtual tensor** — внутренний буфер графа, недоступный CPU;
- **NBG** — скомпилированный целиком граф для VIPLite.

## Сборка

```text
./fused_q1_fc_nbg_builder KERNEL.vxgcSL OUTPUT.nb ROWS
```

VXC compiler собирает `.vx` в `.vxgcSL`. Builder регистрирует custom kernel,
соединяет его virtual output со стандартным FC и вызывает `vxGenerateNBG`.
В Git не добавляются проприетарные `.vxgcSL`, `.gcPGM` и `.nb`.
Target runner использует `repeat_cap=100`: не более 100 повторов на процесс,
потому что для tiny NBG отдельно воспроизведено зависание vendor runtime на
241-м запуске.

## Доказанный результат на Orange Pi Zero 3W

После замены динамических масок `VXC_BitExtract` на статические raw-маски:

- unpack-only: `2048/2048` байт совпали с независимым CPU reference;
- R16 и R64: output побайтно совпал с golden;
- 99/99 steady-повторов стабильны;
- compiler plan содержит `SH → NT → GEMM`, то есть нативный матричный блок
  действительно используется;
- VIP-SRAM peak: 4352 байта для R16 и 16640 байт для R64;
- NSI PMU показал, что рост внутреннего expanded tensor на 6144 байта не дал
  сопоставимого роста DDR traffic: добавилось лишь около 827 байт чтения на
  запуск. Это подтверждает on-chip размещение промежуточных данных.

Но путь отклонён по скорости. Для одинаковой формы R64:

| Вариант | device, µs | E2E, µs | cycles |
|---|---:|---:|---:|
| Прямой EVIS E003 | 35 | 74,041 | 28514 |
| E009 EVIS → FC | 49 | 101,291 | 42472 |

E009 медленнее прямого EVIS в `1,37×` по end-to-end. Главный наблюдаемый
лишний этап — `TensorTranspose`. Поэтому четырёхчастный scale-aware Q8_0 путь
не реализуется поверх этого графа: четыре FC только увеличили бы overhead.

Следующий gate — E010: подать тот же virtual tensor в `MatrixMultiply` либо
эквивалентный `Conv1×1` с layout, который не требует `NT`. Кандидат должен
сохранить exact golden, нативный GEMM и on-chip scratch, а на R64 быть быстрее
`74,041 µs` E2E и `35 µs` device.

Первый E010 probe использует тот же builder с режимом
`q1-as-batched-input`:

```text
./fused_q1_fc_nbg_builder KERNEL.vxgcSL OUTPUT.nb 64 q1-as-batched-input
```

Он меняет роли операндов FC: Q1 `[128,64]` становится 64 batched inputs, а
активация `[128,1]` — одним weight-вектором. Линейный output `[1,64]` содержит
те же INT16 значения. Цель — убрать transpose Q1 или хотя бы сократить его с
8192 до 128 байт. Успех определяется только compiler plan и target A/B.

Полные метрики и термины сохранены в
`benchmarks/results/vip9000-fused-q1-fc-20260811-001/summary.json` и
`docs/evidence/vip9000-fused-q1-native-fc-2026-08-11.md`.
