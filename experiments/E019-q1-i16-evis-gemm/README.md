# E019 — прямой EVIS INT16 GEMM без public MatrixMultiply

## Зачем нужен эксперимент

E018 подтвердил, что стандартный `vxTensorMatrixMultiplyNode` принимает только
`INT16 DFP8`, но host-exporter падает в закрытой функции
`vxoBinaryGraph_ReSaveInputAndPatchTable` до создания NBG. E019 обходит именно
сломанный узел графа: открытый shader TIM-VX регистрируется как custom EVIS
kernel по уже проверенному на A733 пути E003.

Это ещё не ускорение Bonsai. E019 становится кандидатом только после трёх
последовательных ворот: host создаёт ненулевой NBG, A733 проходит независимый
golden, а end-to-end время одной и полной проекции оказывается ниже CPU/E014.

## Термины

- **GEMM** — матричное умножение. При генерации одного токена `N=1`, поэтому
  форма фактически превращается в **GEMV**: матрица весов умножается на один
  вектор активаций.
- **DFP8** — dynamic fixed point с восемью дробными битами. Физическое INT16
  `256` означает вещественное `1.0`, `-256` означает `-1.0`.
- **EVIS** — программируемые векторные инструкции Vivante внутри VIP9000.
- **NBG** — заранее скомпилированный бинарный граф, который исполняет VIPLite.
- **partial dot** — точная частичная сумма по K=32 или K=128. Полный K=5120
  нельзя писать напрямую в INT16: worst case `655360` переполнится.
- **FP32 reduce** — применение Q1/Q8 scales и сложение partial dots в FP32.
- **golden** — независимый CPU-ответ. Повторяемость одного и того же вывода NPU
  не считается golden.
- **H2D / D2H** — подготовка входов и чтение выходов. На A733 память общая,
  поэтому это map/memcpy/cache sync, а не передача по PCIe.

## Точный контракт

Минимальная форма:

```text
A INT16 DFP8 [K=32, M=1]  physical ±256
B INT16 DFP8 [N=1, K=32]  physical q ∈ [-128, 127]
  → com.vivantecorp.extension.evis.gemm_I16I16toI16
C INT16 DFP8 [N=1, M=1]  physical Σ(sign × q)
```

Tensor layout width-first взят из TIM-VX:
`A={K,M}`, `B={N,K}`, `C={N,M}`. Kernel имеет три tensor-параметра и семь
`INT32` scalar-параметров: transpose/adjoint для A/B и размеры M/K/N.

Прямой K5120 запрещён. Производственный кандидат должен держать packed Q1 в
DDR, распаковывать малый K32/K128 tile в on-chip SRAM, считать partial dot и
сразу выполнять FP32 scale/reduce. Если expanded INT16 weights проливаются в
DDR, главная экономия памяти Q1 исчезает и путь отклоняется.

## Воспроизводимость

Исходный shader из старого upstream оказался только отправной точкой:

- TIM-VX commit: `f792df34f760840a2e37c662a4af18ccbc447be5`;
- `matrixmul_evis.c` SHA-256:
  `acf679dd27e7df1d38b0dd9880d26374d4917ffa1cc379f4e39feddbc90f1e94`;
- `matrixmul_i16.vx` SHA-256:
  `d03edaa81c06cc8fa588d7e969899f0421a22c8e4b1bfc3ac3beb23f82e4e1d8`.

На A733 он исполнялся, но всегда возвращал ноль. Вариант
`matrixmul_i16_coordfix.vx` содержит две минимальные target-проверенные правки:

- swizzle адреса image array заменён с `xyww` на `xywz`, как в TIM-VX после
  обновления internal 22Q3;
- saturation-bit выходного `DP2x8` выключен, иначе отрицательные INT16 суммы
  превращались в `32767`.

SHA-256 проверенного shader source:
`98febd8df3fbdfdbb0a38ebe78844ba7029e86efcd8435a2aea42e96553ff3bc`.

Host gates:

```bash
python3 -m unittest tests.test_q1_i16_evis_gemm_contract -v
python3 -m unittest tests.test_generate_q1_i16_evis_gemm_fixture -v
E019_RUN_HOST_GATE=1 \
  python3 -m unittest tests.test_q1_i16_evis_gemm_host_gate -v
```

Target fixtures создаются так:

```bash
python3 tooling/generate_q1_i16_evis_gemm_fixture.py \
  --output-dir /tmp/e019-fixture --m 1 --k 32 --n 1
```

Четыре adversarial случая содержат `-128`, `-127`, `-1`, `0`, `1`, `126`,
`127`; ожидаемые physical INT16 результаты равны `-248`, `248`, `-256`, `256`.

## Результаты на Orange Pi Zero 3W

Runtime платы: VIPLite `2.0.3.2-AW-2024-08-30`. Оба NBG прошли host-export,
загрузку на VIP9000 и четыре adversarial golden случая. Во всех 100 повторах
выход был повторяемым.

- `M1×K32×N1`: ответы `-248`, `248`, `256`, `-256` совпали бит-в-бит;
  steady wall mean `85.07–90.23 µs`, device mean `35.66–36.61 µs`.
- `M1024×K128×N1`: все 1024 выхода каждого из четырёх случаев совпали
  бит-в-бит; wall mean `6.232–6.457 ms`, device mean `6.088–6.190 ms`.
- Температура после масштабных прогонов: `37.324 °C`; троттлинг не наблюдался.
- Эффективная скорость масштабного INT16 пути: около `0.021 GMAC/s`.

Итог: custom NBG действительно ремонтирует MatrixMultiply, но штатный INT16
EVIS GEMM слишком медленный для Bonsai. H2D/D2H занимают лишь малую часть
времени; основное узкое место — устройство (`~6.13 ms`). Следующий кандидат —
прямой packed-Q1 integer dot без expanded INT16 весов в DDR и без FP32
conversion на каждый элемент.

Машиночитаемые измерения: `results/2026-08-11-a733-target.json`.
