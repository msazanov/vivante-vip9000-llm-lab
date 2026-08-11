# E022 — fused packed-Q1 dot на VIP9000

Это отдельная исследовательская ветка `codex/e022-fused-q1`. Она проверяет
главную идею для Bonsai: читать Q1-знаки из packed carrier и сразу участвовать
ими в `DP16x1`, не создавая в DDR развёрнутый `+1/-1` или INT8-тензор весов.

## Что именно делает ядро

Для каждого 16-элементного фрагмента используется точное тождество:

```text
sum((2*b - 1) * q) = 2 * sum(b*q) - sum(q)
```

`b` — извлечённый `0/1`-бит Q1, `q` — signed INT8 активация Q8. `VXC_BitExtract`
выдаёт биты только в EVIS-регистры, затем два `DP16x1` считают выбранную и
общую сумму в INT32. Материализации `+1/-1` в памяти нет.

В carrier E022:

- Q1 super-block на 128 значений: 16 packed sign bytes + 2 байта под FP16 scale;
- Q8 block на 32 значения: 2 байта под FP16 scale + 32 raw INT8 bytes;
- текущий integer gate намеренно оставляет scale bytes нулевыми и проверяет
  только exact integer core;
- для `K=5120` это 40 Q1 super-blocks (720 bytes на строку) и 160 Q8 blocks
  (5440 bytes на activation).

## Термины профиля

- **golden** — независимый CPU-расчёт, с которым сравнивается каждый NPU output;
- **H2D** — host-to-device: map/copy и синхронизация входного carrier перед
  запуском;
- **run** — полный вызов `VIPLite` от enqueue до возврата;
- **D2H** — device-to-host: чтение output carrier обратно;
- **device** — timestamp самого VIP9000, а не время CPU-драйвера;
- **cycles** — аппаратный счётчик VIP9000 для участка выполнения.

## Target-результаты

Плата: Orange Pi Zero 3W, A733, VIP9000, VIPLite `2.0.3.2-AW-2024-08-30`.
После финального scale-aware запуска температура NPU была `33.728°C`, DDR
`34.286°C`, skin `30.646°C`; текущие частоты в момент снимка: big `780 MHz`, little
`1.794 GHz`. Признаков температурного троттлинга в этой серии не было.

### Малый gate `M=4,K=128`

Все пять паттернов прошли побитово, 100/100 повторов каждого были стабильны.

| вариант | run, среднее на вызов | device | cycles | смысл |
|---|---:|---:|---:|---|
| одна строка на work-item | 59.161 мкс | 11.214 мкс | 4 601 | контроль |
| четыре строки + общий Q8 load | 60.730 мкс | 13.602 мкс | 6 964 | почти та же цена вызова, 4 результата |

Вариант rows4 не ускоряет один вызов, но даёт примерно `3.9×` больше строк на
тот же `run` и примерно `2.6×` меньше device cycles на строку. Это именно
переиспользование одной активации, а не утверждение об ускорении всего токена.

### Полная длина `K=5120`, `M=4`

| паттерн | golden | run | device | cycles |
|---|---|---:|---:|---:|
| zeros | exact, 4/4 | 243.017 мкс | 191.020 мкс | 184 909 |
| ones | exact, 4/4 | 294.891 мкс | 207.758 мкс | 185 352 |
| alternating | exact, 4/4 | 240.486 мкс | 191.010 мкс | 184 660 |
| mixed | exact, 4/4 | 229.725 мкс | 190.071 мкс | 184 498 |
| random | exact, 4/4 | 233.626 мкс | 188.758 мкс | 184 898 |

В каждой строке таблицы `repeat_equal=1` для всех 99 steady-итераций после
первого запуска. Разброс `ones` объясняется общей нагрузкой платы; это не
ошибка golden.

### Полный tile `M=1024,K=5120`

Это первый запуск размера, близкого к одной большой Bonsai projection.
Матрица Q1 во внешней памяти занимает `737 280 bytes` вместо
`1024×5120 = 5 242 880 bytes` развёрнутого INT8 carrier.

| участок | среднее |
|---|---:|
| H2D | 251.187 мкс |
| run | 9.130931 мс |
| D2H | 138.145 мкс |
| device | 8.941404 мс |
| H2D + run + D2H | 9.520262 мс |
| cycles | 8 922 720 |

`1024/1024` значений совпали с integer golden, `repeat_equal=1` в 100/100
запусках. Для ориентира E014 в документации проекта имеет около `13.970 мс`
на близкой форме. Это сравнение integer core было диагностическим: E014 уже
применяет scale semantics и выдаёт FP32, а raw E022 использовал нулевые scale
bytes. Поэтому `9.52 мс` — измеренная скорость integer микроядра, а не
окончательная скорость Bonsai.

### Scale-aware gate `M=1024,K=5120`

Следующий вариант оставляет carrier физически `UINT8`, но декодирует два scale
bytes без ручного pack:

```text
UINT8 read → bit-preserving COPY → vxc_half8 → DP4x4 FP16→FP32
```

Q1 scale применяется после четырёх Q8 partial dots каждого 128-блока. На пяти
adversarial-паттернах `M=4,K=5120` получено `max_abs=0.0` и 100/100 стабильных
повторов. На полной форме:

| участок | среднее |
|---|---:|
| H2D | 235.890 мкс |
| run | 13.142393 мс |
| D2H | 139.772 мкс |
| device | 12.894131 мс |
| H2D + run + D2H | **13.518054 мс** |
| cycles | 12 828 653 |

Все `1024/1024` FP32 результатов совпали с независимым scale-aware golden с
`max_abs=0.0`; `repeat_equal=1` в 100/100. После серии: NPU `33.232°C`, DDR
`33.108°C`, skin `31.427°C`; падения cycles или температурного троттлинга не
наблюдалось.

Относительно E014 (`13.970 мс`) это `~3.2%` end-to-end выигрыш на одной
проекционной форме при том же packed-Q1/Q8 представлении. Прямой повтор без
обёртки generic runner дал `13.315 мс`; оба прогона ниже E014. Это первый E022
результат, который можно классифицировать как `accelerates` для проверенного
GEMV-тайла; он всё ещё не означает автоматически такой же прирост токенов
Bonsai, пока не подключён к llama.cpp и не проверены все формы/планировщик.

## Что пока не доказано

1. `DP16x2_b` не интегрирован: в SDK нет точной документации lane mapping,
   перебранные uniforms давали нули или неверные значения. Он остаётся отдельной
   диагностической веткой.
2. Выход scale-aware gate сейчас хранится в безопасных четырёх-float слотах
   (lane 0 каждого слота),
   потому что частичная запись `vxc_int4` в `INT32` carrier на этом SDK усекала
   знак. Это диагностический ABI; после production integration можно уплотнить
   output.
3. Это custom EVIS ядро, а не штатный native UINT8 FC. Нельзя переносить его
   `device` напрямую на токены Bonsai без проверки всех projection shapes,
   scale layout и CPU/NPU scheduling.

## Воспроизведение

Host export (нужен локальный Docker-образ Vivante SDK):

```bash
tooling/run_e022_fused_host_gate.sh --output-dir /tmp/e022-host --rows 4
tooling/run_e022_fused_rows4_host_gate.sh --output-dir /tmp/e022-rows4-host --rows 4
tooling/run_e022_fused_rows4_k5120_host_gate.sh \
  --output-dir /tmp/e022-k5120-host --rows 1024
tooling/run_e022_fused_rows4_k5120_scales_host_gate.sh \
  --output-dir /tmp/e022-scaled-host --rows 1024
```

Carrier fixture:

```bash
tooling/generate_e022_fused_carrier_fixture.py \
  --output-dir /tmp/e022-fixture --m 4 --k 128 --pattern mixed
tooling/generate_e022_fused_long_carrier_fixture.py \
  --output-dir /tmp/e022-k5120-fixture --m 1024 --k 5120 --pattern mixed
tooling/generate_e022_fused_scaled_fixture.py \
  --output-dir /tmp/e022-scaled-fixture --m 1024 --k 5120 --pattern mixed
```

Generic target runner сохраняет `profile.log`, `thermal_before/after` и
проверяет golden автоматически:

```bash
tooling/run_e022_fused_target.sh \
  --output-dir /tmp/e022-target \
  --network /tmp/e022-scaled-host/e022_q1_fused_dp16x1_rows4_m4_k5120_scales_f32.nb \
  --weights /tmp/e022-scaled-fixture/weights_q1.bin \
  --activation /tmp/e022-scaled-fixture/activation_q8.bin \
  --golden /tmp/e022-scaled-fixture/golden_f32.json --dtype f32
```

Профили и golden сохранены в `results/`. Машиночитаемая сводка —
`results/target_summary.json`.
