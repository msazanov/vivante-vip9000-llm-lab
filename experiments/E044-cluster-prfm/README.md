# E044 — cluster-aware PRFM для Q1 на A733

Дата: 2026-08-13 (UTC)
Плата: Orange Pi Zero 3W / Allwinner A733, `aarch64`, Linux `6.6.98-sun60iw2`
Модель: `Bonsai-27B-Q1_0.gguf`
Модельный SHA-256: `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`

Статус: **короткий screen — PROMISING; полный production gate — UNQUALIFIED**.

E044 проверяет две версии планирования загрузок в CPU-ядре `Q1_0 × Q8_0`:

* **A / split** — разделённый путь, в котором prefetch используется только
  для части рабочих ядер;
* **B / big-only PRFM** — тот же cluster-aware путь с более агрессивным
  `PRFM` только на двух быстрых A76 (CPU6–7), без изменения формата модели.

`PRFM` — это подсказка AArch64-процессору заранее подать строку памяти в
  cache. Это не вычисление и не увеличение пропускной способности DDR. Поэтому
  E044 должен считаться успешным только если короткий выигрыш сохраняется в
  полном запуске и не ломает качество или стабильность платы.

## Что измерялось

Все короткие прогоны были одной последовательностью, чтобы видеть дрейф
платы:

```text
native → A/split → B/big-only PRFM → native
```

Команда screen использовала `llama-bench`, `p=0`, `n=8`, `r=1`, `delay=1`,
`t=8`, `C=0xff`, `cpu-strict=0`, `poll=50`, flash attention, `ngl=0`, mmap,
`b=512`, `ub=512`, F16 KV и stock DDR. Профиль собирался через
`profile_command.py` с sample каждые 100 ms, поверх него работал thermal guard
с лимитом 85 °C. CPU policy временно переводились
`ondemand → performance → ondemand` и после каждого запуска восстанавливались.

`full` gate был отдельным более строгим запуском: `p=0`, `n=32`, `r=3` с теми
же параметрами. Для него требовались завершённый throughput, golden/quality
проверка и отсутствие reset/thermal abort.

## Screen: результат скорости

| Порядок | Вариант | Скорость, токенов/с | Exit | Thermal abort |
|---:|---|---:|---:|---|
| 1 | native | 0.944547 | 0 | нет |
| 2 | A / split | 0.968064 | 0 | нет |
| 3 | **B / big-only PRFM** | **1.018629** | 0 | нет |
| 4 | native | 0.955468 | 0 | нет |

Оценка ranking screen:

* B против A: `+0.050565 токенов/с`, `+5.2233%`;
* B против двух native повторов: geometric-mean `+7.2250%`;
* максимальная температура CPU big/little среди screen — `65.224 °C`;
  DDR — `54.808 °C`, GPU — `58.962 °C`, NPU — `53.196 °C`.

Это **не** означает, что B уже даёт стабильный `1.018629 токенов/с` в
рабочем полном декодировании: это одна короткая серия `n=8/r=1`.

График с увеличенной, но явно подписанной шкалой Y:

* [e044_screen_throughput.png](generated/e044_screen_throughput.png)
* [e044_screen_throughput.svg](generated/e044_screen_throughput.svg)

Данные: [screen_results.csv](data/screen_results.csv).

## Full gate: safety/reset результат

Три sustained-load попытки не дошли до строки throughput и не запустили
quality/golden. Плата перезагрузилась во время работы, причём thermal guard не
успел завершить child штатно:

| Запуск | Время до потери платы | CPU max | Результат |
|---|---:|---:|---|
| A / split (`041418`) | 21.515 с | 48.546 °C | reset, no throughput |
| B / big-only PRFM (`043043`) | 36.125 с | 64.914 °C | reset, no throughput |
| B / big-only PRFM (`045646`) | 35.510 с | 65.100 °C | reset, no throughput |

Во всех трёх попытках:

* лимит guard `85 °C` **не достигнут**, событие `event=abort` отсутствует;
* зафиксированное CPU frequency оставалось на configured maximum;
* нет `metadata.json`/`validation.json` с успешным завершением benchmark и
  нет quality SHA;
* kernel/pstore/journal не сохранили однозначный reset reason. Возможны
  watchdog, PMIC/питание или другая нестабильность платформы; связывать reset
  именно с PRFM пока нельзя — один A-run тоже завершился reset.

График строится по компактной агрегации thermal trace (максимум за 1 с), а не
по огромным исходным JSONL:

* [e044_full_reset_temperature.png](generated/e044_full_reset_temperature.png)
* [e044_full_reset_temperature.svg](generated/e044_full_reset_temperature.svg)

Данные: [full_temperature_1s.csv](data/full_temperature_1s.csv).

На графике крест `RESET` — это последняя валидная sample перед потерей платы;
это не измеренная температура reset-механизма. Надпись `нет tok/s / quality`
означает, что скорость и качество для full gate не были получены.

Решение для production: **E044 не интегрировать и не объявлять полным
ускорением до отдельного расследования стабильности платы и повторного
qualified full gate**. Это решение означает `UNQUALIFIED`, а не измеренный
регресс: полного значения скорости не существует. Короткий результат B
сохранить как кандидат для
следующего controlled эксперимента.

## Воспроизводимость графиков и проверка данных

В репозитории сохранены только нормализованные данные, manifest и генератор;
сырые thermal/telemetry trace остаются в staging-каталогах эксперимента.
Manifest фиксирует их абсолютные пути и SHA-256:

* [manifest.json](data/manifest.json)
* [plot_e044.py](plot_e044.py)

Проверка схемы, cardinality и safety invariants:

```bash
MPLCONFIGDIR=/tmp/e044-mpl python3 experiments/E044-cluster-prfm/plot_e044.py --check
```

Ожидаемый вывод:

```text
E044 data/schema checks: PASS (screen=4 rows, full=95 rows, 3 reset runs, CPU < 85 C)
```

Перегенерация PNG/SVG:

```bash
MPLCONFIGDIR=/tmp/e044-mpl \
  python3 experiments/E044-cluster-prfm/plot_e044.py \
  --output-dir experiments/E044-cluster-prfm/generated
```

Повторная генерация в чистый каталог дала одинаковые SHA-256 всех четырёх
файлов — генератор детерминированный. Для построения требуется только уже
имеющийся `matplotlib`; формат и данные не зависят от сети.

Проверенные SHA-256 графиков:

```text
d73aa4ba47ca10531e0f14b11d8259aa11511195ec5c2b539fa1e65854aa324e  e044_full_reset_temperature.png
da1d4161daa15c2631d71eaead629053157c9802194bc31499eaa1a766a73821  e044_full_reset_temperature.svg
6ca77ec5113df1de3c489f2bce313a49594a5c746bf06035327b8d24981f7c0a  e044_screen_throughput.png
3be8bfaf95827a78d4d6f3ce4e950e20bf43b200d96cbe6ee6c989e03b6be9b7  e044_screen_throughput.svg
```

## Provenance

Screen отчёт: `/tmp/e044-screen-report.md`, SHA-256
`1b158a84a5ab942efcba1e31d41436a5b6f1885b4a94c9703e3ea27481a1c38a`.
Full crash/reset отчёт: `/tmp/e044-full-crash-report.md`, SHA-256
`bf66913322cc1ce3ad60d16455cde33c45e1e27d75dfd8169a3854086ec816e7`.

Общие исходные факты screen:

* llama.cpp source commit: `38c66ad0241da4f9fcce541cda8edc219086cec5`;
* E044 `repack.cpp` artifact SHA-256:
  `3152af996a71ac874866f12ce39b7f374be333bd667ab8c7b2d095102d8b6c5f`;
* B benchmark ELF SHA-256:
  `a21ec77b193a455640d3ac42023a160e95c7b2d7d9c949c1904dd9c78ad04033`.

Полные source paths и SHA исходных trace указаны в
[data/manifest.json](data/manifest.json). Модель, vendor SDK, NBG, ELF и
гигантские traces намеренно не коммитятся.
