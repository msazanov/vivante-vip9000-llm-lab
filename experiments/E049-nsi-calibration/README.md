# E049 — калибровка `sunxi-nsi` на A733

Статус: **E049 v4 прошёл safety gate и end-synchronized калибровку на
target**. Все исходные v1/v2/v3 trace сохранены без изменений, но их unit-fit
не считается финальным научным результатом: v1 ошибочно трактовал raw timer
как миллисекунды, а до v4 не было строгого доказательства, что точный объём
load целиком выполнен внутри измеряемого PMU-окна и не выходит за его конец.

Цель — разделить две гипотезы о счётчиках NSI:

1. `bandwidth_*` — объём данных, накопленный за окно `pmu_timer`;
2. `bandwidth_*` — скорость/пропускная способность, уже нормированная на окно.

Это не измерение инференса Bonsai и не попытка разогнать DDR. В workload
используется контролируемый последовательный read-stream из буфера 32/64/128/
256 MiB, чтобы сначала получить калибровочную шкалу самого PMU.

## Что меняет инструмент

`tooling/nsi_calibrate.py` имеет белый список файлов:

- читает `available_pmu`, `pmu_bandwidth*`, `pmu_cmd*`, `pmu_latency*`;
- записывает **только** точный файл `pmu_timer`;
- никогда не ищет, не открывает и не пишет `port_*`.

Перед первым изменением сохраняется числовой `pmu_timer`. Каждое окно
обёрнуто в `timer_window`: запись нового значения проверяется read-back, а в
`finally` старое значение записывается и снова проверяется. Если восстановление
не удалось, выбрасывается `TimerRestoreFailure`, текущий тест немедленно
останавливается, а `failure.json` и append-only `raw.jsonl` сохраняются.

Тепловой предохранитель abort-ит workload при `max thermal >= 85 °C`. Снимки
температур, CPU `scaling_cur_freq` и доступных `devfreq/cur_freq` пишутся в
каждую measurement-запись.

## Host gate

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests \
  -p 'test_nsi_calibrate.py' -v
cc -std=c11 -Wall -Wextra -Werror -pedantic \
  tooling/nsi_sequential_read.c -o /tmp/nsi_sequential_read
```

Тесты включают успешное восстановление fake `pmu_timer`, искусственный отказ
на восстановлении, проверку числового parser и fit/R²-классификацию. В fake
sysfs есть sentinel `port_mode`; его значение проверяется после окна.

## Target protocol

Скопировать на Orange Pi только два исходника/скрипт и собрать read-helper
локально на target:

```bash
ssh -F /dev/null orangepi@192.168.31.117 'mkdir -p /tmp/e049-nsi'
scp -F /dev/null tooling/nsi_calibrate.py \
  tooling/nsi_sequential_read.c orangepi@192.168.31.117:/tmp/e049-nsi/
ssh -F /dev/null orangepi@192.168.31.117 \
  'cc -O2 -std=c11 -Wall -Wextra -Werror -pedantic \
    /tmp/e049-nsi/nsi_sequential_read.c \
    -o /tmp/e049-nsi/nsi_sequential_read'
ssh -tt -F /dev/null orangepi@192.168.31.117 \
  'sudo -S -p "" sh -c "install -d -o root -g root -m 0755 \
    /tmp/e049-nsi-v2-root; install -o root -g root -m 0755 \
    /tmp/e049-nsi/nsi_calibrate.py /tmp/e049-nsi-v2-root/nsi_calibrate.py; \
    install -o root -g root -m 0755 /tmp/e049-nsi/nsi_sequential_read \
    /tmp/e049-nsi-v2-root/nsi_sequential_read"'
```

Запуск требует root только для записи `pmu_timer`. Пароль не должен попадать
в командную строку или trace:

```bash
ssh -tt -F /dev/null orangepi@192.168.31.117 \
  'sudo -S -p "" python3 /tmp/e049-nsi-v2-root/nsi_calibrate.py \
    --output-dir /tmp/e049-nsi/results-<UTC> \
    --sizes-mib 32,64,128,256 \
    --windows-us 100000,250000,500000,1000000 \
    --repetitions 3 --thermal-limit-c 85'
```

После появления запроса `sudo` пароль вводится отдельным stdin-сообщением
SSH-сессии. Каталог результатов сначала забирается с target, затем
проверяется финальный:

```bash
scp -F /dev/null -r orangepi@192.168.31.117:/tmp/e049-nsi/results-<UTC> \
  ./experiments/E049-nsi-calibration/results/
ssh -F /dev/null orangepi@192.168.31.117 \
  'cat /sys/devices/platform/soc@3000000/2020000.nsi-controller/nsi-pmu/hwmon0/pmu_timer'
```

Ожидаемое состояние после штатного запуска — исходное значение (на текущем
target это было `0`). Если проверка не совпала, новые окна запускать нельзя.

## Артефакты

- `raw.jsonl` — неизменяемый по смыслу журнал каждого start/sample/helper/
  measurement/failure события;
- `manifest.json` — конфигурация, начальный PMU и статус;
- `summary.json` — fit/R²/CV/idle-ratio и классификация unit hypothesis;
- `summary.md` — краткая таблица для чтения человеком;
- `failure.json` — traceback при любом остановленном/провальном запуске.

## Интерпретация

В текущем ядре `sunxi_nsi` `pmu_timer` разбирается как decimal и для
`topology_type=2` умножается на частоту NSI после деления частоты на `1e6`;
это source-grounded признак **микросекунд**: `clk_hz / 1_000_000` даёт число
тактов на микросекунду. В device tree A733 указаны
`ia_pmu_data_unit=16`, `ra/ta/cpu_pmu_data_unit=64`. `pmu_bandwidth_rd/wr`
масштабируются соответствующим data-unit, но software path не делит их на
`pmu_timer`, поэтому без этого эксперимента нельзя честно назвать число
«MB/s». `pmu_cmd_*` — сырые command counts, а `pmu_latency_*` — отношение
внутренних счётчиков и не объявлены в sysfs как ns.

Fit помечает сигнал как `rate` или `volume` только если одна модель заметно
лучше (`R²` и зазор); при нулевых/неустойчивых данных результат
`inconclusive`. Это калибровка единиц и причинности, а не доказательство
насыщения LPDDR в полном Bonsai decode.

## Фактический target evidence v1 (отклонён для unit inference)

Target: Orange Pi Zero 3W / A733, kernel `6.6.98-sun60iw2`, NSI module
`sunxi_nsi 1.1.0`, root был нужен только для записи `pmu_timer`. На момент
обоих запусков начальное значение было `pmu_timer=0`.

| Запуск | Конфигурация | Результат | Артефакты |
|---|---|---|---|
| `e049-20260813T224711Z-a2a34180` | 1 повтор, 4 размера × 4 окна | 32 точки, `complete`, ~19 с | `results/20260814T0145Z/` |
| `e049-20260813T224826Z-6f7f8e86` | 3 повтора, 4 размера × 4 окна | 96 точек, `complete`, ~57 с | `results/20260814T0150Z-r3/` |

В каждом случае:

- read-helper предварительно touch-ил буфер и затем читал каждую 64-byte
  cache-line; его `bytes_read` и `elapsed_s` сохранены в `raw.jsonl`;
- были записаны idle и sequential-read samples для всех `32/64/128/256 MiB`
  и `100/250/500/1000 ms`;
- thermal guard имел предел `85 °C`; наибольшая температура серии была около
  `36.3 °C`;
- после финального окна read-back дал `pmu_timer=0`;
- финальный read-only SHA-256 `port_mode` был
  `8ce76146bb47151d932443ee3ede667bd7b1a7e633b8a162a2bdf296aac41aca`, а
  `port_select` — `bb06be5a65269cf405294994077e10ddbd7247d069bd153440e9d749c15883a6`;
  сам инструмент эти control-файлы не открывает.

### Что можно утверждать сейчас

Трёхповторный fit даёт для `pmu_bandwidth(total)` `volume R²=0.397`,
`rate R²=0.207`, для `pmu_bandwidth_rd(total)` — `0.351` против `0.200`.
Порог уверенной классификации не достигнут: оба сигнала помечены
`inconclusive/low`, а `read_cv` примерно `1.19` и `1.38`. Это означает, что
данный bounded stream на idle-системе дал высокую вариативность и **не
позволяет честно объявить MB/s или доказать насыщение DRAM**.

Второй важный результат — счётчики не были постоянно нулевыми после
включения `pmu_timer`: на `total` появились ненулевые read/command значения.
Следовательно, прежнее наблюдение «timer=0 → нулевые counters» объясняется
не отсутствием аппаратного PMU, а выключенным окном. Однако текущий sampling
нужно улучшить (отдельный idle baseline, больше повторов, синхронная граница
таймера и фиксированная affinity), прежде чем использовать его для
разделения memory bandwidth и unpack/compute в Bonsai.

Провальных/остановленных событий в этих двух сериях нет: `raw.jsonl` содержит
`start`, все `sample`, `helper_result`, `measurement`, `measurement_end` и
`complete`; `failure.json` отсутствует именно потому, что abort не сработал.

## Review fixes и target evidence v2

V2 открывает `pmu_timer` через `O_NOFOLLOW|O_CLOEXEC`, проверяет basename,
`fstat` regular attribute и принадлежность пути sysfs mount. Произвольного
`--helper` больше нет: разрешён только root-owned sibling
`nsi_sequential_read` с встроенным SHA-256
`5603b2f4ca8a0f917fdbe5c6716e7182561be1459341e99ab91bd573d7bf27d1`.
Полный target-run был выполнен скриптом staging SHA-256
`34c6b77d4c6e1c5f8b5aeec3d977430b9c8ec8f79bf860244ae320b7c38fd852`;
после запуска host-код получил дополнительную fail-closed проверку ownership
и расчёт thermal maxima по всему raw trace.

Независимый parent-watchdog блокирует `SIGINT/SIGTERM/SIGHUP` во время
save/fork, пересылает сигнал worker-процессу и в любом случае восстанавливает
и проверяет исходный timer. `SIGKILL` нельзя обработать внутри убитого
процесса; adversarial host-тест убивает child через `SIGKILL` и подтверждает,
что живой parent восстановил timer. Если погибнет сам watchdog или вся ОС,
in-process гарантий нет — это остаётся явной границей.

Helper заранее выделяет/touch-ит буфер и ждёт start-gate. После `READY`
оркестратор сначала arm-ит PMU, затем отпускает helper; поэтому helper не
читает измеряемый буфер до начала окна. Для v2 применены окна
`100000/250000/500000/1000000 µs`, то есть те же 100/250/500/1000 ms по
реальному времени.

| Запуск | Результат | Артефакты |
|---|---|---|
| direct `/run` | `Permission denied`, до NSI | отдельный deployment failure, timer не менялся |
| noexec helper | `failed`, partial summary/watchdog trace сохранены | `results/v2-failed-smoke-20260814T0206Z/` |
| v2 smoke | 2 точки, `complete` | `results/v2-smoke-20260814T0208Z/` |
| v2 calibration r3 | 96 точек, `complete` | `results/v2-calibration-20260814T0210Z-r3/` |
| v3 exact-load calibration r3 | 96 точек, `complete` | `results/v3-calibration-20260814T0230Z-r3/` |

На reported aggregate channel с именем `total` (это **не называется суммой**
без отдельного доказательства) результат v2:

- `pmu_bandwidth`: volume `R²=0.997879`, rate `R²=0.000550`;
- `pmu_bandwidth_rd`: volume `R²=0.998503`, rate `R²=0.000666`;
- `pmu_cmd_rd`: volume `R²=0.998515`, rate `R²=0.000668`.

На этапе v2 это выглядело как сильное доказательство семантики **объём за
окно**, а не уже готовой скорости. End-sync review и v4 ниже показали, что
такой вывод был преждевременным. Коэффициент пересчёта в bytes/MB не
объявляется. CV v2 считается по каждой одинаковой ячейке size/window;
большинство read-ячеек имели CV порядка 0.1–2.7%, исключение —
`32 MiB @ 100000 µs` (~29%).

Максимумы пересчитаны по всем telemetry-bearing событиям raw JSONL, а не
только endpoint: CPU `47.492 °C`, DDR `42.346 °C`, GPU `41.912 °C`, NPU
`41.168 °C`. После полного запуска `pmu_timer=0`; SHA control-файлов
`port_mode`/`port_select` совпали с предыдущими read-only снимками.

### Финальная v3 проверка known bytes

Дополнительный self-review обнаружил, что v2 helper делал один byte-load на
cache-line, но называл весь cache-line `bytes_read`. Raw v2 снова сохранён без
изменений, однако итоговой калибровкой считается v3: helper выполняет
volatile 64-bit load каждые 8 байт и явно пишет `load_width_bytes=8`, поэтому
`bytes_read = bytes_per_pass × passes` — точный объём архитектурных load,
выполненных строго между start-gate и остановкой helper.

V3 на reported aggregate channel:

- `pmu_bandwidth`: volume `R²=0.988737`, rate `R²=0.183826`;
- `pmu_bandwidth_rd`: volume `R²=0.989454`, rate `R²=0.181989`;
- `pmu_cmd_rd`: volume `R²=0.988165`, rate `R²=0.183351`.

Точные SHA target-run v3: script
`186c1467b007bdec55559d05ffbc53011e563f02b66f26120db7b7b4fd6bdf29`, helper
`5603b2f4ca8a0f917fdbe5c6716e7182561be1459341e99ab91bd573d7bf27d1`.

До end-sync review v3 интерпретировался как подтверждение накопленного
**объёма за окно**, а не готовых MB/s. V4 ниже не воспроизвёл этот сильный fit,
поэтому данный вывод теперь исторический и не используется как доказанный.
Максимум v3: CPU `56.048 °C`, DDR `49.972 °C`, GPU `52.018 °C`, NPU
`49.476 °C`; после серии `pmu_timer=0`.

## V4: точный объём целиком внутри PMU-окна

Независимый review отклонил v3 как финальное unit evidence: точные
архитектурные load уже считались правильно, но контроллер не доказывал, что
последний байт завершился до конца того же PMU-окна. V4 устраняет именно эту
неопределённость:

1. helper выделяет, first-touch-ит и калибрует буфер **до** `READY`, вне PMU;
2. контроллер arm-ит `pmu_timer` и передаёт helper точный `planned_bytes` и
   абсолютный deadline;
3. helper проверяет deadline каждые 4096 байт, выполняет ровно запланированный
   объём и сообщает `DONE`;
4. контроллер сохраняет `active_us`, затем держит оставшуюся часть окна как
   `idle_tail_us` и читает PMU только после полного programmed window;
5. точка попадает в fit только при `bytes_read == planned_bytes`, полном
   вложении workload и `1.0 <= elapsed/programmed <= 1.01`.

Target использовал root-owned helper SHA-256
`0506e6cff3f22816b3b89c7a334b57af6e71945cb3106c561ed46869af5a5d82`
и Python controller SHA-256
`ebed3b0ab97bded027208273b07617a66506351aff3040683b7da0c8b37c5b55`.
Записывался только `pmu_timer`; `port_*` не открывались и не изменялись.

| Запуск | Конфигурация | End-sync результат | Артефакты |
|---|---|---|---|
| v4 smoke | 32/64 MiB, 100000 µs, 1 повтор | 2/2 read-точки приняты, 0 отклонено | `results/v4-smoke-20260814T003540Z/` |
| v4 calibration r3 | 32/64/128/256 MiB × 100000/250000/500000/1000000 µs × 3 | 48/48 read-точек приняты, 0 отклонено; всего 96 idle/read точек | `results/v4-calibration-20260814T003644Z-r3/` |

Полный raw содержит 1220 событий: 48 `helper_ready`, 48 `helper_result`, 48
`idle_result`, 96 `measurement`, 882 промежуточных telemetry samples и один
`complete`; failure/abort/signal событий нет. Для read-окон:

- `elapsed/programmed`: `1.000002875 … 1.007755417`;
- `active_us`: `14314.083 … 120851.792`;
- положительный `idle_tail_us`: `48563.166 … 985535.833`;
- каждый `helper_result` имеет `status=done`, точное равенство
  `bytes_read == planned_bytes` и `workload_fully_contained=true`.

Для idle-окон `elapsed/programmed` лежит в
`1.000002917 … 1.001884959`. Максимальная температура полного запуска —
`42.284 °C`; thermal abort не сработал. Watchdog восстановил исходный
`pmu_timer=0`, что отдельно проверено read-back после smoke и полного r3.

### Научный итог v4

После строгого ограничения границ прежняя уверенная классификация не
повторилась. На reported aggregate channel `total`:

- `pmu_bandwidth`: volume `R²=0.195942`, active-rate `R²=0.020774`;
- `pmu_bandwidth_rd`: volume `R²=0.251152`, active-rate `R²=0.025262`;
- `pmu_cmd_rd`: volume `R²=0.251848`, active-rate `R²=0.025217`.

Все три результата — `inconclusive/low`. Это не означает, что счётчик
бесполезен: результат показывает, что в полном programmed window он также видит
сильно меняющийся фоновый трафик во время намеренно оставленного idle tail.
Поэтому высокие v2/v3 R² нельзя использовать как доказательство физической
единицы, а v4 raw нельзя называть MB/s без следующей channel/data-unit
калибровки. E049 пока калибрует измерительный прибор и сам по себе ещё не
отделяет RAM bandwidth от Q1 unpack/compute в токене Bonsai.
