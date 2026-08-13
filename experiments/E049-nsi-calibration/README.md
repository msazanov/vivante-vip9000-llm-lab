# E049 — калибровка `sunxi-nsi` на A733

Статус: **E049 v2 прошёл safety gate и новую bounded-калибровку на target**.
Два исходных v1 trace сохранены без изменений, но их unit-fit отклонён review:
они ошибочно трактовали raw timer как миллисекунды и не синхронизировали
начало workload с окном PMU.

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

Это сильное доказательство семантики **объём за окно**, а не уже готовой
скорости. Коэффициент пересчёта в bytes/MB пока не объявляется: для этого
нужна отдельная калибровка data-unit и channel-specific semantics. CV теперь
считается по каждой одинаковой ячейке size/window; большинство read-ячеек
имеют CV порядка 0.1–2.7%, исключение — `32 MiB @ 100000 µs` (~29%).

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

Итог остаётся устойчивым: `bandwidth*` и `cmd_rd` — накопленный **объём за
окно**, не готовые MB/s. Конкретный bytes-per-counter unit ещё не объявляется.
Максимум v3: CPU `56.048 °C`, DDR `49.972 °C`, GPU `52.018 °C`, NPU
`49.476 °C`; после серии `pmu_timer=0`.
