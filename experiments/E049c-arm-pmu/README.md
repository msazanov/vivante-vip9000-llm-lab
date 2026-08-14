# E049c — ARM PMU: CPU/cache/memory control

Статус: **bounded control gate PASS на Orange Pi; три Bonsai-wrapper попытки
классифицированы как невалидные; Bonsai benchmark не засчитан**.

Цель E049c — получить аппаратные счётчики уровня CPU для следующего
разделения узкого места Bonsai: физические cache/memory-access события против
распаковки и арифметики Q1. Это не прямой измеритель DDR-байтов. Значения PMU
публикуются только как количество событий, с `time_enabled`/`time_running` для
контроля multiplexing.

## Почему это новый эксперимент

Перед гипотезой выполнен all-branch preflight по 14 локальным и
remote-tracking refs. Полный JSON-снимок находится в
[`data/e049c-all-branch-preflight.json`](data/e049c-all-branch-preflight.json),
а краткое решение — в
[`data/e049c-all-branch-preflight.md`](data/e049c-all-branch-preflight.md).
Для сравнения сохранён также первоначальный E047-снимок:
[`data/e047-all-branch-preflight.md`](data/e047-all-branch-preflight.md).
Ранее E048 измерял logical bytes и время узлов, а E049 — root-only NSI
калибровку. Ни один из найденных refs не содержал процессный
`perf_event_open`-launcher с ARMv8 raw events. E049c поэтому не повторяет
E048/E049: он добавляет независимый hard-counter слой.

## Что сделано

[`tooling/a733_pmu_exec.c`](../../tooling/a733_pmu_exec.c) — небольшой Linux
launcher, который:

1. создаёт child setup gate, чтобы counters открылись до `exec`;
2. открывает raw ARMv8 PMUv3 events на PID процесса с `inherit=1`, то есть
   включает его worker threads;
3. по умолчанию оставляет parent privileged, а child до `exec` переводит в
   `orangepi:orangepi`;
4. в exact режиме ждёт от workload маркеры `S`/`E` через
   `A733_PMU_SYNC_FD`, включая и выключая counters только вокруг работы;
5. проверяет thermal guard (85 °C), сохраняет exit status, errno и каждое
   unavailable-событие в JSON.

События и raw-коды из Linux `arm_pmuv3.h`:

| Имя | Код | Семантика |
|---|---:|---|
| `cpu_cycles` | `0x11` | CPU_CYCLES, count |
| `instructions` | `0x08` | INST_RETIRED, count |
| `l1d_cache_refill` | `0x03` | L1D refill events, не bytes |
| `l2d_cache_refill` | `0x17` | L2D refill events, не bytes |
| `l3d_cache_refill` | `0x2A` | L3D refill events, не bytes |
| `mem_access` | `0x13` | MEM_ACCESS events |
| `bus_access` | `0x19` | BUS_ACCESS events |
| `stall_backend` | `0x24` | backend-stall events |

[`tooling/a733_pmu_control.c`](../../tooling/a733_pmu_control.c) содержит два
детерминированных workload-а:

* `cpu` — фиксированный integer-хеш с checksum;
* `memory` — 64-byte stride read+write с заданным размером и числом проходов.

## Host TDD gate

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_a733_pmu_exec -v
Ran 3 tests ... OK
```

На development host все ARM raw opens вернули `EPERM`; это ожидаемая и
сохранённая ветка `counter_unavailable`, а не подстановка нулей. Это также
доказывает, что launcher корректно публикует отказ ядра.

## Target fingerprint и права

На Orange Pi перед control run:

```text
Linux orangepizero3w 6.6.98-sun60iw2 #1.0.0 SMP PREEMPT ... aarch64
uid=1000(orangepi) gid=1000(orangepi)
/proc/sys/kernel/perf_event_paranoid = 2
```

Обычный `orangepi` не может открыть raw PMU при `paranoid=2`; запуск сделан
root-assisted через `sudo`, при этом сам workload выполнялся после drop в
`orangepi:orangepi`. Никакие OPP, DDR или thermal settings не менялись.

## Target control results

Все JSON raw-файлы и stdout/stderr сохранены без фильтрации в
`results/controls/` и в
[`results/recovery-20260814/`](results/recovery-20260814/). Для числового
сравнения ниже приведён
`scaled_value = value × time_enabled / time_running`; это коррекция
multiplexing, а не дополнительное физическое измерение.

### Recovery control: точное сравнение CPU и memory pressure

Свежая bounded-серия использовала одинаковый launcher и thermal guard, но
разные детерминированные workload-ы. Все 8 событий имеют
`support=supported`, `exit.code=0`, маркеры `S/E` присутствуют.

| Control | Работа | Время, ms | cycles | instructions | L1D | L2D | L3D | MEM_ACCESS | BUS_ACCESS | STALL_BACKEND |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 1 000 000 итераций | 8.862 | 17 714 596 | 15 079 820 | 1 024 | 3 780 | 1 575 | 7 989 562 | 30 960 | 7 841 507 |
| Memory-small | 1 MiB × 2, stride 64 B | 1.816 | 3 238 396 | 1 561 201 | 11 738 | 41 375 | 18 487 | 646 961 | 394 706 | `null` (`time_running=0`) |
| Memory-large | 8 MiB × 2, stride 64 B | 15.057 | 26 652 382 | 13 874 271 | 111 103 | 332 477 | 257 680 | 5 147 706 | 3 079 611 | 19 893 092 |

При переходе с 1 MiB на 8 MiB (8× рабочий набор при одинаковом числе
проходов) время выросло в **8.29×**, L1D refill — в **9.47×**, L2D — в
**8.04×**, L3D — в **13.94×**, MEM_ACCESS — в **7.96×**, BUS_ACCESS — в
**7.80×**. Это подтверждает реакцию CPU PMU на memory pressure. Это не
означает, что событие равно байту или что измерен DDR-трафик Bonsai.

Температуры recovery control: CPU **34.782 °C**, 1 MiB **35.526 °C**, 8 MiB
**38.316 °C**; thermal guard `85 °C` ни разу не сработал. Сырые файлы и их
хеши перечислены в `data/manifest.tsv`.

### CPU, 10 000 000 итераций, три повтора

Checksum каждого повтора одинаковый: `15258602366918737304`.

| Run | ms | cycles | instructions | L1D refill | L2D refill | L3D refill | MEM_ACCESS | BUS_ACCESS | STALL_BACKEND |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88.290 | 176,645,232 | 150,135,349 | 1,863 | 5,808 | 3,689 | 80,110,489 | 50,116 | 79,805,730 |
| r2 | 88.692 | 172,908,749 | 150,142,385 | 2,089 | 7,066 | 4,145 | 80,114,626 | 48,018 | 74,998,950 |
| r3 | 88.886 | 171,152,730 | 150,129,696 | 2,411 | 7,822 | 3,783 | 80,127,944 | 49,112 | 72,794,174 |

`instructions` и `MEM_ACCESS` воспроизводятся особенно стабильно; cache
refill и stall чувствительнее к планированию и multiplexing. Это полезный
контроль семантики, но не разрешение на вывод о DDR bandwidth.

### Memory, read+write по cacheline

| Run | Рабочий набор | Проходы | ms | L1D refill | L2D refill | L3D refill | MEM_ACCESS | BUS_ACCESS | STALL_BACKEND |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 1 MiB | 16 | 2.264 | 62,169 | 26,546 | 49,006 | 1,917,671 | 2,125,200 | `null` (не было running window) |
| large | 64 MiB | 4 | 115.460 | 3,422,624 | 2,807,168 | 4,811,290 | 51,673,167 | 42,789,998 | 153,620,524 |

Увеличение рабочего набора на контрольной программе заметно поднимает
L1/L2/L3 refill и BUS_ACCESS. Следовательно, на target счётчики реально
открываются и реагируют на memory pressure. Для small `stall_backend` был
временно вытеснен multiplexing-ом (`time_running=0`); это опубликовано как
`scaled_value=null`, а не как ноль.

Все старые target control JSON и три recovery JSON имеют `status=ok`,
`exit.code=0`, S/E markers и `thermal.tripped=false`. Это результат
**калибровки instrumentation**, а не результат инференса модели.

## Interrupted agent и провальные Bonsai-wrapper попытки

Работа была восстановлена из незакоммиченного worktree
`codex/e049c-arm-pmu` (HEAD `d20c7af`). Незакоммиченный код сначала прошёл
host-TDD и ручной safety review; настройки платы, OPP, DDR и system install не
менялись. Три compatibility-запуска сохранены в
`results/compat-00[1-3]-*/` и **не являются успешными Bonsai-тестами**:

| Попытка | Результат | Что сломалось | Статус измерения |
|---|---|---|---|
| compat-001 | `Bad fd number` | POSIX wrapper не смог передать marker FD | failed до валидного запуска измерительного workload-а |
| compat-002 | `cannot create /proc/self/fd/198: Permission denied` | wrapper не передал boundary marker | failed; llama.cpp фактически успел стартовать, поэтому этот raw-log нельзя описывать как «модель не запускалась» |
| compat-003 | `eval: Syntax error: Bad fd number` | повторная несовместимость shell FD | failed до валидного запуска измерительного workload-а |

Итоговая классификация всей тройки: **Bonsai PMU measurement failed / не
засчитано**, а не «получена скорость» и не «успешный инференс». Попытка
compat-002 содержит лог llama.cpp, но граница PMU не была установлена, поэтому
её время и counters нельзя использовать для сравнения. Wrapper намеренно не
исправляется в этом commit.

## Что ещё не утверждается

* Счётчики не дают количества DDR bytes. Нельзя умножать refill count на
  размер cacheline и называть результат observed DDR traffic без подтверждения
  конкретной PMU semantics A733.
* Multiplexing означает, что для коротких runs часть событий может иметь
  нулевое `time_running`; тогда scaled value намеренно `null`.
* Control gate подтверждает instrumentation, а не bottleneck Bonsai.
* Bonsai запускается только после review этой калибровки и с E047:
  `seed=123`, `temperature=0`, `context=512`, `batch/ubatch=512`, `threads=8`,
  `flash_attention=on`, `mmap=on`, `kv_cache=f16`, общий prompt и
  `n_predict=4`, без n=32 в этом этапе.

## Воспроизведение на target

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror tooling/a733_pmu_exec.c -o /tmp/a733-pmu-exec
cc -std=c11 -O2 -Wall -Wextra -Werror tooling/a733_pmu_control.c -o /tmp/a733-pmu-control

printf '<sudo-password>\n' | sudo -S -p '' /tmp/a733-pmu-exec \
  --output /tmp/e049c-control.json \
  --start-on-ready --sync-timeout-ms 5000 \
  -- /tmp/a733-pmu-control --mode cpu --iterations 10000000
```

В JSON обязательно проверять `status`, все `events[].support`,
`time_running_ns`, `thermal.tripped`, `exit.code` и `failure_reason`.
