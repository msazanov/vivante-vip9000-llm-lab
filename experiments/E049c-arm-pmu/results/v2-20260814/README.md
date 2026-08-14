# E049c-v2 — квалифицированная PMU-калибровка

Статус: **PASS для instrumentation controls; не benchmark модели**. Bonsai не
запускался. Никакого вывода о tokens/s или качестве модели из этой серии нет.

## Измерительный протокол

Один запуск содержит только одну PMU-группу, не превышающую объявленную ёмкость
четырёх hardware counters:

- `core`: `cpu_cycles`, `instructions`, `stall_backend`;
- `cache`: `l1d_cache_refill`, `l2d_cache_refill`, `l3d_cache_refill`;
- `memory`: `mem_access`, `bus_access`.

Child пишет `S` в fd 9 и ждёт ACK в fd 8. Parent одной групповой операцией
делает `RESET + ENABLE`, посылает ACK, затем child выполняет workload и пишет
`E`. После `E` parent делает group `DISABLE` и читает каждый counter вместе с
`time_enabled`/`time_running`. Run принят только если все выбранные события
поддержаны, `time_running / time_enabled >= 0.95` и `sample_valid=true`.

Root нужен только parent для `perf_event_open` при `perf_event_paranoid=2`.
Identity-probe подтвердил, что workload после drop выполнялся как
`uid=1000(orangepi), gid=1000(orangepi)`. В этом же probe были намеренно
унаследованы fd 3 и 4, чтобы реальные pipe попали в конфликтную комбинацию
marker=8/ACK=9; исправленный remap завершился успешно.

## Финальные controls

Все девять controls имеют `status=ok`, `sample_valid=true`, полный
`S/ACK/E`, `exit.code=0`, а у каждого события
`time_running/time_enabled=1.000000000`. Максимальная температура в JSON —
**36.332 °C**, guard 85 °C не сработал.

| PMU-группа | Workload | Время, ms | События: точные count |
|---|---|---:|---|
| core | CPU, 1 000 000 итераций | 11.114999 | cycles 16 347 891; instructions 15 109 612; stall 6 135 658 |
| core | memory 1 MiB × 2 | 3.568916 | cycles 2 497 518; instructions 1 580 118; stall 1 147 537 |
| core | memory 8 MiB × 2 | 18.079205 | cycles 21 564 795; instructions 12 316 318; stall 11 742 397 |
| cache | CPU, 1 000 000 итераций | 11.887207 | L1D 950; L2D 3 936; L3D 1 752 |
| cache | memory 1 MiB × 2 | 2.692499 | L1D 12 087; L2D 43 039; L3D 27 338 |
| cache | memory 8 MiB × 2 | 18.668580 | L1D 99 705; L2D 322 638; L3D 251 686 |
| memory | CPU, 1 000 000 итераций | 16.623414 | MEM_ACCESS 8 044 482; BUS_ACCESS 33 124 |
| memory | memory 1 MiB × 2 | 2.627791 | MEM_ACCESS 560 626; BUS_ACCESS 425 471 |
| memory | memory 8 MiB × 2 | 14.162581 | MEM_ACCESS 4 412 830; BUS_ACCESS 3 372 910 |

Отношение 8 MiB / 1 MiB внутри каждой отдельной группы:

| Группа | Время | События |
|---|---:|---|
| core | 5.065741× | cycles 8.634490×; instructions 7.794556×; stall 10.232696× |
| cache | 6.933551× | L1D 8.248945×; L2D 7.496410×; L3D 9.206453× |
| memory | 5.389539× | MEM_ACCESS 7.871255×; BUS_ACCESS 7.927473× |

Группы сняты **разными процессными запусками**, поэтому counts разных строк
нельзя складывать и выдавать за один одновременный sample. Разброс времени
между группами также нельзя интерпретировать как влияние самого counter-а без
повторов и фиксации частот.

## Негативные safety-controls

| Проверка | Raw-результат | Ожидаемая классификация |
|---|---|---|
| thermal sysfs отсутствует | `status=failed`, `thermal_unreadable`, `sample_valid=false`, exit 143 | fail closed |
| marker `S` не пришёл за 300 ms | `status=failed`, `start_marker_timeout`, exit 143 | вся session/process-group остановлена; grandchild `killed_and_reaped` |
| output — symlink | launcher exit 2, `safe open: File exists` | symlink не прослежен, sentinel после запуска содержит `keep` |

У symlink-теста намеренно нет PMU JSON: безопасное `O_EXCL|O_NOFOLLOW`
отказалось создавать файл. Его stdout/stderr, exit code, link target и копия
sentinel перечислены в общем `data/manifest.tsv`.

## Ограничение семантики

`MEM_ACCESS`, `BUS_ACCESS` и cache refill — **счётчики событий, не DDR bytes**.
Нельзя умножать их на 64 и объявлять фактическим DDR-трафиком без отдельного
доказательства семантики A733/uncore. Эта серия подтверждает пригодность
process-scoped PMU envelope и реакцию controls на рабочий набор; она не решает,
является ли Bonsai memory-bound или compute/unpack-bound.

## Воспроизведение без секрета в журнале

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror tooling/a733_pmu_exec.c -o /tmp/a733-pmu-exec
cc -std=c11 -O2 -Wall -Wextra -Werror tooling/a733_pmu_control.c -o /tmp/a733-pmu-control

printf '<sudo-password>\n' | sudo -S -p '' /tmp/a733-pmu-exec \
  --output /tmp/core.json --event-group core --min-running-ratio 0.95 \
  --start-on-ready --sync-timeout-ms 5000 --max-temp-c 85 -- \
  /tmp/a733-pmu-control --mode cpu --iterations 1000000
```

Установка пакетов, изменение OPP/DDR/thermal policy и запуск Bonsai не
выполнялись.
