# E049c recovery controls — 2026-08-14

Это короткий bounded-gate после восстановления оборванного агента. Bonsai не
запускался. Каждый запуск выполнялся через root-assisted launcher с drop
ребёнка до `orangepi:orangepi` и thermal guard `85 °C`.

| Файл | Workload | elapsed, ms | max temp, °C | status | markers | PMU events |
|---|---|---:|---:|---|---|---|
| `e049c-cpu.json` | CPU, 1 000 000 итераций | 8.862 | 34.782 | `ok` | S/E | 8/8 supported |
| `e049c-memory-small.json` | 1 MiB × 2, stride 64 B | 1.816 | 35.526 | `ok` | S/E | 8/8 supported |
| `e049c-memory-large.json` | 8 MiB × 2, stride 64 B | 15.057 | 38.316 | `ok` | S/E | 8/8 supported |

`scaled_value` в JSON равен `value × time_enabled / time_running` и учитывает
multiplexing. Для memory-small `stall_backend` намеренно имеет
`time_running=0`, поэтому `scaled_value=null`.

Ключевые scaled values:

| Control | cycles | instructions | L1D | L2D | L3D | MEM_ACCESS | BUS_ACCESS | STALL_BACKEND |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 17 714 596 | 15 079 820 | 1 024 | 3 780 | 1 575 | 7 989 562 | 30 960 | 7 841 507 |
| 1 MiB | 3 238 396 | 1 561 201 | 11 738 | 41 375 | 18 487 | 646 961 | 394 706 | null |
| 8 MiB | 26 652 382 | 13 874 271 | 111 103 | 332 477 | 257 680 | 5 147 706 | 3 079 611 | 19 893 092 |

Отношение 8 MiB к 1 MiB: время `8.29×`, L1D `9.47×`, L2D `8.04×`, L3D
`13.94×`, MEM_ACCESS `7.96×`, BUS_ACCESS `7.80×`. PMU counts — это counts
событий, не DDR bytes; вывод о фактической пропускной способности памяти из
этих файлов делать нельзя.

Сырые файлы:

- `*.json` — полный результат launcher;
- `*.stdout` — неизменённый stdout control workload-а;
- `*.stderr` — неизменённый stderr (пустой в этой серии).
