# E047 preflight перед E049c: ARM PMU/cache/stall

Дата снимка: 2026-08-14, локальный clone
`/tmp/vivante-vip9000-llm-lab`. Проверка read-only; модель и плата до
запуска control workload не изменялись.

## Проверенные refs

Проверены все 14 refs, видимые через
`git for-each-ref --format='%(refname)' refs/heads refs/remotes` (локальные
ветки включены намеренно):

| Ref | Commit |
|---|---|
| `refs/heads/codex/e047-hard-profiling` | `203f3db5454a` |
| `refs/heads/codex/e048-per-op-trace` | `f5bd54eebbfc` |
| `refs/heads/codex/e049b-trace-analysis` | `06a810c2f333` |
| `refs/heads/codex/e049c-arm-pmu` | `d20c7afc7284` |
| `refs/heads/codex/e054-a76-q1-multiversion` | `203f3db5454a` |
| `refs/heads/main` | `426d1467563d` |
| `refs/remotes/origin/codex/e022-fused-q1` | `d5ac6d0a4570` |
| `refs/remotes/origin/codex/e023-ggml-q1-seam` | `7a79466f7f65` |
| `refs/remotes/origin/codex/e047-hard-profiling` | `203f3db5454a` |
| `refs/remotes/origin/codex/e048-per-op-trace` | `f5bd54eebbfc` |
| `refs/remotes/origin/codex/e049-nsi-calibration` | `c8b7e2696c8f` |
| `refs/remotes/origin/codex/e049b-trace-analysis` | `06a810c2f333` |
| `refs/remotes/origin/codex/profiling-foundation` | `c071476773ad` |
| `refs/remotes/origin/main` | `426d1467563d` |

Поиск выполнялся по каждому ref через `git grep` по термам:
`perf_event_open`, `PERF_TYPE_RAW`, `arm_pmuv3`,
`L1D_CACHE_REFILL`, `L2D_CACHE_REFILL`, `L3D_CACHE_REFILL`, `mem_access`,
`bus_access`, `stall_backend`, `perf`, `PMU`, `cache`, `stall`, `NSI`, `DDR`,
`E049`, `E049c`. Большие raw-файлы E048/E049 были исключены из текстового
вывода, но refs и их commit SHA оставались в области проверки.

## Что уже было и почему E049c не дублирует это

| Найденная работа | Что она измеряет | Отличие E049c |
|---|---|---|
| E025/E005 DDR frequency/PLL | OPP/частота и thermal outcome | не открывает процессные ARM PMU events и не разделяет cache/memory/stall |
| E048 per-op trace | llama.cpp node/worker timing и logical bytes | не получает физические CPU PMU counters; `observed_direct_ddr_*` остаются `null` |
| E049 NSI calibration | root-only NSI read-window/DDR-port калибровка | не является `perf_event_open` CPU PMU и не даёт ARM core events |
| E049b trace analysis | агрегация E048 по layer/op | post-processing, без аппаратного счётчика |
| E009/E023 prefetch/seam | программные Q1 traffic controls | не калибруют hardware event semantics |

Точный поиск имён ARMv8 PMUv3 raw событий в прежних refs не нашёл
`perf_event_open`, `PERF_TYPE_RAW`, `arm_pmuv3` или реализацию событий
`L1D/L2D/L3D_CACHE_REFILL`, `MEM_ACCESS`, `BUS_ACCESS`, `STALL_BACKEND`.
Поэтому E049c — новый контрольный слой: он публикует только raw event counts,
время multiplexing и `unavailable`/`EPERM` evidence; PMU event count никогда не
выдаётся за количество байт DDR.

## Решение preflight

`duplicate_found` для точного `E049-arm-pmu` отсутствует. Существующие E048 и
E049 имеют смежные измерения и должны быть связаны в интерпретации, но новый
launcher не повторяет их метод. До Bonsai обязательны CPU и memory controls;
при отсутствии хотя бы одного поддержанного счётчика запуск модели должен
остановиться с сохранением failure evidence.
