# E049c — all-branch preflight

Дата снимка: 2026-08-14. Проверка выполнялась read-only до recovery controls.
Сканировались все refs, видимые локальному clone через
`git for-each-ref refs/heads refs/remotes`.

## Решение

| Проверка | Результат |
|---|---:|
| Проверенные refs | 14 |
| Remote-tracking refs | 8 |
| Точный duplicate для `E049-arm-pmu` | нет |
| Candidate count | 0 |
| Active ref | `refs/heads/codex/e049c-arm-pmu` |
| Active commit | `d20c7afc728400ee59d942e0a75baf434476273b` |

Поиск включал `perf_event_open`, `PERF_TYPE_RAW`, `arm_pmuv3`,
`L1D_CACHE_REFILL`, `L2D_CACHE_REFILL`, `L3D_CACHE_REFILL`, `MEM_ACCESS`,
`BUS_ACCESS`, `STALL_BACKEND`, `PMU`, `DDR` и `E049c`. Найденные E048/E049
измеряют соседние уровни (logical bytes и NSI), но отдельного process-scoped
ARM PMUv3 launcher в refs не найдено. Поэтому E049c не объявляется повтором
предыдущего эксперимента.

Полный machine-readable evidence и SHA refs находится в
[`e049c-all-branch-preflight.json`](e049c-all-branch-preflight.json).
