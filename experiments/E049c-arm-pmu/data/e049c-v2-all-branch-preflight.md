# E049c-v2 — all-branch preflight

Перед изменением протокола PMU выполнены `git fetch --prune origin` и
read-only сканирование **16** локальных и remote-tracking refs. Машиночитаемый
снимок: [`e049c-v2-all-branch-preflight.json`](e049c-v2-all-branch-preflight.json).

Поисковые термины: `a733_pmu_exec`, `PERF_IOC_FLAG_GROUP`. Совпадения найдены
только в `codex/e049c-arm-pmu` и её `origin/`-копии на одном commit
`d5dc982fc9fa75fe02a8b0d9eea8e31b6b403a84`. Другой реализации атомарной
PMU-группы и двустороннего `S → ACK → E` протокола не найдено.

Идентификатор preflight задан как `E049-ARM-PMU`, потому что текущий parser
экспериментальных ID принимает буквенный суффикс только после дефиса. Решение:
`no_duplicate`, `candidate_count=0`. Совпадение текущей ветки по терминам
рассматривается как исходная E049c-v1, которую v2 исправляет, а не как отдельный
повтор гипотезы.
