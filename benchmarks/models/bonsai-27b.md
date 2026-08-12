# Bonsai 27B — накопленная таблица экспериментов

Pinned модель: `Bonsai-27B-Q1_0.gguf`, SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
Основная метрика — steady-state decode, токенов/с. `rejected` означает, что
вариант измерен, но не превосходит pinned CPU reference `0.888039 ток/с`.

| Эксперимент | Идея / режим | Скорость, ток/с | Изменение к reference | Качество | Статус |
|---|---|---:|---:|---|---|
| E026 | CPU native, F16 KV, flash on, performance governor | **0.888039** | — | throughput baseline | reference |
| E029 | `-mtune=cortex-a55` | 0.856562 | −3.54% | веса не менялись | rejected |
| E030 | inline pair: повторное использование Q8 для двух Q1-групп | 0.559322 | −37.02% | exact stdout | rejected |
| E031 | pair + `noinline` single-half helper | 0.576260 | −35.11% | exact stdout | rejected |
| E032 | один `noinline` helper на четыре Q8-блока (block4), `strict=0` | 0.958360 | +7.92% к E026; −1.02% к E033 | exact stdout | rejected ниже E033 |
| E033 | native GEMV, all-core `strict=0` scheduler | **0.968236** | **+9.03%** | exact stdout | лучший подтверждённый CPU режим |

## Последний результат

E033 сейчас является лучшим подтверждённым CPU-режимом: ослабление жёсткой
round-robin привязки потоков до `strict=0` дало 0.968236 ток/с при совпавшем
quality gate. E032 математически точен, не расширяет Q1-веса и не троттлился,
но отдельный block4 helper дал 0.958360 ток/с — немного хуже одного scheduler
изменения. Подробные карточки: [`experiments/E032-q1-block4-reuse/README.md`](../../experiments/E032-q1-block4-reuse/README.md)
и [`experiments/E033-scheduler-strict0/README.md`](../../experiments/E033-scheduler-strict0/README.md).

Каждая строка должна оставаться append-only: новый запуск получает новый ID,
а rejected/unqualified результаты не удаляются и не превращаются в «оценку».
