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

## Последний результат

E031 математически точен, не расширяет Q1-веса и не троттлился, но четыре
вызова helper увеличили размер GEMV и не дали end-to-end ускорения. Подробная
карточка: [`experiments/E031-q1-noinline-reuse/README.md`](../../experiments/E031-q1-noinline-reuse/README.md).

Каждая строка должна оставаться append-only: новый запуск получает новый ID,
а rejected/unqualified результаты не удаляются и не превращаются в «оценку».
