# E033 — scheduler `strict=0` на всех ядрах A733

Дата: 2026-08-12  
Плата: Orange Pi Zero 3W / Allwinner A733, 12 GiB  
Модель: `Bonsai-27B-Q1_0.gguf` (`qwen35`, 26.896B параметров)

## Вопрос

В штатном запуске `--cpu-strict 1` worker-потоки round-robin закрепляются на
CPU из маски. На A733 это может заставить быстрые Cortex-A76 ждать медленные
A55 на барьерах. E033 оставляет маску `0xff`, но передаёт
`--cpu-strict 0`, чтобы Linux scheduler распределял worker'ы динамически.
Математика модели и packed Q1-веса не меняются.

`strict=0` — это не «добавить ядра»: используются те же 8 CPU. Меняется только
политика размещения потоков и, как следствие, cache locality/баланс между
кластерами.

## Affinity screen

Короткий экран `n=8` зафиксировал ожидаемую иерархию:

| Режим | CPU mask | Потоки | Ток/с |
|---|---:|---:|---:|
| All-core strict=1 | `0xff` | 8 | 0.884488 |
| A76-only | `0xc0` | 2 | 0.641160 |
| Mixed 2+2 | `0xf0` | 4 | 0.766557 |
| Mixed 4+2 | `0xfc` | 6 | 0.808081 |
| A55-only | `0x3f` | 6 | 0.720282 |

Это screen, а не финальный ranking: для каждого режима был один короткий
измерительный запуск. Невалидные ранние вызовы, где hex-маска ошибочно
передавалась непосредственно в `taskset -c`, в таблицу не включены.

## Target gate

Pinned GGUF SHA-256:
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
Параметры: all-core `0xff`, 8 потоков, F16 KV, flash attention on,
`batch/ubatch=512`, `poll=50`, `ngl=0`, governor `performance`, DDR raw `0x54`,
`n=32`, `r=3`. Запуск обёрнут `profile_command.py` и
`thermal_exec_guard.py` с лимитом 85 °C.

| Вариант | Samples, ток/с | Средняя скорость | Δ к E026 | Решение |
|---|---|---:|---:|---|
| E026 native, strict=1 | 0.885429 / 0.901302 / 0.877385 | **0.888039** | — | reference |
| E033 native, strict=0 | 0.980676 / 0.964671 / 0.959361 | **0.968236** | **+9.03%** | **лучший подтверждённый CPU режим** |

Средняя скорость E033 рассчитана `llama-bench` как reciprocal среднего времени
трёх decode-прогонов; это не среднее округлённых sample rates.

Температурный trace: CPU big `72.275 °C`, CPU little `74.340 °C`, DDR
`64.480 °C`, NPU `63.364 °C`, GPU `66.198 °C`, skin `38.868 °C`. Governor во
время прогона — `performance`, частоты policy — 1794/2002 MHz. Thermal abort
не было; после запуска governor восстановлен в `ondemand`.

Артефакты на плате:

```text
results/e033-affinity-screen-20260812-003/
results/e033-scheduler-screen-20260812-001/
results/e033-scheduler-performance-flash-allcore-n32-r3-strict0-001/
results/e033-quality-strict0-001/
```

## Quality gate

Короткий deterministic completion с теми же prompt, seed=123, temp=0 и
остальными sampling-параметрами, что E030/E031/E032, дал stdout SHA-256:

```text
f70a3ee296f70c270f0530b3794e06ed1e261f5c04c470fb057d4ec5ce8549eb
```

Он совпал с native reference и E032. Поэтому scheduler не изменил проверенный
поток токенов; это не заменяет тесты на разных seed и sampling-политиках.

## Вывод

E033 — текущий production-кандидат запуска: изменение scheduler дало +9,03%
без изменения весов и deterministic качества. E032 block4 на том же `strict=0`
дал 0,958360 ток/с, то есть на 1,02% хуже; его код сохраняется как отдельная
ветка исследования, но не включается по умолчанию. Цель `>1 ток/с` всё ещё не
достигнута.
