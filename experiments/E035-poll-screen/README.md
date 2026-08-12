# E035 — интервал `poll` в scheduler

Дата: 2026-08-12  
Плата: Orange Pi Zero 3W / Allwinner A733  
Модель: pinned `Bonsai-27B-Q1_0.gguf`

## Термин

`poll` — настройка, задающая частоту проверки worker'ом общей очереди и
добровольной уступки CPU. Частая проверка может добавлять overhead; редкая —
ухудшать баланс A55/A76. E035 сохраняет native Q1 GEMV, all-core `0xff`,
8 workers и `--cpu-strict 0`, меняя только `poll`.

## Screen и полный gate

Короткий screen (`n=8/r=1`) дал:

| poll | Screen, ток/с | Полный `n=32/r=3`, ток/с | Samples полного gate | Решение |
|---:|---:|---:|---|---|
| 0 | 0.973818 | — | — | screen |
| 25 | 0.897133 | — | — | screen, хуже |
| **50** | **0.991160** | **0.972497** | 0.988354 / 0.969448 / 0.959690 | **лучший подтверждённый poll** |
| 100 | 0.983271 | 0.964588 | 0.966104 / 0.968048 / 0.959614 | rejected |

Для `poll=50` полный gate использовал F16 KV, flash on, `batch/ubatch=512`,
governor `performance`, DDR raw `0x54`, thermal limit 85 °C.

Температуры `poll=50`: CPU big `73.455 °C`, little `73.691 °C`, DDR `65.100 °C`,
NPU `62.620 °C`, GPU `66.729 °C`, skin `38.738 °C`; thermal abort не было.

## Quality gate

Completion для poll=50, prompt/seed/sampling как в E033, завершился с кодом 0.
stdout SHA:

```text
f70a3ee296f70c270f0530b3794e06ed1e261f5c04c470fb057d4ec5ce8549eb
```

Он совпал с E033/native reference. Таким образом, scheduler tuning не изменил
проверенный deterministic token stream.

Артефакты на плате:

```text
results/e035-poll-screen-20260812-001/
results/e035-poll50-performance-flash-allcore-n32-r3-strict0-001/
results/e035-poll100-performance-flash-allcore-n32-r3-strict0-001/
results/e035-quality-poll50-001/
```

## Вывод

`poll=50` оставляем текущей настройкой: повторный полный gate дал
0.972497 ток/с, что на 0.44% выше E033 и находится в его естественном
разбросе. `poll=100` не принят. Короткий screen `0.991160` не выдаётся за
реальный рекорд — длинный gate обязателен.
