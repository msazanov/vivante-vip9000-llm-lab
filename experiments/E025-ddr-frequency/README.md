# E025 — ветка частоты DDR: fail-closed попытка raw `0x63`

Дата: 2026-08-12
Плата: Orange Pi Zero 3W / A733, 12 GiB LPDDR5
Модель: `Bonsai-27B-Q1_0.gguf`, SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`
Guard: `a733_ddr_pll_guard.py`, watchdog requested 15 s / actual 16 s,
thermal/cpufreq/clock/readback telemetry every 250 ms.

## Гипотеза

Увеличение DDR с штатных 510 MHz (raw `0x54`, регистр
`0xf9005400`) примерно до 600 MHz (raw `0x63`, регистр `0xf9006300`)
могло бы поднять пропускную способность памяти и ускорить Q1 decode. Это
была отдельная аппаратная ветка, не изменение ядра GGML и не NPU-тест.

## Что произошло

| Этап | Наблюдение | Статус |
|---|---|---|
| Контроль raw `0x54` | all-core, `n=32`, 3 runs: **0.873368 ток/с** | контроль |
| Запись raw `0x63` | SSH оборвался сразу после arm/watchdog и pre-write sample | **аппаратный reset** |
| После перезагрузки | PLL вернулся к `0xf9005400`, DDR — к штатному raw `0x54` | восстановлено |
| Workload Bonsai | JSON результата не появился; до инференса не дошло | не измерялся |

В event log есть только события `start`, `watchdog` и `sample` до записи.
Событий успешной записи, workload или штатного restore нет. Температура до
попытки была около 40 °C, поэтому это не тепловой троттлинг.

## Вывод

Режим raw `0x63` классифицирован как
`failed_hardware_reset_at_PLL_write`. Скорость нельзя экстраполировать из
этого запуска: частота не была стабильно применена. Raw `0x63` повторно не
использовать. Любой будущий DDR-тест должен начинаться с меньшего шага,
отдельного fail-closed guard и короткого memory-only smoke до Bonsai.

Артефакты на плате:

* `/home/orangepi/vip9000-lab/results/e025-ddr600-allcore-n32/guard-events.jsonl`
* `/home/orangepi/vip9000-lab/results/e025-ddr-baseline-allcore-n32-r3/`

Машиночитаемая сводка: `results/e025_ddr_frequency.json`.
