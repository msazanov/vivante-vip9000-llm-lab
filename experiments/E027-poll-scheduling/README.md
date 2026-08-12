# E027 — polling threadpool: `poll=0` против `poll=50`

Дата: 2026-08-12
Цель: проверить, не съедает ли служебный polling threadpool время decode.

`poll` — это частота внутренних проверок завершения работы CPU-worker. Ноль
убирает периодические проверки; это не частота ядра и не polling NPU.

## Строгий A/B

Оба прогона использовали один GGML binary `38c66ad/9594`, all-core mask
`0xff`, 8 strict threads, `batch/ubatch=512`, `F16 KV`, flash attention,
`ngl=0`, performance governor, DDR raw `0x54`, `n=32`, `r=3`.

| `poll` | Средняя скорость, ток/с | Samples, ток/с | Пик CPU, °C | Статус |
|---:|---:|---|---:|---|
| 50 | 0.888039 | 0.885429 / 0.901302 / 0.877385 | 72.570 | reference |
| 0 | 0.885928 | 0.899180 / 0.884917 / 0.873685 | 70.859 | **−0.24%, отклонён** |

## Вывод

Polling не является узким местом: его отключение слегка ухудшило среднюю
скорость. Оставляем штатное `poll=50`; искать следующие проценты нужно в
repack/GEMV и traffic памяти, а не в настройке ожидания worker.

Артефакт на плате: `results/e027-poll0-performance-flash-allcore-n32-r3b/`.
Сводка: `results/e027_poll_scheduling.json`.
