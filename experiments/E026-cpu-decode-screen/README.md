# E026 — CPU decode screen: KV, flash attention и governor

Дата: 2026-08-12
Цель: найти безопасный CPU-only выигрыш для Bonsai без изменения весов и
качества. NPU и GPU намеренно отключены (`ngl=0`, backend CPU).

## Единые условия

Модель — pinned `Bonsai-27B-Q1_0.gguf`; штатный бинарник commit `38c66ad`,
build `9594`; all-core CPU mask `0xff`, 8 потоков, `--cpu-strict 1`,
`batch/ubatch=512`, `n=32`, `r=3`, `F16` KV если не указано иное, thermal
guard 100 ms, предел 85 °C. DDR оставалась штатной raw `0x54`.

## Результаты

| Вариант | KV | Flash attention | Governor | Средняя скорость, ток/с | Изменение к лучшему E026 | Пик CPU, °C | Решение |
|---|---|---|---|---:|---:|---:|---|
| Контроль all-core | F16 | штатный | ondemand | 0.873368 | −1.65% | 68.0 | контроль |
| KV quantized | Q8_0 | auto/off | ondemand | 0.863995 | −2.71% | 68.7 | **отклонён** |
| Flash attention | F16 | on | ondemand | 0.876827 | −1.26% | 71.3 | принят как базовая настройка |
| Flash + performance governor | F16 | on | performance | **0.888039** | — | 72.6 | лучший CPU-only |

`performance` governor включался только на время guarded run и затем
восстанавливался в `ondemand`. Все три sample каждого варианта завершились
нормально; swap/OOM и thermal abort не наблюдались.

## Интерпретация

Q8 KV уменьшает объём KV cache, но на этом коротком decode добавляет работу
по преобразованию/доступу и оказался медленнее. Flash attention даёт небольшой
плюс, а фиксированный governor убирает задержки переходов частоты; вместе
получено 0.888 ток/с. Это лишь примерно +1.7% к свежему ondemand/F16
контролю и всё ещё ниже цели 1 ток/с.

Полный результат качества не заявляется: эти прогоны измеряют throughput,
а не perplexity или совпадение с reference token stream. Ни один вариант не
менял GGUF и не материализовал Q1 в INT8.

Артефакты на плате:

* `results/e025-ddr-baseline-allcore-n32-r3/`
* `results/e026-kv-q8-allcore-n32-r3/`
* `results/e026-flash-attn-on-allcore-n32-r3/`
* `results/e026-performance-flash-allcore-n32-r3/`

Машиночитаемая таблица: `results/e026_cpu_decode_screen.json`.
