# E029 — A55-tuned DOTPROD CPU variant

Дата: 2026-08-12
Цель: проверить, улучшит ли расписание GCC для шести A55 all-core decode,
не меняя Q1-математику и формат весов.

## Сборка

Отдельный target build:

* `-mtune=cortex-a55` для C/C++;
* прежний `-march=armv8.2-a+dotprod`;
* `GGML_CPU_REPACK=ON`, `GGML_NATIVE=OFF`, LTO выключен;
* CPU-only, без NPU/GPU.

Бинарник: `/home/orangepi/vip9000-lab/build/cpu-38c66-mtune-a55/bin/llama-bench`
SHA-256: `8d184ac31007a7088acd7d9b8e8c834fc26b2132a807db5b3dc7d3549afee42d`

## Target gate

All-core mask `0xff`, 8 strict threads, `batch/ubatch=512`, `F16 KV`, flash
attention, `poll=50`, `ngl=0`, performance governor, DDR raw `0x54`, `n=32`,
`r=3`, thermal guard 100 ms.

| Вариант | Средняя скорость, ток/с | Samples | Изменение к reference | Пик CPU | Решение |
|---|---:|---|---:|---:|---|
| native reference | 0.888039 | 0.885429 / 0.901302 / 0.877385 | — | 72.570 °C | reference |
| `-mtune=cortex-a55` | 0.856562 | 0.868249 / 0.866974 / 0.834463 | **−3.54%** | 69.325 °C | **отклонён** |

Более низкая температура не является выигрышем: средняя скорость и худший
sample заметно хуже. На неоднородном A733 единый A55 scheduling hint не
компенсирует стоимость работы на двух A76 и общий поток весов из LPDDR5.

Артефакт: `results/e029-mtune-a55-performance-flash-allcore-n32-r3/`.
Сводка: `results/e029_mtune_a55.json`.
