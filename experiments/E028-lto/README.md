# E028 — GCC LTO для CPU Q1 decode

Дата: 2026-08-12
Цель: проверить, даст ли Link-Time Optimization выигрыш без изменения
математики Q1, формата GGUF или распределения CPU/NPU.

## Сборка

Отдельная чистая конфигурация:

* `GGML_LTO=ON`;
* `GGML_CPU_REPACK=ON`;
* `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`;
* `GGML_NATIVE=OFF`, `GGML_CUDA/OCL/Vulkan=OFF`;
* GCC 12.2, Release.

Конфигурация подтвердила `dotprod`, но не `i8mm`, что соответствует
`asimddp` без `i8mm` в `Features` A733. Бинарник:

`/home/orangepi/vip9000-lab/build/cpu-38c66-lto/bin/llama-bench`
SHA-256 `3d87f57fc7b23084b54718811f39cf4d17fc5711ec96e09d8f1ff8d91f754c40`

## Target gate

Одинаковые условия E027 reference: all-core `0xff`, 8 strict threads,
`batch/ubatch=512`, `F16 KV`, flash attention, `poll=50`, `ngl=0`, performance
governor, DDR raw `0x54`, `n=32`, `r=3`.

| Вариант | Средняя скорость, ток/с | Samples | Изменение к reference | Пик CPU | Решение |
|---|---:|---|---:|---:|---|
| штатный `native-git` | 0.888039 | 0.885429 / 0.901302 / 0.877385 | — | 72.570 °C | reference |
| LTO | 0.875359 | 0.862534 / 0.877282 / 0.886262 | **−1.43%** | 68.617 °C | **отклонён** |

LTO не приблизил нас к 1 ток/с. Более низкая температура не компенсирует
потерю скорости; это ожидаемо для memory-bound Q1 GEMV, где линковка не
уменьшает поток весов из LPDDR5.

Артефакт на плате: `results/e028-lto-performance-flash-allcore-n32-r3/`.
Сводка: `results/e028_lto.json`.
