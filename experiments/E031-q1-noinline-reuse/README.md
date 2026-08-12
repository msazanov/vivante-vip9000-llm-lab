# E031 — `noinline`-половина для повторного использования Q8

Дата: 2026-08-12  
Плата: Orange Pi Zero 3W / Allwinner A733 / 12 GiB  
Модель: `Bonsai-27B-Q1_0.gguf` (`qwen35`, 26.896B параметров)

## Вопрос

Можно ли сохранить идею E030 — загрузить один Q8-вектор и использовать его
для двух соседних групп Q1-строк — но убрать раздувание регистрового состояния
в вызывающей функции?

В E030 компилятор встроил парный helper прямо в GEMV. Это дало точный вывод,
но увеличило frame с 176 до 288 байт и число обращений к стеку с 27 до 48.
В E031 helper `ggml_q1_0_dot_half` помечен `noinline` и возвращает только один
`int32x4_t` (одну SIMD-половину), а не структуру из двух векторов.

Термины:

* `noinline` — запрет компилятору встраивать функцию; это должно ограничить
  число одновременно «живых» SIMD-регистров в GEMV;
* spill/reload — временная выгрузка регистра в стек и обратная загрузка;
* `int32x4_t` — четыре 32-битных целых в одном NEON-регистре;
* Q1 остаётся упакованным, полноценный INT8/FP16-тензор весов не создаётся.

## Корректность на host

Reference-модель в `tooling/e031_q1_noinline_golden.py` сравнивает парный
pipeline с независимой скалярной формулой `sum(sign * q8)`.

Проверены `K=128`, `K=256` и `K=5120`, случайные, нулевые, единичные,
чередующиеся и экстремальные `int8` значения, независимые масштабы восьми
строк и отсутствие мутации входов:

```text
python3 -m unittest tests/test_e030_q1_paired_golden.py tests/test_e031_q1_noinline_golden.py -v
Ran 8 tests ... OK
```

## Сборка и статический gate на A733

Исходный Prism commit: `38c66ad0241da4f9fcce541cda8edc219086cec5`.  
Патч: `patches/0001-q1-half-noinline-reuse.patch`.  
SHA-256 patched `repack.cpp`:
`3fdbbb7326884bfc70f7835797a4556482db152cfaaf03cfc1bbc01910389a89`.  
Флаг: `-DGGML_Q1_E031_NOINLINE_PAIR`; остальные флаги совпадают с reference:
Release, `GGML_CPU_REPACK=ON`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`,
`GGML_NATIVE=OFF`, OpenMP ON, LTO OFF, CPU-only.

Target binary SHA-256:

```text
llama-bench       734719d20d11c1eae7931625d5a445beb218138f63f6636244d84b7b62108d15
libggml-cpu.so   fdf96d801097c62b8e8b4713fce929223e4a9adfbff5ed4c77521d1a372249fc
```

На плате helper имеет размер `0x78` (120 байт), не содержит prologue,
`[sp]` или stack reference и возвращает результат через `v0`. У вызывающего
`ggml_gemv_q1_0_4x4_q8_0` frame остаётся 176 байт, но размер машинного кода
становится `0x714` (1812 байт) против `0x494` (1172 байт) у reference; число
stack-reference инструкций — 29 против 27. То есть локальный spill устранён,
но цена четырёх вызовов helper и разросшейся ветки осталась.

Сборка была обёрнута `profile_command.py` и `thermal_exec_guard.py`; код
возврата — 0.

## Target performance gate

Одинаковые условия: pinned GGUF SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`, all-core
`0xff`, 8 strict threads, `batch/ubatch=512`, F16 KV, flash attention on,
`poll=50`, `ngl=0`, performance governor, DDR raw `0x54`, `n=32`, `r=3`.
Измерение: `results/e031-half-performance-flash-allcore-n32-r3-faon-003/`.

| Вариант | Средняя скорость, ток/с | Samples, ток/с | Δ к reference | Пик CPU | Решение |
|---|---:|---|---:|---:|---|
| Native 4x4 reference | **0.888039** | 0.885429 / 0.901302 / 0.877385 | — | 72.570 °C | reference |
| E031-half `noinline` | **0.576260** | 0.569652 / 0.571637 / 0.587490 | **−35.11%** | 65.549 °C | **отклонён** |

Температурный trace E031: CPU big `65.224 °C`, CPU little `65.549 °C`, DDR
`59.334 °C`, NPU `58.404 °C`, GPU `62.682 °C`, skin `37.109 °C`;
оба CPU governor — `performance`, частоты 1794/2002 MHz, thermal abort не было.
Следовательно, проигрыш не объясняется троттлингом.

## Quality gate

Reference и E031 запускались с одним prompt
`Explain the A733 memory bottleneck in one short sentence.`, seed 123,
greedy `temp=0`, `top-k=0`, `top-p=1`, `min-p=0`, repeat penalty 1,
context 512, `n=4`, F16 KV, flash attention on, all-core/8 strict.

Оба stdout имеют одинаковый SHA-256:

```text
f70a3ee296f70c270f0530b3794e06ed1e261f5c04c470fb057d4ec5ce8549eb
```

На коротком quality run completion eval был около `0.94 ток/с` у reference и
`0.55 ток/с` у E031-half; это подтверждает тот же вывод, но не является
заменой длинному `n=32/r=3` gate.

## Вывод

E031 подтверждает две вещи:

1. повторное использование Q8 при сохранении упакованных Q1 математически
   корректно и не меняет deterministic token stream;
2. на текущем GCC/A733 четыре `noinline`-вызова дороже экономии повторных
   загрузок. Даже без frame у helper end-to-end throughput ниже на 35,11%.

Кандидат **не включается** в production-путь. Следующая ветка должна менять
порядок данных или fusion границу так, чтобы один вызов обрабатывал больше
работы без удержания двух полных аккумуляторных групп, либо вернуться к
нативному 4x4 и искать выигрыш в планировщике/размещении ядер. NPU/GPU в E031
не использовались: это отрицательный CPU-kernel gate, а не доказательство
непригодности VIP9000.

