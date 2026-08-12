# E032 — один вызов для четырёх Q1/Q8 блоков

Дата: 2026-08-12  
Плата: Orange Pi Zero 3W / Allwinner A733, 12 GiB  
Модель: `Bonsai-27B-Q1_0.gguf` (`qwen35`, 26.896B параметров)

## Гипотеза

E030/E031 пытались повторно использовать Q8-активации между соседними
группами выходов, но несколько аккумуляторов в одной GEMV-функции создавали
spill/reload — выгрузку NEON-регистров во временные ячейки стека. E032 меняет
границу fusion: один `noinline` helper принимает сразу четыре Q8-блока,
соответствующих одному `block_q1_0x4`, и возвращает один `float32x4_t`.

Это не расширяет Q1-веса в INT8/FP16: знаки по-прежнему распаковываются из
упакованных байтов только в NEON-регистрах. Один вызов покрывает 512 значений
вместо 128, поэтому стоимость вызова ниже, а четыре Q8-блока читаются внутри
одной функции.

Термины:

* `Q1/Q8` — формат весов Q1_0 и квантованных активаций Q8_0;
* `DOTPROD`/`sdot` — NEON-инструкция четырёх парных INT8 dot-product;
* `noinline` — helper не встраивается в GEMV, ограничивая живое регистровое
  состояние;
* `float32x4_t` — четыре FP32 результата в одном SIMD-регистре.

## Host golden

Независимая скалярная модель сравнивает четыре строки с формулой
`sum(sign * q8) * scale`. Проверены нули, единицы, чередующиеся знаки,
экстремальные INT8 и случайные данные для `K=512` и `K=5120`; входы не
мутируются.

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests/test_e030_q1_paired_golden.py \
  tests/test_e031_q1_noinline_golden.py \
  tests/test_e032_q1_block4_golden.py -v
Ran 11 tests ... OK
```

## Сборка и статический gate

Исходный Prism commit: `38c66ad0241da4f9fcce541cda8edc219086cec5`.  
Патч: [`0001-q1-block4-reuse.patch`](0001-q1-block4-reuse.patch), SHA-256
`ecd3b239fe86abd615b8d1d3966758649bdae616931447945775d579f89a8de4`.  
SHA-256 изменённого `repack.cpp`:
`de58128736bb408a4594c1af693125b613e45665f1cc412ba25b1c0bd9a44345`.

Флаги: Release, `GGML_CPU_REPACK=ON`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`,
`GGML_NATIVE=OFF`, `GGML_LTO=OFF`, OpenMP ON, CPU-only. Сборка на плате
завершилась с кодом 0 под `thermal_exec_guard.py` и `profile_command.py`.

```text
llama-bench       3dff1b0cc8605d01f94fe2edb26ca0eea7f1f3cac24344bcf19e9ad57b6c8618
llama-completion  55a2a3b144c885330ad824be9ff366cf83176a88021f02e0110655da3fd57c3a
libggml-cpu.so   6846a764e0121b4ed6faa8ae19d1067cb93bdd6d3ff2fcdc9c49d0f9445551a3
```

Target `objdump`:

| Функция | Native | E032 |
|---|---:|---:|
| `ggml_gemv_q1_0_4x4_q8_0` | 1172 B (`0x494`) | 244 B (`0xf4`) |
| `ggml_q1_0_dot_block4` | — | 948 B (`0x3b4`) |

Размер GEMV уменьшился, но helper стал отдельным вызовом. Это статическое
наблюдение; окончательное решение принимается только по end-to-end gate.

## Target performance gate

Pinned GGUF SHA-256:
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
Общие параметры совпадают с E033: all-core `0xff`, 8 потоков,
`--cpu-strict 0`, `batch/ubatch=512`, F16 KV, flash attention on, `poll=50`,
`ngl=0`, governor `performance`, DDR raw `0x54`, `n=32`, `r=3`, thermal limit
85 °C.

| Вариант | Средняя скорость, ток/с | Samples, ток/с | Δ к E026 | Δ к E033 | Решение |
|---|---:|---|---:|---:|---|
| E026 native strict=1 | **0.888039** | 0.885429 / 0.901302 / 0.877385 | — | — | reference |
| E033 native strict=0 | **0.968236** | 0.980676 / 0.964671 / 0.959361 | +9.03% | — | текущий рекорд |
| E032 block4 strict=0 | **0.958360** | 0.954903 / 0.959495 / 0.960682 | +7.92% | −1.02% | **кандидат отклонён** |

E032 заметно лучше старой native-привязки, но уступает одному только
планировщику E033. Повторное использование Q8 и укрупнение helper полезны как
техническая база для следующего fusion-эксперимента, но production-путь пока
не меняется.

Температурный trace E032: CPU big `73.160 °C`, CPU little `74.753 °C`, DDR
`65.224 °C`, NPU `62.930 °C`, GPU `65.844 °C`, skin `38.950 °C`; thermal abort
не было. Частоты policy: 1794 MHz (little) и 2002 MHz (big).

Артефакты на плате:

```text
results/e032-build-001/
results/e032-block4-performance-flash-allcore-n32-r3-strict0-001/
results/e032-quality-strict0-001/
```

## Quality gate

Completion запускался с тем же prompt, seed и greedy-параметрами, что E030/E031.
stdout E032, E033 и native reference имеет одинаковый SHA-256:

```text
f70a3ee296f70c270f0530b3794e06ed1e261f5c04c470fb057d4ec5ce8549eb
```

Это подтверждает совпадение deterministic token stream на объявленном коротком
quality gate; это не утверждение о тождестве всех возможных sampling seed.

## Вывод

E032 подтверждает, что более широкий software pipeline сохраняет точность и
снижает end-to-end проигрыш с −35…−37% у E030/E031 до +7.92% к старому
baseline. Однако memory-bound decode всё ещё лучше работает с native GEMV и
`strict=0`: дополнительный helper не компенсирует стоимость отдельного вызова.
Следующая ветка — совместить E032-представление с scheduler-профилем E033 или
измерить более крупную межоперационную пачку, не материализуя расширенные Q1
веса.
