# E030 — парное повторное использование Q8-активаций в Q1 DOTPROD GEMV

Дата: 2026-08-12  
Плата: Orange Pi Zero 3W / Allwinner A733, 12 GiB  
Модель: `Bonsai-27B-Q1_0.gguf` (`qwen35`, 26.896B параметров)

## Вопрос и гипотеза

В decode один вектор активаций Q8 используется многими выходными строками,
но штатное repack-ядро `q1_0_4x4_q8_0` проходит выходы по четыре и повторно
читает тот же Q8-поток. E030 объединил две соседние группы по четыре строки:
Q8-половины загружались один раз, а затем использовались для обеих групп.
Q1 оставался в штатном 72-байтовом `block_q1_0x4`; расширения весов в INT8/FP16
не было. Теоретический inner-loop traffic снижался примерно с 416 до 280
байт на пару групп (−32.7%), но число DOTPROD и распаковок знаков не менялось.

## Host correctness gate

`tooling/e030_q1_paired_golden.py` сравнивает парный путь с независимой
скалярной формулой `sum(sign_i * q8_i)` для `K=128` и `K=5120`, строк
`M=8` и `M=1024`, включая нулевые, единичные, чередующиеся, случайные и
экстремальные `int8` значения. Тест:

```text
python3 -m unittest tests/test_e030_q1_paired_golden.py -v
Ran 5 tests ... OK
```

## Сборка

Исходный commit Prism: `38c66ad0241da4f9fcce541cda8edc219086cec5`.  
Патч: `patches/0001-q1-paired-activation-reuse.patch`.  
Target source SHA-256 patched `repack.cpp`:
`cba6283ff6ad5db5cff3a0f755c96475e1e3c19c6511caadbd5f4abf34320d52`.  
E030 binary SHA-256:
`69aed548c4da1a50f63e1006f45b46880ea5ef29c4d2a70e561760390b277b12`.

Флаги совпадали с эталоном: Release, `GGML_CPU_REPACK=ON`,
`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`, `GGML_NATIVE=OFF`, OpenMP ON,
LTO OFF, CPU-only. Cross-compile syntax check и сборка target завершились
с кодом 0.

## Target performance gate

Общие условия: pinned GGUF SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`,
all-core `0xff`, 8 strict threads, `batch/ubatch=512`, F16 KV, flash attention
`on`, `poll=50`, `ngl=0`, performance governor, DDR raw `0x54`, `n=32`, `r=3`.
Каждый запуск был обёрнут `thermal_exec_guard.py` (85 °C, 100 ms) и
`profile_command.py` (100 ms). Профайлер завершился без ошибки, target
thermal abort не было.

| Вариант | Средняя скорость, ток/с | Samples, ток/с | Δ к reference | Пик CPU | Решение |
|---|---:|---|---:|---:|---|
| Native 4x4 reference | **0.888039** | 0.885429 / 0.901302 / 0.877385 | — | 72.570 °C | reference |
| E030 paired, flash on | **0.559322** | 0.565944 / 0.550712 / 0.561310 | **−37.02%** | 66.552 °C | **отклонён** |

Неправильный ранний прогон без явного `-fa on` (`flash_attn=-1`) дал
`0.569138 tok/s`; он помечен только как unqualified и не используется для
сравнения. Машиночитаемые результаты сохранены в `results/e030_q1_paired.json`.

## Quality gate на реальном completion

Оба бинарника запускались с одинаковыми `prompt`, `seed=123`, greedy
`temp=0`, `top-k=0`, `top-p=1`, `-c 512`, `-n 4`, `-no-cnv`, `-fa on`,
`0xff`/8 strict threads и thermal/profile wrappers. `stdout.log` native и
E030 имеют одинаковый SHA-256:

`f70a3ee296f70c270f0530b3794e06ed1e261f5c04c470fb057d4ec5ce8549eb`.

Значит, в этом deterministic gate расхождения вывода не обнаружено. При этом
perf строки completion показали `0.69 tok/s` native против `0.65 tok/s` E030
(короткий gate, не заменяет n=32/r=3 benchmark).

## Почему гипотеза отклонена

Дизассемблирование target `libggml-cpu.so` дало:

| Метрика функции `ggml_gemv_q1_0_4x4_q8_0` | Native | E030 |
|---|---:|---:|
| размер машинного кода | 0x494 (1172 B) | 0x7ac (1964 B) |
| stack frame | 176 B | 288 B |
| инструкций с обращением к stack | 27 | 48 |

Это наблюдение, а не предположение о счётчиках cache: компилятор удерживал две
группы аккумуляторов и два Q8-вектора ценой дополнительных сохранений и
spill/reload. В итоге локальное сокращение чтений Q8 было перекрыто ростом
регистрового давления, машинного кода и стековых обращений. Температуры были
ниже эталона, поэтому перегрев не объясняет проигрыш.

## Текущий ответ и следующий эксперимент

E030 доказал, что математически точное activation reuse возможно, но на этом
компиляторе прямое inline-парное ядро неэффективно. В production-путь оно не
включается. Следующая ветка E031 должна сохранить только одну дополнительную
активационную загрузку/малый software pipeline и измерить, не раздувая frame;
если GCC снова добавит spill/reload, ветка закрывается без дальнейших
комбинаций внутри этого ядра. NPU/GPU в E030 намеренно не использовались:
это CPU repack-исследование и не доказательство переноса Bonsai на VIP9000.
