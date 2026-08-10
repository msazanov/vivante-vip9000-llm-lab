# E007 — PowerVR OpenCL packed-Q1 GEMV baseline

Статус: исходники, CPU golden и target PowerVR sweep проверены. Direct packed
Q1_0×F32 вычисляет правильный результат, но текущий путь отклонён по полной
задержке. Эксперимент не содержит и не заявляет результат Bonsai в tok/s.

## Вопрос

Может ли PowerVR BXM-4-64 на A733 выполнить корректный Q1_0×F32 GEMV прямо из
canonical packed-весов без явной expanded host-копии, и какова полная цена
H2D, kernel, D2H и host submit/wait для реальных shapes Bonsai? Внутренние
staging/repack/expansion драйвера этим runner не наблюдаются.

## Реализация

- `powervr_opencl_q1_runner.c` — C11 runner без compile-time зависимости от
  OpenCL. Он делает только `dlopen("/usr/lib/libPVROCL.so.1")`; generic
  `libOpenCL.so`/ICD не используется.
- `q1_packed_gemv.cl` — переносимый OpenCL C 1.2 kernel. Один work-item считает
  одну output row; разрешённые local size: 32, 64 и 128.
- Q1_0 block содержит 18 bytes на 128 коэффициентов: little-endian FP16 scale,
  затем 16 sign bytes, LSB-first (`1` = `+scale`, `0` = `-scale`). Весовой
  buffer загружается один раз и остаётся resident на warmup и steady calls.
- Activation и output имеют F32 layout. `M` — число output rows, `K` — число
  input columns; оба аргумента обязаны быть положительными и кратными 128.
- Scalar CPU golden декодирует те же 18-byte blocks без SIMD/repack. GPU output
  проходит `atol=1e-3`, `rtol=1e-5`; non-finite или первый mismatch завершает
  run без публикации JSONL/output. Сам scalar golden также отклоняет non-finite
  activation, scale и accumulator/output.
- Публикация fail-closed: runner проверяет каждый OpenCL status, отказывается
  перезаписывать результаты и публикует пару временных файлов через
  `link(temp, final)` без замены существующего пути. Если второй link не удался,
  первый откатывается. Поле
  `explicit_host_expanded_weight_allocation_bytes=0` означает только отсутствие
  явной expanded host-копии в runner. Поле
  `runtime_internal_expansion_observed="unknown"`: внутреннюю память PowerVR
  runtime/driver runner не измеряет и не утверждает, что expansion там нет.
- Квалифицированный steady run требует `warmup >= 1` и `iterations >= 50`;
  runner отклоняет меньшие значения до создания output-файлов.

Это корректный baseline, не оптимизированный production kernel. Здесь пока нет
vector loads, subgroup, integer-dot extension, command buffer или fused post-op.

## Сборка на target

OpenCL headers и generic ICD для сборки не нужны:

```bash
cmake -S experiments/E007-powervr-opencl-q1 \
  -B /tmp/e007-powervr-opencl-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/e007-powervr-opencl-build --parallel 2
/tmp/e007-powervr-opencl-build/powervr-opencl-q1-runner --print-contract
```

Runner ожидает существующие exact-size файлы:

```text
weights bytes    = M * (K / 128) * 18
activation bytes = K * 4
output bytes     = M * 4
```

CPU-only golden можно получить без PowerVR runtime:

```bash
/tmp/e007-powervr-opencl-build/powervr-opencl-q1-runner \
  --cpu-reference-only \
  --weights /data/fixture/synthetic-m128-k256.q1_0.bin \
  --activation /data/fixture/synthetic-k256.f32.bin \
  --m 128 --k 256 --local-size 64 \
  --warmup 1 --iterations 50 --run-id e007-cpu-golden-128x256 \
  --output-jsonl /tmp/e007-cpu-golden.jsonl \
  --output-f32 /tmp/e007-cpu-golden.f32.bin
```

## Guarded target run

Thermal policy остаётся снаружи. Первый подготовленный model fixture —
`blk.0.ffn_gate.weight`, runner shape `M=17408`, `K=5120`; переданный сокращённый
SHA-256 packed weights: `0f42…6fe`. Полный digest нужно сохранить рядом с
конкретным fixture и JSONL. Пример квалифицированного target run:

```bash
python3 tooling/profile_command.py \
  --output-dir /tmp/e007-profile-17408x5120-l64 \
  --run-id e007-powervr-17408x5120-l64 --interval-ms 10 -- \
python3 tooling/thermal_exec_guard.py \
  --limit-mc 85000 --interval-ms 10 \
  --trace /tmp/e007-profile-17408x5120-l64/thermal.jsonl -- \
/tmp/e007-powervr-opencl-build/powervr-opencl-q1-runner \
  --kernel experiments/E007-powervr-opencl-q1/q1_packed_gemv.cl \
  --weights /data/fixture/blk.0.ffn_gate.weight.q1_0.bin \
  --activation /data/fixture/activation-k5120.f32.bin \
  --m 17408 --k 5120 --local-size 64 \
  --warmup 1 --iterations 50 --run-id e007-powervr-17408x5120-l64 \
  --output-jsonl /tmp/e007-powervr-17408x5120-l64.jsonl \
  --output-f32 /tmp/e007-powervr-17408x5120-l64.f32.bin
```

Каждый local size и shape получает новый run-id/output path; существующие
результаты runner атомарно не перезаписывает. Короткий GPU smoke с меньшим
числом итераций этим qualified runner намеренно не поддерживается.

## Три реальные Bonsai shapes

Runner использует `M×K = ne1×ne0`. Форма `5120×5120` — synthetic/non-model:
в актуальном workload manifest нет Q1_0 tensor такой формы.

| Runner `M×K` | Tensor `ne0×ne1` | Count | Packed weights | Activation F32 | Output F32 |
|---:|---:|---:|---:|---:|---:|
| `17408×5120` | `5120×17408` | 128 | 12,533,760 B | 20,480 B | 69,632 B |
| `5120×17408` | `17408×5120` | 64 | 12,533,760 B | 69,632 B | 20,480 B |
| `5120×6144` | `6144×5120` | 64 | 4,423,680 B | 24,576 B | 20,480 B |

Первым выполняется local-size sweep `32/64/128` для уже извлечённого
`blk.0.ffn_gate.weight` (`17408×5120`):

```bash
for local_size in 32 64 128; do
  /tmp/e007-powervr-opencl-build/powervr-opencl-q1-runner \
    --kernel experiments/E007-powervr-opencl-q1/q1_packed_gemv.cl \
    --weights "/data/fixture/weights-17408x5120.q1_0.bin" \
    --activation "/data/fixture/activation-k5120.f32.bin" \
    --m 17408 --k 5120 --local-size "$local_size" \
    --warmup 1 --iterations 50 \
    --run-id "e007-17408x5120-l${local_size}" \
    --output-jsonl "/tmp/e007-17408x5120-l${local_size}.jsonl" \
    --output-f32 "/tmp/e007-17408x5120-l${local_size}.f32.bin"
done
```

После него отдельно извлекаются `blk.0.ffn_down.weight` для `5120×17408` и
подходящий tensor формы `6144×5120`; для каждого повторяется тот же sweep с
собственными `--weights`, `--activation`, `--m` и `--k`. В реальном target
matrix каждый вызов должен быть обёрнут теми же `profile_command.py` и
`thermal_exec_guard.py`, что в примере выше.

## JSONL contract

Успешный GPU run публикует:

1. `identity`: direct runtime, platform/device/driver, shape, local size,
   warmup/iterations;
2. `memory`: packed/activation/output bytes, one resident weight upload,
   `explicit_host_expanded_weight_allocation_bytes=0`,
   `runtime_internal_expansion_observed="unknown"`, `cpu_fallback_count=0`;
3. `transfer`: build; отдельные event/host wall для weights H2D и activation
   H2D; event и host суммы;
4. один `sample` на steady iteration: `kernel_event_ms` отдельно от
   `host_submit_wait_ms`;
5. `result`: scalar golden, error, отдельные D2H event/host wall и общий
   `host_total_ms`.

`host_total_ms` начинается непосредственно перед weights H2D и заканчивается
после final D2H и завершающего `clFinish`. Он включает weights/activation H2D,
event query/release, warmup, все steady submit/wait, D2H и финальную
синхронизацию. Он исключает чтение input-файлов, CPU golden, `dlopen`, создание
context/program/buffers, OpenCL build и публикацию output; `build_ms` записан
отдельно.

Kernel event не является end-to-end временем. Никакое сравнение скорости не
квалифицируется без полного host total, CPU reference, thermal/frequency/fault
telemetry и повторяемости.

## Результат на реальном тензоре Bonsai

Из закреплённой модели извлечён `blk.0.ffn_gate.weight`: runner shape
`M=17408`, `K=5120`, packed weights `12,533,760 B`, SHA-256
`0f42ca3b81099f540ed67941809ee7fdc0a672563bf852b87db75034135fc6fe`.
Activation F32 создан детерминированно с seed `1843`, его SHA-256 —
`053e8523718a3a6fd9a9d3a3a922a9b82b031a564b8f6886980463b8cd538c52`.

Первый kernel с ручным FP16 decoder аварийно завершил PowerVR OpenCL compiler:

```text
LLVM ERROR: out of memory
Allocation failed
```

На плате в этот момент оставалось около 11 GB свободной RAM, системного OOM и
GPU fault не было. Минимальный single-variable probe сохранил ABI, Q1 layout и
loop nest, заменив только ручной decoder на штатный `vload_half` из
`cl_khr_fp16`. Он собрался и дал точный golden. Поэтому production baseline
теперь использует `vload_half`; это обход дефекта/патологии shader compiler, а
не изменение математики.

Квалифицированный sweep (`warmup=1`, `iterations=50`) дал:

| Local size | Host submit/wait median | Device execution | Host total | Golden | Peak temp |
|---:|---:|---:|---:|---|---:|
| 32 | 176.577 ms | invalid selector, не используется | 9043.101 ms | PASS, max error 0 | 42.408 °C |
| **64, corrected rerun** | **176.498 ms** | **176.238 ms** | **9025.294 ms** | **PASS, max error 0** | 42.408 °C |
| 128 | 176.458 ms | invalid selector, не используется | 9037.739 ms | PASS, max error 0 | 41.664 °C |

В первых трёх прогонах runner ошибочно использовал numeric selectors
`0x1280/0x1281`: в OpenCL это `QUEUED/SUBMIT`, а не `START/END`. Поэтому ранее
сохранённые значения `0.0115–0.023 ms` аннулированы: они измеряли очередь до
передачи команды, хотя поле называлось `kernel_event_ms`. Host wall и golden
от этой ошибки не зависят. После исправления на `START=0x1282`, `END=0x1283`
и добавления regression-теста local-size 64 был повторён: настоящее device
execution `176.238 ms` почти полностью объясняет host wall `176.498 ms`.

Разница host wall между 32/64/128 меньше 0.4% и меняет порядок между runs;
победитель local size не квалифицируется. Выбор work-group не устраняет
основную задержку.

Сравнение с CPU operator выполнено на тех же packed weights и восьми ядрах.
Текущий `llama.cpp` CPU_REPACK Q1_0×Q8_0 имеет median `3.850 ms`, а PowerVR
Q1_0×F32 — `176.498 ms`, то есть GPU baseline примерно в **45.8 раза
медленнее** по наблюдаемой задержке одной операции. Это не строгое сравнение
одинаковой арифметики: модель квантует activation в Q8_0, тогда как E007
принимает F32. Поэтому `golden_pass` E007 доказывает корректность только
Q1_0×F32 kernel, но ещё не отсутствие потери качества полной Bonsai.

Вывод: E007 нельзя интегрировать в `llama.cpp`. Результат отсекает стратегию
«один GPU dispatch на один GEMV» и оставляет смысл только у более крупного
fused graph/bundle, способного амортизировать очередь и синхронизацию.

## Следующий эксперимент

Следующий диагностический A/B не меняет kernel: контроль синхронизируется после
каждого enqueue, кандидат ставит серию одинаковых kernels в очередь и делает
один `clFinish`. Corrected `START→END` уже показывает, что почти все 176 ms —
реальное исполнение, поэтому большой выигрыш от batching теперь маловероятен;
эксперимент нужен как проверка queue semantics и возможных пузырей. Даже
положительный результат не станет ускорением модели: для
интеграции затем потребуются Q1_0×Q8_0 semantics, передача промежуточных
activation на device без D2H между слоями и full-model exact golden.
