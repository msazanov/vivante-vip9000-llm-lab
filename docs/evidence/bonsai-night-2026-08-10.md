# Bonsai-27B Q1_0: ночная серия CPU, PowerVR и VIP9000

Дата начала: 2026-08-10 19:39 UTC

Плата: Orange Pi Zero 3W, Allwinner A733, 12 GB RAM

Статус: измерения продолжаются; документ дополняется только подтверждёнными
результатами.

## Что именно оптимизируется

Главная метрика — скорость полного autoregressive decode одной
`Bonsai-27B-Q1_0.gguf` в токенах/с. Модель имеет SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`,
размер `3,803,452,480` байт и запускается закреплённой PrismML-версией
`llama.cpp` commit `38c66ad0241da4f9fcce541cda8edc219086cec5`, build 9594.

Нельзя складывать CPU, GPU и NPU microbenchmarks в одну «скорость модели».
Результат считается ускорением только после полного decode и точного golden.

## Термины простыми словами

- **GEMV** (`General Matrix–Vector multiplication`) — умножение большой
  матрицы весов на небольшой вектор активаций. При генерации одного токена
  Bonsai выполняет сотни таких операций; они читают почти все веса и поэтому
  сильнее всего зависят от DDR bandwidth.
- **Q1** — бинарные по знаку веса, объединённые в блоки по 128 значений. Один
  блок занимает 18 байт: 16 байт знаков и 2 байта масштаба. Их нельзя заранее
  превращать в 128 INT8 или FP16 значений в DDR: это уничтожит экономию
  трафика.
- **H2D** (`host to device`) — подготовка/передача данных от CPU к GPU/NPU.
  На этой плате память физически общая, но операция всё равно может включать
  staging copy, cache flush и синхронизацию.
- **run/device time** — время исполнения kernel или NBG на устройстве. Оно не
  включает автоматически все действия host и поэтому не равно latency
  токена.
- **D2H** (`device to host`) — возврат результата к CPU, включая возможное
  копирование, cache invalidate и ожидание.
- **golden** — заранее полученный правильный результат эталонного CPU пути.
  Для полного запуска сравниваются идентификаторы токенов и текст; для
  operator — числовой вектор с заранее заданным допуском.
- **offload** — сознательный перенос выбранных тензоров/операций на GPU или
  NPU.
- **fallback** — выполнение неподдержанной операции на CPU. Скрытый fallback
  опасен: лог может говорить о GPU/NPU, хотя основная работа осталась на CPU.
- **tg32/tg128** — генерация 32/128 токенов в `llama-bench`. `tg32` — быстрый
  отбор, `tg128` — итоговое подтверждение.
- **CV** — коэффициент вариации, то есть population stddev (делитель `n`),
  делённый на среднее. Малый CV
  показывает, что repetitions устойчивы.

## Cohort перед нагрузкой

Read-only проверка в `2026-08-10T19:39:57Z` подтвердила:

- kernel `6.6.98-sun60iw2`, Debian 12 AArch64;
- source точно на commit `38c66ad0241da4f9fcce541cda8edc219086cec5`;
- SHA и размер модели совпадают с закреплёнными значениями;
- fan service `a733-fan-trip-30c.service` активен;
- вентилятор во время нагрузки находится в состоянии `4/4`;
- доступно около 11.96 GB RAM; swap total 6.23 GB, used 0;
- CPU governor `ondemand`, A55 максимум 1.794 GHz, A76 2.002 GHz;
- NPU работает на 1.008 GHz;
- до теста thermal zones находились примерно в диапазоне 30.8–34.7 °C;
- конкурирующего процесса `llama-bench/cli/completion` не было.

Команда `llama-bench --version` в build 9594 не поддерживается и завершилась
ошибкой аргумента. Это не ошибка модели: identity runtime подтверждена source
commit и полями `build_commit/build_number` в JSON каждого benchmark.

## Свежий CPU reference

Run id: `cpu-a55-6-p50-tg32-r3-001`

Target directory:
`/home/orangepi/vip9000-lab/runs/bonsai-night-20260810/cpu-a55-6-p50-tg32-r3-001`

Параметры: A55 CPU 0–5, 6 threads, strict mask `0x3f`, poll 50, mmap,
`-ngl 0`, `tg32`, 3 repetitions. Workload был обёрнут profiled command и
fail-closed thermal guard `85 °C` с интервалом 100 ms.

Результат полного decode:

| Sample | токенов/с |
|---:|---:|
| 1 | 0.722014 |
| 2 | 0.722008 |
| 3 | 0.722256 |
| **Среднее** | **0.722093** |

`stddev=0.000116 ток/с`, приблизительный `CV=0.0160%`. Здесь и далее для
согласованности с summarizer используется population stddev с делителем `n`,
а не sample stddev из поля `llama-bench`. Процесс завершён с
кодом 0; profiler сохранил 1819 samples за 190.704 s. Наблюдавшаяся во время
ранней steady нагрузки температура большого CPU-кластера была около 37.2 °C,
частота A55 держалась на 1.794 GHz, fan `4/4`. Итоговый peak будет вычислен из
полного telemetry при агрегации серии.

Исторический `tg128` reference равен `0.726175 ток/с`. Свежий short result
ниже на 0.56%, что мало, но показывает необходимость сравнивать ночные
кандидаты со свежим cohort, а не только с одной старой цифрой.

## CPU topology screen: первый реальный model-level выигрыш

Все строки ниже используют один CPU binary, одну модель, `tg32 r3`, poll 50,
mmap, `-ngl 0`, strict affinity и одинаковый profiling/thermal contract.

| Topology | Threads / mask | Samples, ток/с | Median, ток/с | К свежему reference | Peak board temp |
|---|---|---|---:|---:|---:|
| A55×4 | 4 / `0x0f` | 0.510137; 0.510197; 0.510179 | 0.510179 | −29.34% | 52.018 °C |
| A55×6 reference | 6 / `0x3f` | 0.722014; 0.722008; 0.722256 | 0.722014 | 0.00% | 57.846 °C |
| A55×6 + A76×1 | 7 / `0x7f` | 0.823141; 0.820784; 0.816766 | 0.820784 | +13.68% | 65.224 °C |
| A55×6 + A76×2 | 8 / `0xff` | 0.866365; 0.871746; 0.867775 | 0.867775 | **+20.19%** | 71.685 °C |

`llama-bench` также печатает mean: `0.510171`, `0.722093`, `0.820230` и
`0.868628 ток/с` соответственно. В таблице используется median, вычисленный
из raw samples единым агрегатором.

Это полный decode Bonsai, а не operator microbenchmark. Вывод о bottleneck
уточнён: четыре A55 слишком медленны, шесть A55 ещё не насыщают путь полностью,
и добавление каждого A76 продолжает повышать tok/s. Значит текущий path
одновременно ограничен packed-weight traffic и Q1 unpack/dot вычислениями.

All-core screen был подтверждён длинным run
`cpu-all8-p50-tg128-r5-001`: samples
`0.857795 / 0.860580 / 0.856949 / 0.860712 / 0.865216 ток/с`, median
`0.860580`, mean `0.860250`, CV `0.337%`. Это на **18.51%** быстрее
исторического A55×6 `tg128 r5` reference `0.726175 ток/с`. Длина измерения и
число repetitions у reference/candidate совпадают.

Peak температуры в подтверждающем run составил `73.809 °C`, то есть сохранился
запас `11.191 °C` до fail-closed limit. Оба CPU policy держали максимумы
`1.794/2.002 GHz`, состояния `cpufreq` cooling оставались 0, вентилятор доходил
до `4/4`. SwapFree на всех samples был ровно `6,081,100 KiB`, child RSS-HWM —
`7,339,184 KiB`. Thermal abort, OOM, swap activity и частотный throttle не
наблюдались.

Отдельный CPU control тем же Vulkan-enabled binary, но с `-ngl 0 -dev none`,
дал samples `0.865955 / 0.870702 / 0.879984`, median `0.870702 ток/с`. Разница
`+0.34%` относительно CPU-build all-core median находится внутри вариации и
не считается эффектом Vulkan.

### Poll screen

Чтобы не загружать модель пять раз, один all-core bundle держал модель resident
и последовательно менял только `poll=0,1,25,50,100`. Для быстрого отбора
использовался `tg16 r3`, поэтому variance выше и эти числа не заменяют `tg128`.

| Poll | Samples, ток/с | Median | Mean | Вывод screen |
|---:|---|---:|---:|---|
| 0 | 0.840217; 0.903302; 0.826571 | 0.840217 | 0.856697 | высокий разброс |
| 1 | 0.834869; 0.900960; 0.832332 | 0.834869 | 0.856054 | высокий разброс |
| 25 | 0.885945; 0.837920; 0.897094 | 0.885945 | 0.873653 | кандидат |
| 50 | 0.897988; 0.824411; 0.890474 | 0.890474 | 0.870958 | текущий контроль |
| 100 | 0.878410; 0.829134; 0.871066 | 0.871066 | 0.859537 | хуже 25/50 |

`poll` управляет временем busy-wait worker threads перед сном. Низкие значения
могут добавлять wake-up latency, высокие — бесполезно занимать CPU. Короткий
screen не различает 25 и 50 надёжно: их mean/median меняют порядок, а samples
шумны. Поэтому canonical run сохраняет ранее проверенный default 50; отдельный
длинный alternating A/B потребуется перед заявлением микровыигрыша poll 25.

### Exact golden полного генератора

`llama-bench` проверяет скорость, но не сохраняет последовательность токенов.
Поэтому отдельно запущен `llama-completion` с одним prompt, `ctx=512`,
`batch/ubatch=512`, 32 output tokens и полностью deterministic greedy sampling:
`seed=0`, `temp=0`, `top-k=1`, `top-p=1`, `min-p=0`, `repeat-penalty=1`.
Единственная переменная — CPU topology:

- reference: CPU 0–5, 6 threads, masks `0x3f`;
- candidate: CPU 0–7, 8 threads, masks `0xff`.

Оба workload завершились кодом 0 под profiler и thermal guard. Каждый stdout
имеет ровно 115 bytes и SHA-256
`a9d86b7d298be35cf27a156be87f9aeedbae3602b877e74ced0fdc75b41c4a5f`;
`cmp` вернул 0. Следовательно, all-core candidate имеет `exact_match=true`:
изменилось только распределение той же математики по CPU, а не веса, sampling
или ответ модели.

Две подготовительные попытки не приняты в golden:

1. `golden-a55-6-greedy32-001` завершился кодом 1, потому что
   `llama-completion` этой версии не принимает bench-only аргумент `-mmp`;
2. `golden-a55-6-greedy32-002` завершился кодом 0, но `--log-disable` подавил
   generated stdout, поэтому сравнивать пустые файлы было бы ложным PASS.

Их run directories сохранены на target как отрицательные данные; в qualified
график они не входят.

## PowerVR: что подтверждено на железе

GPU — отдельный PowerVR B-Series BXM-4-64 MC1 на `/dev/dri/renderD128`, driver
`pvrsrvkm`; он не использует CPU cores для исполнения shaders. Vulkan сообщает
API 1.3.277 и proprietary driver `24.2@6603887`.

Две строки GPU0/GPU1 имеют одинаковый UUID
`33362035-3620-3130-3420-313833000000`. Это дублирование одного физического
GPU в loader, а не два независимых ускорителя. Поэтому multi-device split на
индексах `0,1` запрещён; первый runtime smoke выбирает только device 0.

Прямой `/usr/lib/libPVROCL.so.1` подтверждает OpenCL 3.0:

- один PowerVR platform и один BXM-4-64 device;
- `compute_units=1`, максимум work-group 512, заявленная частота 600 MHz;
- 4 KiB local memory, `HOST_UNIFIED_MEMORY=1`;
- coarse-grain buffer SVM, без fine-grain system SVM;
- `cl_khr_fp16`, `cl_khr_integer_dot_product`, `cl_khr_command_buffer` и
  dma-buf import extensions заявлены.

Generic `libOpenCL.so.1` возвращает `CL_PLATFORM_NOT_FOUND_KHR`, потому что
ICD registration отсутствует. Это не отсутствие GPU OpenCL runtime; пока
нужно открывать `libPVROCL.so.1` напрямую.

### Короткие OpenCL probes

Read-only probes не меняли системные файлы и использовали resident buffers:

| Kernel | Golden | Host median | Device median | Вывод |
|---|---|---:|---:|---|
| FP32 VADD, 65536 элементов | все `3.75`, PASS | 0.763 ms | 0.005 ms | host/sync floor доминирует |
| Q1-like signs×Q8, M=1024,K=512 | `-1536.0`, PASS | 2.218 ms | 0.013 ms | kernel корректен, но мал для amortization |

Device event нельзя выдавать за end-to-end: во втором probe разница между
`0.013 ms` на device и `2.218 ms` на host состоит из submit, queue, finish,
readback и прочего runtime overhead. Именно поэтому сотни маленьких dispatches
на токен заведомо плохой дизайн.

Несмотря на UMA, `clEnqueueMapBuffer` вернул `CL_INVALID_VALUE (-30)` для
проверенных default/USE_HOST_PTR/ALLOC_HOST_PTR путей. Zero-copy не доказан;
каждый будущий эксперимент обязан отдельно измерять H2D, kernel, D2H и total.

Закреплённая PrismML-ветка уже имеет Vulkan shaders для `Q1_0`, поэтому первым
full-model GPU экспериментом будет отдельная `GGML_VULKAN=ON` сборка и честный
`-ngl 0` против `-ngl 99` A/B одним binary.

### Vulkan build и первый full-model gate

Создан отдельный build
`/home/orangepi/vip9000-lab/build/vulkan-38c66-powervr` с теми же
`armv8.2-a+dotprod`, `GGML_NATIVE=OFF`, `CPU_REPACK=ON`, но
`GGML_VULKAN=ON`. Reference binary не изменён. Сборка `llama-bench` и
`llama-completion` завершилась с кодом 0 за `1123.736 s`, под profiler и
thermal guard; ошибок компилятора не было. `llama-bench --list-devices`
сообщил ровно один backend:

```text
Vulkan0: PowerVR B-Series BXM-4-64 MC1
uma: 1, fp16: 1, shared memory: 16384, int dot: 0, matrix cores: none
```

Первый full-model `-ngl 99`, `tg1` завершился с кодом 1 до загрузки модели:

```text
ggml_vulkan: Error: Shared memory size too small for matrix multiplication.
llama_bench: error: failed to load model '.../Bonsai-27B-Q1_0.gguf'
```

Kernel log не показал GPU fault или OOM; run длился 1.660 s. Причина найдена
в pinned `ggml-vulkan.cpp`: при загрузке shaders backend проходит все
`GGML_TYPE_COUNT` и безусловно бросает exception, если smallest matmul tile не
помещается для любого типа. На 16 KiB PowerVR один из IQ1 LUT paths превышает
лимит, хотя Bonsai использует другой `Q1_0` path. Подготавливается минимальный
type-specific patch: неподдержанный тип отключается, а Q1_0 допускается только
если его собственный shared-memory расчёт проходит. Это устранение ложного
global gate, а не ослабление golden/fallback требований.

Patch SHA-256
`06a35ce52906b6ab379343d5c9a0361f4a7dd151b52969211ad63424a15e1ea2`
был применён к target checkout с чистым pinned HEAD. Инкрементальная сборка
завершилась с кодом 0 за `143.933 s`. Повторный `-ngl 99`, `tg1` прошёл
глобальный shared-memory gate и дошёл до warmup, но затем driver отклонил
конкретный Q1 pipeline:

```text
ggml_vulkan: Compute pipeline creation failed for mul_mat_vec_q1_0_f32_f32
ggml_vulkan: vk::Device::createComputePipeline: ErrorUnknown
terminate called after throwing an instance of 'vk::SystemError'
```

Run завершился `134` за `9.211 s`. В kernel log по-прежнему нет GPU fault или
OOM. Это отделяет две независимые несовместимости:

1. исправленный global type loop ошибочно блокировал Q1_0 из-за IQ1 LUT;
2. сам универсальный `mul_mat_vec_q1_0_f32_f32` SPIR-V не принимается PowerVR
   pipeline compiler с текущими specialization/subgroup параметрами.

Плата сообщает `warp/subgroup size=1`, а pinned backend строит workgroup
constants непосредственно из этого значения; одновременно `glslc` не дал
integer-dot shader variant. Дальнейшее изменение Vulkan shader без маленького
operator golden слишком рискованно. Поэтому full-model Vulkan кандидат пока
имеет статус **failed capability gate**, `tokens/s` отсутствует и CPU fallback
не выдаётся за GPU result. Следующая GPU ветка — прямой OpenCL C 1.2 packed-Q1
operator через доказанный `libPVROCL.so.1`.

## VIP9000: текущий точный статус

VIP9000 исполняет заранее скомпилированные UINT8 NBG через VIPLite. Resident
ShuffleNet доказал ABI/runtime и дал измеримые H2D/run/D2H, но это другая сеть,
поэтому не является результатом Bonsai.

Для Bonsai выбран carrier `Q1_VIP_16x128`: `UINT8` описывает транспорт сырых
байтов, а не восемь бит на один вес. Execution tile содержит 16 строк по 128
весов: 256 байт sign bits + 32 байта FP16 scales = 288 байт. Распаковка должна
происходить в PPU/EVIS registers или доказанном VIP SRAM.

Реальный NPU Q1 run сейчас блокирует host compiler mismatch:

- AcuityLite 6.51;
- generated graph ожидает TIM-VX 1.1.30;
- соседние VSI NN headers сообщают 1.2.30;
- `vxQueryHardwareCaps` расходится по размерам structures;
- ABI shim проходит ранний status, но C0 Add graph падает при создании node.

Следовательно, непроверенный NBG не передаётся на target. Это точный blocker
toolchain, а не доказательство, что VIP9000 не способен выполнять packed Q1.
Production priority остаётся: direct packed Q1 GEMV, затем resident multi-
output bundles для уменьшения `497 → 257` submit boundaries.

## CPU compiler A/B: `+fp16` не ускоряет Bonsai Q1

После отрицательного GPU результата собрана отдельная, не заменяющая reference
сборка `/home/orangepi/vip9000-lab/build/cpu-38c66-fp16`. Единственная
существенная переменная относительно CPU control —
`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`. CMake подтвердил
`HAVE_FP16_VECTOR_ARITHMETIC`, но `SVE` и `i8mm` недоступны. Исходный commit
остаётся `38c66ad`; suffix `-dirty` относится к уже документированному
type-specific Vulkan patch, а Vulkan в этой сборке выключен.

Сначала реальный `blk.0.ffn_gate.weight` прошёл тем же CPU_REPACK Q1_0×Q8_0
путём. Output SHA-256 совпал с control:
`359f62fb43af0150cd2043b26c582e97562aaaf2f8a7134ce34ed144dbf1685f`.
Короткий operator median, однако, ухудшился с `3.850` до `6.834 ms` и имел
высокий CV, поэтому использовался только как correctness gate, не как итоговая
оценка compiler flag.

Решающий all-core full-model screen (`tg32 r3`, poll 50) дал samples
`0.857072 / 0.857097 / 0.880181 ток/с`, median `0.857097`, mean `0.864783`.
Обычная all-core сборка при тех же параметрах имеет median `0.867775`, mean
`0.868629`. Следовательно, `+fp16` изменил median на **−1.23%**, mean на
**−0.44%**. Peak температуры `71.331 °C`, частоты достигали
`1.794/2.002 GHz`, guard завершился штатно; thermal throttle не объясняет
проигрыш.

Ветка остановлена до `tg128`: требуемого screen-выигрыша `>=2%` нет. Для Q1
decode основные kernels используют packed sign bits и Q8 dot product, поэтому
включение FP16 ISA само по себе ожидаемо не меняет критическую арифметику.

## Custom CPU packed layout: 4x8 проиграл 4x4

Аудит ARM source показал две уже существующие реализации Q1_0×Q8_0 GEMV:
`ggml_gemv_q1_0_4x4_q8_0` и `ggml_gemv_q1_0_4x8_q8_0`. Обе содержат NEON
dotprod path, но selector разрешает 4x8 только при `i8mm`. На A733 есть dotprod,
но нет i8mm, поэтому создан минимальный custom patch, меняющий только trait
для ветки Q1+dotprod. GGUF, логические веса и арифметика не изменены; resident
CPU_REPACK остаётся равен packed Q1 payload без expanded weight copy.

Candidate собран из отдельного clean worktree `38c66ad`. Четыре candidate
runs и четыре control runs чередовались; каждый использовал реальный
`blk.0.ffn_gate.weight`, all-core, warmup 1 и 50 iterations. Во всех runs
candidate output byte-for-byte совпал с control, SHA-256
`359f62fb43af0150cd2043b26c582e97562aaaf2f8a7134ce34ed144dbf1685f`.

| Layout | Run medians, ms | Median-of-medians |
|---|---|---:|
| control 4x4 | 3.850; 3.198; 3.712; 3.743 | **3.727 ms** |
| candidate 4x8 | 4.238; 4.158; 7.253; 3.987 | **4.198 ms** |

4x8 медленнее на **12.62%**. Даже лучший candidate run хуже cohort control
median; full-model `tg32` gate не открыт. Наиболее вероятно, что `vcombine` и
register dependencies 4x8 дороже, чем lane-инструкции 4x4 на этих Cortex cores.

Старый E004 runner печатает статическую identity `4x4` независимо от linked
trait. Поэтому candidate JSONL не считается qualified provenance: actual 4x8
доказывается изолированным source diff/build, но metadata bug зафиксирован, а
не замаскирован. Поскольку performance отрицательный, отдельный runner ради
promotion не создаётся.

## Custom CPU kernel: явный Q1 prefetch ухудшил latency

В принятом `ggml_gemv_q1_0_4x4_q8_0` нет явного prefetch, а disassembly control
не содержит `prfm` для packed weight stream. Создан минимальный patch: перед
обработкой каждого 72-byte `block_q1_0x4` запрашивать block `l+4`, то есть
примерно `288 B` вперёд. Compiler с high-locality hint действительно выпустил
`prfm pldl1keep, [x0, #288]`. Других изменений kernel/layout нет.

Четыре чередующихся control/candidate pairs прошли exact output. Run medians:

| Kernel | Run medians, ms | Median-of-medians |
|---|---|---:|
| control 4x4 | 3.417; 3.444; 6.723; 3.147 | **3.430 ms** |
| prefetch +288 B | 3.862; 7.248; 4.160; 4.004 | **4.082 ms** |

Явный prefetch ухудшил latency на **19.00%** и operator speed примерно на
**15.96%**. Наиболее вероятно, последовательный Q1 stream уже обслуживается
hardware prefetcher, а раннее заполнение L1 вытесняет маленький Q8 activation
или 2 KiB sign lookup table. Full-model gate не открыт; patch отклонён.

### Прямой OpenCL packed-Q1 operator: корректен, но пока непригоден

Для первого real-model operator из GGUF извлечён
`blk.0.ffn_gate.weight`, runner shape `M=17408`, `K=5120`. Packed buffer
занимает `12,533,760 B`, его SHA-256 —
`0f42ca3b81099f540ed67941809ee7fdc0a672563bf852b87db75034135fc6fe`.
Это именно packed Q1_0: expanded INT8/FP16 копия на host не создаётся.
Внутренние staging/expansion proprietary runtime наблюдать пока нельзя, поэтому
они честно помечены `unknown`, а не `false`.

Первая реализация ручного FP16 decoder вызвала падение PowerVR OpenCL compiler
с `LLVM ERROR: out of memory`. Это не был системный OOM: свободно оставалось
около 11 GB, kernel log не содержал OOM/GPU fault. В минимальном probe была
заменена ровно одна переменная: ручной decoder на builtin `vload_half` из
`cl_khr_fp16`. После этого тот же loop nest собрался и побитово совпал со
scalar Q1×F32 golden. Этот driver-safe вариант закреплён тестом.

Квалифицированный sweep одного и того же реального тензора:

| Local size | Host median одного GEMV | Device execution | Полный run | Golden | Peak temp |
|---:|---:|---:|---:|---|---:|
| 32 | 176.577 ms | invalid selector, отброшено | 9043.101 ms | PASS, max error 0 | 42.408 °C |
| **64, corrected rerun** | **176.498 ms** | **176.238 ms** | **9025.294 ms** | **PASS, max error 0** | 42.408 °C |
| 128 | 176.458 ms | invalid selector, отброшено | 9037.739 ms | PASS, max error 0 | 41.664 °C |

При аудите выяснилось, что первые runs запрашивали OpenCL profiling selectors
`0x1280/0x1281`. Стандарт определяет их как `QUEUED/SUBMIT`; правильные
`START/END` — `0x1282/0x1283`. Поэтому старые `device-event=0.0115–0.023 ms`
аннулированы: это была задержка до submit, ошибочно подписанная как execution.
Host wall, output и golden оставались корректными.

После TDD-исправления local-size 64 повторён. Реальные transfer/device значения:
weights H2D `9.022 ms` device / `9.292 ms` host, activation H2D
`0.076/0.265 ms`, kernel median `176.238/176.498 ms`, D2H
`0.763/0.933 ms`. Теперь device и host согласуются: почти вся задержка — само
выполнение kernel, а не `clFinish`. Тест жёстко фиксирует numeric values
`START=0x1282` и `END=0x1283`, чтобы ошибка не вернулась.

CPU_REPACK на тех же Q1 weights и восьми ядрах дал median `3.850 ms` на GEMV.
PowerVR baseline медленнее примерно в **45.8 раза**, поэтому перенос одного
GEMV на GPU сейчас уменьшит, а не увеличит скорость модели. Арифметика также
ещё различается: CPU model path использует Q1_0×Q8_0, эксперимент GPU —
Q1_0×F32. Точный operator golden GPU доказывает корректность его собственной
семантики, но не full-model quality. Интеграция E007 в `llama.cpp` отклонена.

Следующая диагностика проверит queue batching и пузыри между kernels. Corrected
profiling уже показывает, что крупного выигрыша от одной синхронизации ждать
не следует: `START→END` почти равно host wall. Единичные GPU dispatches уже
экспериментально исключены; fused kernel должен ускорить саму арифметику и
reuse activation, а не только убрать submit overhead.

## Следующие строки этой серии

CPU topology screen, canonical `tg128 r5`, deterministic exact golden, два
Vulkan capability gates и прямой PowerVR OpenCL packed-Q1 operator завершены.
Machine-readable summary и XY-график завершены. Queue-batching diagnostic
остаётся низкоприоритетной проверкой: corrected OpenCL profiling уже показал,
что `176.238/176.498 ms` приходится на device execution, а не submit overhead.

## Наглядный итог серии

Machine-readable набор находится в
`benchmarks/results/bonsai-night-a733-20260810/summary.json`, детерминированный
SVG — в `benchmarks/charts/bonsai-night-a733-20260810.svg`.

Агрегатор fail-closed принимает только закреплённый путь и identity Bonsai,
сверяет `n_prompt`, `n_gen` и число repetitions между command и JSONL,
отвергает отрицательные частоты/SwapFree/RSS и требует строгий lifecycle
`start → inventory → child_started → sample+ → child_exit → exit` с
монотонными timestamps. Генератор графика отдельно закрепляет SHA-256 модели,
commit runtime и SHA-256 обоих exact-golden stdout; подменённый provenance не
попадает даже в SVG.

Для новых CPU-серий путь без ручной правки schema выглядит так:

```bash
python3 tooling/summarize_bonsai_night.py RUN_REFERENCE RUN_CANDIDATE \
  --reference-run-id RUN_REFERENCE_ID \
  --candidate-run-id RUN_CANDIDATE_ID \
  --golden-reference-stdout GOLDEN_REFERENCE/stdout.log \
  --golden-candidate-stdout GOLDEN_CANDIDATE/stdout.log \
  --output summary.json
python3 tooling/generate_bonsai_night_chart.py summary.json --output chart.svg
```

Без полного набора четырёх `chart-ready` аргументов summarizer сохраняет raw
full-model summary с `quality.status=not_checked`; chart намеренно его не
принимает как qualified результат. Интеграционный тест запускает оба инструмента
подряд и исключает повторение несовместимых контрактов.

График намеренно разделён на три области:

1. верхняя шкала показывает только `tg128 r5` full-model decode, для которого
   deterministic golden совпал byte-for-byte;
2. полые маркеры показывают все `tg32/tg16` screens, но не выдают их за
   qualified quality result;
3. CPU/GPU operator milliseconds и failed GPU/NPU gates перечислены отдельно
   и не смешиваются с токенами/с.

Для каждой full-model точки подписаны и скорость, и обратная величина — среднее
время одного decode-токена `1000 / tok/s`. У эталона это `1377.1 ms/token`, у
all-core candidate — `1162.0 ms/token`. Разность не является профилем одного
конкретного токена: это средняя стоимость по серии `tg128`.

Если ни один вариант не ускорит модель, это будет записано как измеренный
отрицательный результат, без переноса operator/device времени в выдуманные
`tokens/s`.
