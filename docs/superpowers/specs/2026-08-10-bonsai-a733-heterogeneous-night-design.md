# Ночная оптимизация Bonsai-27B Q1_0 на A733, VIP9000 и PowerVR

Дата: 2026-08-10

Статус: допущено к автономному выполнению владельцем проекта. Вопросы во время
ночного цикла не задаются; все допущения, команды, отрицательные результаты и
сырые артефакты фиксируются.

## Цель

Найти воспроизводимую конфигурацию, которая повышает устойчивую скорость
single-stream decode закреплённой `Bonsai-27B-Q1_0.gguf` относительно
сохранённого CPU reference `0.726175 ток/с`, не меняет логические веса и
проходит детерминированный golden-тест генерации.

Это исследование оптимизирует полный путь одного токена, а не отдельный
синтетический kernel. Operator-level время, NPU device time и model-level
`tokens/s` сохраняются раздельно и не подменяют друг друга.

## Неподвижные условия

- Модель: `/home/orangepi/vip9000-lab/models/Bonsai-27B-Q1_0.gguf`.
- Размер: `3,803,452,480` байт.
- SHA-256:
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Runtime: PrismML `llama.cpp`, commit
  `38c66ad0241da4f9fcce541cda8edc219086cec5`, build number `9594`.
- Главная метрика: медиана `tg128`, токенов/с; screen использует `tg32` и
  никогда не объявляется окончательным результатом.
- Quality gate: тот же prompt, seed и greedy sampling; токены и итоговый текст
  должны совпасть с CPU golden. Расхождение означает неускорение, а ошибку.
- Thermal gate: немедленное завершение workload при `85 °C`; сохраняются
  trace температуры, частот, RSS, swap и throttle-событий.
- Каждый sustained workload запускается последовательно. Одновременные CPU,
  GPU и NPU weight streams запрещены без отдельного доказательства выигрыша.
- В этой серии запрещены записи в DDR PLL, U-Boot, device tree и постоянные
  системные настройки.
- Полные веса не разворачиваются из Q1 в INT8, FP16 или FP32 в DDR.

## Исходная гипотеза

Decode читает не менее `3,603,087,360` packed Q1 bytes на токен. Сохранённый
A55 result соответствует примерно `2.616 GB/s` полезного Q1-потока, поэтому
главный дефицит — shared-memory bandwidth и лишние проходы по весам. Победа
возможна, только если backend:

1. читает каждый Q1 block в packed виде;
2. распаковывает sign bits в регистрах, execution tile или доказанном SRAM;
3. не создаёт дополнительную полную копию весов;
4. не добавляет больше синхронизаций и копирований, чем экономит вычислений;
5. подтверждает выигрыш wall-clock полного decode.

## Проверенные возможности вычислителей

### CPU A733

Шесть Cortex-A55 имеют максимум 1.794 GHz, два Cortex-A76 — 2.002 GHz. Все
ядра поддерживают ARM dot product. Текущий Q1 path уже использует
`CPU_REPACK` и ARM `Q1_0 4x4 DOTPROD`; обычное включение repack не является
новой оптимизацией. Сохранённые full-model результаты:

- A55, CPU 0–5, 6 threads: `0.726175 ток/с` decode;
- A76, CPU 6–7, 2 threads: `0.649836 ток/с` decode;
- A76 быстрее на prefill, но проигрывает A55 на bandwidth-bound decode.

Первый CPU рычаг — topology/polling без изменения арифметики. Проверяются
A55×4, A55×6, A76×1, A76×2, A55+один A76 и все восемь ядер. После выбора
топологии меняется только `poll`. `ubatch` относится преимущественно к
prefill и проверяется лишь после decode screen.

### PowerVR BXM-4-64 MC1

На плате подтверждён отдельный integrated GPU с proprietary PowerVR Vulkan
driver `24.2@6603887`, Vulkan API 1.3.277. Два Vulkan device index сообщают
одинаковое имя и идентификаторы; до обратного доказательства это считается
двойным перечислением одного GPU разными ICD, а не двумя GPU.

Текущий CPU binary собран с `GGML_VULKAN=OFF`. В закреплённом исходнике есть
Vulkan paths для `GGML_TYPE_Q1_0`, включая `mul_mat_vec_q1_0`, matmul,
dequantize, get-rows и copy shaders. Поэтому создаётся отдельная сборка, не
затрагивающая CPU reference. Сначала проверяются build, device enumeration и
короткий полный-model smoke. Затем `-ngl 99` сравнивается с `-ngl 0` на одной
сборке. Ускорение принимается только при фактическом GPU offload, отсутствии
CPU fallback, точном golden и меньшем host wall-clock.

На UMA `H2D` и `D2H` не обязательно означают физическую шину PCIe: это
логические границы host-to-device и device-to-host, которые могут включать
flush/invalidate, staging или копирование внутри общей DDR. Они всё равно
измеряются, потому что конкурируют за ту же пропускную способность.

### Vivante VIP9000

Target доказал исполнение resident UINT8 NBG через VIPLite, но это не
Bonsai. Известный ShuffleNet измеряет ABI и overhead: его нельзя выдавать за
LLM `tokens/s`. Публичный target API загружает заранее скомпилированный NBG и
не принимает исходник OpenCL kernel напрямую.

Основной NPU путь остаётся `Q1_VIP_16x128`: `UINT8` tensor используется как
byte carrier, но 128 Q1 weights занимают исходные 18 байт, а sign bits
раскрываются только внутри PPU/EVIS tile. Приоритеты:

1. direct packed Q1 PPU/EVIS GEMV;
2. resident multi-output bundles, сокращающие примерно `497 → 257` submits;
3. PPU unpack в VIP SRAM и native NN только при нулевом expanded DDR traffic.

Сегодня реальный Q1 NBG закрыт несовместимостью host compiler bundle:
AcuityLite 6.51 генерирует TIM-VX 1.1.30 graph рядом с headers 1.2.30, а
OpenVX ABI расходится по размерам capability structures. До прохождения
минимального Add C0 и Q1 golden непроверенный NBG на плату не отправляется.
Отрицательный compiler gate документируется как blocker, а не как вывод о
непригодности VIP9000.

## Последовательность эксперимента

### Фаза 0 — cohort и одинаковые условия

Перед нагрузкой проверяются hash модели, binary version, governor/frequencies,
fan service, доступная RAM/swap, отсутствие конкурирующего `llama` процесса и
Vulkan/NPU identities. Создаётся новый уникальный run directory. Сохранённый
`0.726175 ток/с` остаётся историческим reference, но для честного ночного A/B
первым выполняется свежий A55×6 `tg32`, 3 повтора.

### Фаза 1 — CPU screen

Все варианты используют одну модель, `-p 0 -n 32 -r 3`, batch/ubatch 512,
`-ngl 0`, mmap и одинаковый cooldown. Меняются только process mask, model
mask, thread count и затем polling. Победитель short screen подтверждается
`tg128`, 5 повторов.

При разнице меньше 2% вариант не считается выигрышем без дополнительного
повтора: такой зазор сравним с системным шумом и collector overhead. Восьми-
ядерный вариант отдельно помечается, потому что profiler тоже использует CPU.

### Фаза 2 — Vulkan Q1 full-model A/B

Создаётся отдельный build directory с `GGML_VULKAN=ON`, теми же CPU flags и
без BLAS/CUDA/OpenCL. Сначала `llama-bench --list-devices`, затем короткая
загрузка/`tg1`, после неё `tg32` с `-ngl 99`. Контроль `-ngl 0` запускается
тем же Vulkan binary, чтобы отделить влияние новой сборки от GPU offload.

Если model load вызывает OOM, swap, driver reset, unsupported Q1 op или
скрытый fallback, попытка останавливается и фиксируется. Частичный offload
может быть полезен только если профайлер показывает меньший wall-clock полного
токена; перенос мелких операций сам по себе успехом не считается.

### Фаза 3 — безопасный CPU compiler A/B

Только если GPU не дал принятого результата, создаётся отдельная CPU сборка с
`armv8.2-a+dotprod+fp16`. Она проходит operator golden, затем тот же short
screen. Никакая сборка не заменяет reference binary. Source-level prefetch или
новый kernel допускается только после baseline и отдельного failing test.

### Фаза 4 — подтверждение победителя

Лучший кандидат выполняет canonical `tg128`, 5 повторов, затем
детерминированный `llama-completion`. Сохраняются:

- raw stdout/stderr и JSONL `llama-bench`;
- host timing и параметры команды;
- 100 ms telemetry температуры, CPU/NPU частот, RSS и swap;
- thermal-guard trace;
- model/runtime/build hashes и фактический backend/offload;
- golden tokens/text и результат exact comparison.

## Критерии решения

Принятый speedup обязан одновременно удовлетворять всем условиям:

- полный Bonsai decode, не operator microbenchmark;
- медиана `tg128` выше свежего cohort reference;
- предпочтительно `>= 5%`; результат `2–5%` требует повторного A/B;
- exact golden tokens;
- swap delta 0, нет OOM, driver reset, thermal abort или throttle;
- нет expanded full-weight copy и неуказанного CPU fallback;
- все команды и сырые артефакты воспроизводимы.

Если ни один кандидат не быстрее, результат всё равно полезен: в документации
останется измеренный предел CPU topology, цена Vulkan offload и точный NPU
compiler blocker. Запрещено объединять улучшения разных запусков в
гипотетическую итоговую скорость.

## Выходные артефакты

Итог оформляется по-русски в трёх уровнях:

1. machine-readable raw runs на плате;
2. компактный JSON/CSV summary в репозитории;
3. понятный XY-график: X — варианты backend/topology, Y — полный decode
   `tokens/s`, отдельная линия/reference для golden CPU и явные error bars.

На графике NPU operator/device microseconds не смешиваются с model tok/s.
ShuffleNet остаётся только capability/overhead evidence.
