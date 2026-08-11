# План ночной оптимизации Bonsai-27B Q1_0

> **Для agentic workers:** выполнять задачи последовательно через
> `superpowers:subagent-driven-development`; для любого изменения поведения
> сначала использовать `superpowers:test-driven-development`.

**Цель:** измерить и, если железо позволяет, повысить скорость полного
single-stream decode `Bonsai-27B-Q1_0.gguf` на A733 за счёт CPU topology,
нативного Vulkan Q1 backend и проверяемого NPU пути без потери качества.

**Архитектура:** CPU reference остаётся неизменным. Каждая стратегия получает
отдельный build/run directory и проходит одинаковую лестницу `capability →
smoke → tg32 screen → tg128 confirm → deterministic golden`. Все sustained
нагрузки обёрнуты существующими `profile_command.py` и
`thermal_exec_guard.py`; только full-model `llama-bench` сравнивается в
`tokens/s`.

**Стек:** pinned PrismML llama.cpp `38c66ad`, CMake/Ninja или Make, AArch64
GCC 12.2, ggml CPU/Vulkan, proprietary PowerVR Vulkan 1.3 driver, VIPLite
2.0.3.2, Python 3.11, JSONL/CSV/SVG, `unittest`, SSH.

## Глобальные ограничения

- Exact model SHA-256:
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Reference binary:
  `/home/orangepi/vip9000-lab/build/cpu-38c66-native-git/bin/llama-bench`.
- Main result: `tg128`, пять повторов; `tg32`, три повтора — только screen.
- Thermal abort `85 °C`, interval 100 ms, swap delta 0.
- Запуски только последовательно; между sustained runs проверить температуру.
- Не менять DDR PLL, U-Boot, DT, governor policy или постоянные system files.
- Не считать ShuffleNet, operator throughput или NPU device time скоростью
  Bonsai.
- Не объявлять speedup до exact token/text golden.

---

## Задача 1 — зафиксировать cohort и свежий CPU reference

**Артефакты:**

- создать на target новый каталог
  `/home/orangepi/vip9000-lab/runs/bonsai-night-20260810/`;
- создать локально
  `benchmarks/results/bonsai-night-20260810/cohort.json`;
- создать локально
  `docs/evidence/bonsai-night-2026-08-10.md`.

- [ ] Read-only проверить SHA модели, commit/version binary, CPU topology,
  governor/frequencies, NPU frequency, Vulkan summary, fan service, RAM/swap и
  отсутствие другого `llama` workload.
- [ ] Сохранить точные команды и UTC timestamps в `cohort.json`.
- [ ] Запустить свежий A55×6 short reference:

```bash
taskset -c 0-5 llama-bench -m MODEL -p 0 -n 32 -r 3 --delay 1 \
  -b 512 -ub 512 -t 6 -C 0x3f --cpu-strict 1 --poll 50 \
  -ngl 0 -mmp 1 --progress -o jsonl
```

- [ ] Обернуть команду profiler и thermal guard, сохранить raw stdout/stderr,
  telemetry, metadata и guard trace.
- [ ] Проверить ненулевой exit code, thermal abort, swap delta и полноту трёх
  repetitions; fail closed.
- [ ] Записать median/stddev/CV `tg32`, peak temperature и effective clocks.

## Задача 2 — CPU topology и polling screen

**Артефакты:**

- target raw runs `cpu-{a55-4,a55-6,a76-1,a76-2,mix-7,all-8}`;
- локальный `benchmarks/results/bonsai-night-20260810/cpu-screen.jsonl`.

- [ ] Последовательно запустить при неизменных остальных аргументах:
  `0-3/t4/0x0f`, `0-5/t6/0x3f`, `6/t1/0x40`, `6-7/t2/0xc0`,
  `0-6/t7/0x7f`, `0-7/t8/0xff`.
- [ ] Между runs убедиться, что плата ниже thermal start threshold и swap 0.
- [ ] Для all-core отдельно сохранить collector overhead caveat.
- [ ] На лучшей topology выполнить `poll=0,1,25,50,100`, меняя только poll.
- [ ] Отбраковать варианты с CV/thermal/fault нарушением; выбрать по median
  `tg32`, а не по лучшему одиночному sample.
- [ ] Если разница меньше 2%, повторить прямой A/B победителя и reference.

## Задача 3 — собрать и проверить Vulkan backend

**Артефакты:**

- target build
  `/home/orangepi/vip9000-lab/build/vulkan-38c66-powervr`;
- target raw runs `vulkan-ngl0-*` и `vulkan-ngl99-*`;
- локальный `benchmarks/results/bonsai-night-20260810/vulkan-screen.jsonl`.

- [ ] До сборки сохранить source `git rev-parse HEAD` и CMake cache reference.
- [ ] Создать отдельную конфигурацию:

```bash
cmake -S /home/orangepi/vip9000-lab/src/llama-prismml-shallow-38c66 \
  -B /home/orangepi/vip9000-lab/build/vulkan-38c66-powervr \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_NUMBER=9594 \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod \
  -DGGML_CPU_REPACK=ON -DGGML_VULKAN=ON -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF -DGGML_OPENCL=OFF -DLLAMA_CURL=OFF
```

- [ ] Сборку запускать с ограниченной параллельностью и thermal guard;
  сохранить configure/build log и итоговый CMake cache.
- [ ] Проверить `llama-bench --list-devices`; одинаковые device 0/1 не считать
  двумя GPU до доказательства разных DRM nodes/UUID.
- [ ] Запустить full-model `tg1` с `-ngl 99`; проверить реальные offloaded
  layers, отсутствие unsupported-op fatal, OOM, swap и driver reset.
- [ ] Тем же Vulkan binary выполнить `tg32 r3` с `-ngl 0` и `-ngl 99`.
- [ ] Если `-ngl 99` хуже, проверить один ограниченный partial-offload только
  при наличии данных о фактическом размещении; не перебирать слои вслепую.
- [ ] Записать load time, offloaded layers, median tok/s, peak RSS/swap,
  temperatures и ошибки драйвера.

## Задача 4 — NPU compiler/runtime gate без ложного model result

**Артефакты:**

- дополнить `docs/evidence/bonsai-night-2026-08-10.md`;
- при новом результате создать
  `benchmarks/results/bonsai-night-20260810/npu-gates.jsonl`.

- [ ] Сопоставить версии AcuityLite, generated TIM-VX, TIM-VX headers,
  OpenVX capabilities ABI и target VIPLite/CID.
- [ ] Не передавать на плату NBG, пока минимальный host C0 Add не завершится
  без ABI shim/crash и не выдаст валидный compiled artifact.
- [ ] Если compiler gate неожиданно пройден, сначала выполнить Q1
  `16x128` golden vectors, затем 50 resident operator repetitions.
- [ ] Требовать `expanded_weight_ddr_bytes=0`, exact numerical golden и
  раздельные H2D/run/D2H timings.
- [ ] Не интегрировать operator result в llama.cpp этой ночью без пройденных
  correctness и performance gates.

## Задача 5 — CPU compiler fallback A/B

**Условие:** выполнять только если Vulkan не дал принятого ускорения и осталось
безопасное окно времени/температуры.

**Артефакты:**

- target build `/home/orangepi/vip9000-lab/build/cpu-38c66-fp16`;
- target raw run `cpu-fp16-*`;
- локальная summary row с пометкой compiler A/B.

- [ ] Собрать отдельный binary с
  `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`; reference не заменять.
- [ ] Сначала запустить существующий Q1 operator golden/self-test.
- [ ] Затем выполнить `tg32 r3` с той же CPU topology и poll, что у победителя.
- [ ] При результате `>=2%` провести повторный alternating A/B; иначе
  остановить ветку и записать отрицательный результат.
- [ ] Source prefetch/kernel change начинать только с failing unit/operator
  test и отдельного коммита.

## Задача 6 — подтвердить лучший full-model вариант и golden

**Артефакты:**

- target raw run `winner-tg128-r5`;
- target raw deterministic completion для reference и winner;
- локальный `benchmarks/results/bonsai-night-20260810/summary.json`.

- [ ] Выполнить alternating fresh-reference/winner, `tg128 r5`, с одинаковым
  cooldown и profiling contract.
- [ ] Посчитать median, mean, stddev, CV, bootstrap или sample interval и
  процент изменения относительно свежего reference.
- [ ] Запустить одинаковый prompt с fixed seed, `temp=0`, `top-k=1`,
  `top-p=1`; сохранить tokens/text.
- [ ] Сравнить токены и текст byte-for-byte. Любое расхождение отменяет
  performance claim.
- [ ] Проверить exit codes, temperatures, frequencies, RSS, swap, throttle,
  kernel/driver logs и фактический backend.

## Задача 7 — тестируемый агрегатор и понятный XY-график

**Файлы:**

- создать `tests/test_generate_bonsai_night_chart.py`;
- создать `tooling/generate_bonsai_night_chart.py`;
- создать `benchmarks/charts/bonsai-night-20260810.svg`;
- обновить `docs/evidence/bonsai-night-2026-08-10.md`.

- [ ] RED: fixture с reference, CPU variants, Vulkan/NPU eligible/ineligible
  rows; тест требует fail-closed schema, median/error bars и запрет смешивать
  operator timings с model tok/s.
- [ ] Запустить тест и сохранить ожидаемое падение из-за отсутствия generator.
- [ ] GREEN: реализовать минимальный stdlib JSONL→SVG generator.
- [ ] Повторно запустить unit test; затем весь релевантный test suite.
- [ ] График: X — backend/topology, Y — full decode токенов/с; горизонтальная
  golden reference, error bars, подписи `%`, exact-quality markers и явная
  область кандидатов, не допущенных к model comparison.
- [ ] Проверить SVG визуально и отсутствие наложения подписей.
- [ ] В русской документации объяснить термины `GEMV`, `H2D`, `run`, `D2H`,
  `golden`, `tg32/tg128`, `offload`, `fallback`, `CV`.

## Задача 8 — проверка, Git и публикация

- [ ] Выполнить `git diff --check` и unit tests generator/parser.
- [ ] Проверить JSON/JSONL schemas отдельной командой.
- [ ] Сопоставить все итоговые числа с raw target output; не переписывать
  samples вручную.
- [ ] Закоммитить только файлы этой серии, не затрагивая ранее dirty paths.
- [ ] Обновить/закрыть Beads tasks с точными результатами и blockers.
- [ ] Опубликовать коммиты в `codex/profiling-foundation` и проверить remote
  content; работа не считается опубликованной до успешной проверки.
