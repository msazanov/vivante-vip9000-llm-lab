# E006 — PowerVR Vulkan: type-specific shared-memory gate для Q1_0

Статус: pinned patch применён и проверен на target; он устранил первый global
gate, но следующий PowerVR Q1 pipeline gate завершился ошибкой драйвера.

## Контекст

Эксперимент относится к сборке PrismML на commit `38c66ad02`:

```text
ci(release): full self-contained Windows Vulkan/HIP bundles + README fork note (#78)
```

На Orange Pi с A733 обнаружен PowerVR B-Series BXM-4-64 MC1 через Vulkan. В выводе
драйвера: Vulkan 1.3.277, driver `24.2@6603887`,
`maxComputeSharedMemorySize = 16384` байт; subgroup/warp size драйвер сообщает как
`1`. В текущем коде размер используется с защитными clamp-значениями 8/16/32.

## Корневая причина

В `ggml/src/ggml-vulkan/ggml-vulkan.cpp` цикл инициализации проверяет shared-memory
бюджет для каждого `ggml_type`. Для IQ1-типа в `ggml_vk_matmul_shmem_support()` к
буферам тайла добавляется LUT размером `2*2048 + 4*2048 = 12288` байт. На устройстве
с лимитом 16 KiB самый маленький тайл для такого типа не помещается, после чего
исходный код бросает одно общее исключение:

```text
Shared memory size too small for matrix multiplication.
```

Это прерывает инициализацию всего Vulkan backend, хотя `Q1_0` LUT не использует и
его малый тайл укладывается в 16 KiB. Ошибка была глобальной, а причина —
type-specific.

## Что делает patch

Файл `powervr-q1-shmem.patch` содержит минимальное изменение pinned PrismML:

1. Для типа, чей малый regular MMQ-тайл не проходит проверку, выставляет
   `mul_mat_s/m/l[i] = false` и пишет явное предупреждение с именем типа и лимитом.
   Глобальный `throw` удалён, поэтому неподдерживаемый IQ1 не блокирует `Q1_0`.
2. В `ggml_backend_vk_device_supports_op()` добавляет fail-safe для обычного
   `MUL_MAT`: если для конкретного типа не остался ни один подходящий тайл,
   backend возвращает `false`. Для `Q8_1` проверяются отдельные `_int`-флаги.
   Это предотвращает запуск over-budget shader. Возврат `false` только разрешает
   планировщику выбрать другой backend; успешный CPU fallback этим экспериментом не
   заявляется и должен быть подтверждён отдельным target-тестом.
3. `MUL_MAT_ID` и уже существующие integer shared-memory проверки не изменяются.

Патч не меняет формат весов, вычисления Q1_0, шейдеры или Vulkan ABI. Он только
устраняет ошибочную глобальную блокировку и явно отказывает отдельным типам.

## Риски

- Это не доказательство работоспособности полного Bonsai на GPU. Нужны сборка,
  загрузка Q1_0 pipeline, короткий decode и golden-сравнение на самой плате.
- Если граф исполнения не имеет другого backend для отключённого типа, операция
  должна остаться на CPU; это может снизить скорость, но лучше, чем crash или
  запуск тайла с превышением shared-memory лимита.
- Из-за reported subgroup size `1` следует отдельно проверить сгенерированные
  shader constants и фактическую работу Q1_0; данный patch не подменяет subgroup
  semantics.
- Предупреждение ожидается для типов с LUT, но точный список должен быть снят из
  target-лога, а не предположен по имени типа.

## Verification gates

Проверено без изменения исходного checkout:

```bash
git -C /home/random/src/llama-prismml rev-parse HEAD
# 38c66ad0241da4f9fcce541cda8edc219086cec5

git -C /home/random/src/llama-prismml apply --check \
  /tmp/vip9000-profiling-foundation/experiments/E006-powervr-vulkan-q1/powervr-q1-shmem.patch
# exit 0

PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/E006-powervr-vulkan-q1/tests/test_powervr_vulkan_shmem_patch.py -v
# Ran 1 test ... OK
```

Тест создаёт временную копию pinned checkout, выполняет `git apply --check`, применяет
patch, проверяет `git diff --check`, наличие всех трёх type-specific флагов и guard
`supports_op`, а также отсутствие исходного unconditional `throw`.

Перед применением на target обязательны следующие gates:

1. Собрать отдельный Vulkan build из commit `38c66ad` с patch; CPU build не заменяет
   эту проверку.
2. Запустить `llama-bench`/`llama-cli` с `-ngl 99` на маленьком Q1_0 smoke test и
   проверить, что инициализация больше не падает на IQ1 type loop, а Q1_0 pipeline
   действительно загружен.
3. Провести короткий decode под `profile_command.py` + thermal guard 85 C; сохранять
   stdout, Vulkan log, telemetry и git/source identity.
4. Сравнить первые golden tokens с CPU reference. До exact match нельзя считать
   Vulkan вариант кандидатом на скорость.
5. Только после smoke/golden — canonical Bonsai decode и сравнение tok/s с CPU
   baseline. DDR/bootloader менять не требуется.

## Команды для следующего этапа

Применять patch только в отдельном рабочем checkout:

```bash
git apply experiments/E006-powervr-vulkan-q1/powervr-q1-shmem.patch
cmake -S . -B build/vulkan-38c66-powervr -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod
cmake --build build/vulkan-38c66-powervr --target llama-bench llama-cli -j2
```

## Target result 2026-08-10

Patch SHA-256:
`06a35ce52906b6ab379343d5c9a0361f4a7dd151b52969211ad63424a15e1ea2`.
Он был применён после успешного `git apply --check` к чистому target checkout с
HEAD `38c66ad0241da4f9fcce541cda8edc219086cec5`.

Инкрементальная сборка `llama-bench` и `llama-completion` под profiler и thermal
guard завершилась с кодом 0 за `143.933 s`:

```text
run_id: build-vulkan-shmem-v1-001
build: /home/orangepi/vip9000-lab/build/vulkan-38c66-powervr
```

Повторный full-model smoke:

```text
run_id: vulkan-shmem-v1-ngl99-tg1-r1-001
model: Bonsai-27B-Q1_0.gguf
arguments: -p 0 -n 1 -r 1 -ngl 99 -dev Vulkan0
```

Первый global exception `Shared memory size too small for matrix multiplication`
исчез: модель дошла до warmup generation. Затем PowerVR pipeline compiler
отклонил уже конкретный Q1 shader:

```text
ggml_vulkan: Compute pipeline creation failed for mul_mat_vec_q1_0_f32_f32
ggml_vulkan: vk::Device::createComputePipeline: ErrorUnknown
terminate called after throwing an instance of 'vk::SystemError'
```

Profiler/child status `134`, elapsed `9.211 s`; kernel log не содержит GPU fault
или OOM. Следовательно, patch доказал и исправил только type-global
shared-memory blocker. Он не доказывает работоспособность Q1_0 shader на PowerVR.
Для этого binary нет `tokens/s` и golden, статус full-model кандидата — failed.

Следующий безопасный шаг — отдельный маленький Q1 pipeline/operator harness,
который варьирует workgroup/subgroup specialization и проходит exact CPU golden.
До него дальнейшее изменение универсального Vulkan shader не переносится в
production. Параллельная практическая ветка использует уже подтверждённый direct
OpenCL C 1.2 runtime `libPVROCL.so.1` и packed Q1 без expanded DDR copy.
