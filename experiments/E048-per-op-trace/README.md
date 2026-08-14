# E048 — per-op/per-layer trace для steady-state decode

Статус: `TRACE_ONLY — target smoke, 5 paired A/B и raw publication завершены`.

Цель E048 — получить минимально инвазивный trace одного CPU decode-графа
llama.cpp 38c66ad (build number 9594) на A733: границы токена, каждый graph
node и worker, Q1 4x4/4x8 kernel, фазу F32→Q8, CPU/core, barrier-aware время и
раздельный ledger logical/unique bytes. Прямые DDR bytes намеренно остаются
`null`: `/dev/nsi` на плате root-only, а доступного прямого DDR PMU counter нет.

## Что зафиксировано

Исходный baseline `/home/orangepi/vip9000-lab/src/llama-prismml-38c66` не
изменялся. Patch применялся к отдельной копии
`src/e048-per-op-trace-38c66` и собирался в отдельный
`build/e048-per-op-trace-a733-q1-4x8` с параметрами:

`Release`, `armv8.2-a+dotprod`, `GGML_CPU_REPACK=ON`, `GGML_OPENMP=ON`,
`GGML_NATIVE=OFF`, build number `9594`.

Последняя версия patch содержит дополнительную правку: перед `token_begin`
ring resize выполняется с фактическим числом llama threads. Она применена к
изолированной target-копии и подтверждена повторной сборкой
`llama-completion` (только `llama-context.cpp.o` и relink). Thermal-guarded
smoke завершился `RC=0`, `overflow_count=0`; прежнее сообщение
`refusing worker-ring resize` в raw stderr отсутствует. Baseline source/build
не изменялись.

## Smoke-результаты

| Вариант | Результат | События | Overflow | Классификация |
|---|---:|---:|---:|---|
| trace on, capacity 4096 | `RC=0` | 32 772 JSONL | 6208 | провальный/неполный trace, сохранён как важное evidence |
| trace on, capacity 65536 | `RC=0` | 45 658 JSONL | 0 | полный smoke trace |

`capacity=4096` нельзя использовать для layer/op вывода: переполнение
зафиксировано и не скрывается. Для полного графа smoke нужен
`GGML_E048_TRACE_CAPACITY=65536` или дальнейшая калибровка ёмкости.

Файлы raw и SHA находятся в `raw/` и `data/partial-summary.json`.

## Compatibility A/B (ровно 5 пар)

После исправления resize выполнены ровно пять пар на Orange Pi: один и тот же
изолированный бинарник, модель, prompt, seed, affinity, число потоков и
thermal guard; менялась только переменная trace (`off` против `on`).
`n_predict=4`, общий prompt:
`Explain the A733 memory bottleneck in one short sentence.` Все десять
запусков завершились `RC=0`. Полная команда и идентичность workload записаны
в raw metadata на плате и опубликованы в packed/provenance manifests.
Model/binary/token-capture SHA опубликованы в
`data/compat-ab-workload-identity.json` со статусом
`post_hoc_current_only`: до запуска immutable before/after hashes не
снимались, поэтому эти значения не доказывают историческую identity каждого
запуска.

| Проверка | Результат |
|---|---:|
| Пар off/on | 5 / 5 |
| Полный token stream совпал | да, hash `97dab516...d18d65` |
| Generated stream совпал | да, `[271, 248068, 198, 8160]` |
| Trace schema errors | 0 |
| Overflow в каждом trace-on | 0 |
| Trace events в каждом trace-on | 181 194 (5 token boundaries) |
| Thermal guard | 10/10 без срабатывания; max CPU zone ~68.735 °C |

Для overhead используем `profile_elapsed_ms`: wall-clock от
`profile_command` для полного одинакового workload (prompt + decode), а не
разницу внутренних llama.cpp фаз. Paired summary:

| Wall timer | off median | on median | median overhead | off p95 | on p95 | p95 overhead |
|---|---:|---:|---:|---:|---:|---:|
| полный `profile_elapsed_ms` | 35485.12 ms | 35622.87 ms | **+0.39%** | 36865.29 ms | 37299.92 ms | **+1.18%** |

Для таблицы используется линейная интерполяция по отсортированным пяти
значениям: `index=(n−1)×0.95`, то есть между соседними order statistics.
Контрольный nearest-rank (`rank=ceil(n×0.95)=5`) даёт raw p95 `37131.006990 ms`
для off и `37576.843859 ms` для on; при этом методе overhead также
положительный (`+1.20%`).

Итоговая классификация — `TRACE_ONLY`: p95 wall overhead `+1.18%` выше
порога `1%`. Внутренние llama.cpp числа (`eval`: off median 3599.47 ms,
on median 2933.79 ms) дают отрицательное delta, но это не может означать
отрицательную стоимость профайлера: это run-to-run шум/разная фазовая
вариативность. Поэтому `eval` не используется как overhead basis. `prompt
eval` отдельно даёт median `+1.41%` и p95 `+10.61%`, что дополнительно
подтверждает необходимость не называть A/B speedup.

Этот A/B — только compatibility/стоимость instrumentation. Он не является
измерением ускорения Bonsai, не доказывает улучшение tokens/s и не является
финальным verdict E047.

### Что trace реально насчитал

В каждом trace-on сохранено 83 373 405 bytes JSONL. Для steady-state portion
(`token_seq >= 3` в текущем capture) aggregate одного запуска содержит 41 952
Q1 kernel events: примерно 10.809 GB packed-Q1 logical reads, 20.417 GB
logical Q8-activation reads и 49.828 MB output writes. Experiment-level
unique Q8 bytes — `null`/`unknown_unproven`: текущий event не содержит
allocation ID, base/offset или range, поэтому дедупликация не доказуема.
Нижняя и верхняя границы также оставлены `null`, поскольку их нельзя строго
вывести из capture. Это declared/ledger traffic, не измеренный DDR traffic: поля
`observed_ddr_read_bytes` и `observed_ddr_write_bytes` намеренно `null`,
`counter_method=none`. Полная агрегация, включая per-op node bytes и все пять
trace hashes, находится в `data/compat-ab-ledger.json`.

Все десять raw stdout/stderr/telemetry/phases/metadata/thermal/token-id файлов и пять
полных trace **забраны в git как zstd-артефакты** в
`raw/compat-ab/compressed/`. Упаковка детерминирована: zstd level 10,
single-thread, без timestamp. Файл
`data/compat-ab-packed-manifest.tsv` содержит для каждого из 95 артефактов
source path, git path, SHA-256/размер до распаковки и SHA-256/размер сжатого
файла. Все compressed files меньше 20 MB, поэтому split не потребовался.

`data/compat-ab-manifest.tsv` оставлен отдельным provenance manifest исходных
файлов на плате; он не заменяет опубликованные git artifacts. Target raw path:
`/home/orangepi/vip9000-lab/results/e048-compat-ab-20260814/`.

### Source provenance и будущая dedup-схема

Target source diff baseline→E048 опубликован в
`data/e048-target-vs-baseline.diff`; его SHA/размеры и SHA изменённых target
файлов находятся в `data/e048-source-provenance.json`. Target source не
объявляется byte-identical с checked-in patch: в `src/llama-context.cpp`
есть только comment-only delta — patch содержит четыре строки комментария,
target три строки. Кодовая часть delta совпадает, но это явно зафиксировано,
а не скрыто.

Будущий harness обязан снимать model/binary/preload hashes до первого запуска
и после последнего, сравнивать их и помечать эксперимент invalid при изменении;
это требование также отражено в workload identity artifact.

Для будущего trace v2 добавлено требование
`data/trace-v2-allocation-identity-requirement.json` и regression test. Перед
любой публикацией experiment-level unique bytes он требует для weight,
activation и output стабильные `allocation_id`, `base_address`,
`byte_offset`, `byte_length`. E048 v1 этому требованию не соответствует;
его per-kernel `q8_activation_logical_read_bytes` остаётся только logical
descriptor.

## Acceptance gate

Compatibility A/B выше завершён как диагностический прогон. Для каждой пары
сохранены stdout, stderr, telemetry, phases, E048 JSONL и exit metadata;
compact summary/ledger/raw manifests лежат в `data/`. Сначала проверялись
`overflow_count=0`, JSONL schema и одинаковый generated token stream. Затем
считался paired median/p95 overhead:

- `PASS` — median и p95 overhead ≤ 1%;
- `TRACE_ONLY` — trace корректен, но overhead > 1%;
- `INVALID` — различается token stream или нарушен raw/schema contract.

До прохождения compatibility gate профиль `n_predict=32` не запускается.

## Известные ограничения

- `logical_*` — declared tensor movement, а не физические DDR транзакции;
- `unique_*` в raw v1 — локальный event-side descriptor без опубликованных
  allocation identity; experiment-level Q8 dedup не утверждается и в ledger
  оставлен `null`;
- Q1 ledger считает packed Q1 4x4/4x8 и Q8 activation отдельно, но не
  утверждает фактический DDR traffic;
- trace overhead не является скоростью модели и не используется как
  optimization result;
- fused node записывается как один timed event с `fused_nodes > 1`, а
  `GGML_CPU_DISABLE_FUSION` остаётся отдельным диагностическим контролем.
