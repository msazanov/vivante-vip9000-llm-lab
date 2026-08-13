# E048 — per-op/per-layer trace для steady-state decode

Статус: `PARTIAL — target smoke PASS, compatibility A/B ещё не выполнен`.

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

Последняя локальная версия patch содержит дополнительную правку: перед
`token_begin` ring resize выполняется с фактическим числом llama threads.
Она ещё не прошла повторную сборку на target. Поэтому текущий документ не
выдаёт её как target-proven результат.

## Smoke-результаты

| Вариант | Результат | События | Overflow | Классификация |
|---|---:|---:|---:|---|
| trace on, capacity 4096 | `RC=0` | 32 772 JSONL | 6208 | провальный/неполный trace, сохранён как важное evidence |
| trace on, capacity 65536 | `RC=0` | 45 658 JSONL | 0 | полный smoke trace |

`capacity=4096` нельзя использовать для layer/op вывода: переполнение
зафиксировано и не скрывается. Для полного графа smoke нужен
`GGML_E048_TRACE_CAPACITY=65536` или дальнейшая калибровка ёмкости.

Файлы raw и SHA находятся в `raw/` и `data/partial-summary.json`.

## Acceptance gate

Незавершённый gate: ровно пять пар `trace-off/trace-on` на одном E047
compatibility workload (`n_predict=4`, seed 123, общий prompt, одинаковые
threads/affinity/thermal guard). Для каждой пары сохраняются stdout, stderr,
telemetry, E048 JSONL и exit metadata. Сначала проверяются `overflow_count=0`,
JSONL schema и одинаковый generated token stream. Затем считается paired
median/p95 overhead:

- `PASS` — median и p95 overhead ≤ 1%;
- `TRACE_ONLY` — trace корректен, но overhead > 1%;
- `INVALID` — различается token stream или нарушен raw/schema contract.

До прохождения compatibility gate профиль `n_predict=32` не запускается.

## Известные ограничения

- `logical_*` — declared tensor movement, а не физические DDR транзакции;
- `unique_*` — union адресных диапазонов доступных tensor buffers;
- Q1 ledger считает packed Q1 4x4/4x8 и Q8 activation отдельно, но не
  утверждает фактический DDR traffic;
- trace overhead не является скоростью модели и не используется как
  optimization result;
- fused node записывается как один timed event с `fused_nodes > 1`, а
  `GGML_CPU_DISABLE_FUSION` остаётся отдельным диагностическим контролем.
