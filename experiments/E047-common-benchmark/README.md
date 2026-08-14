# E047-common-benchmark: единый бенчмарк

Статус: `planned`. Модель на плате ещё не запускалась.

Цель — воспроизводимо сравнить Bonsai и LFM2.5 на Orange Pi Zero 3W с
фиксацией скорости, качества, памяти и полного трассирования.

Контракт: [e047-common-benchmark.example.json](../../benchmarks/contracts/e047-common-benchmark.example.json),
его проверяемая схема: [e047-common-benchmark.schema.json](../../benchmarks/schema/e047-common-benchmark.schema.json).
Для результата trace A/B используется отдельная схема
[e047-trace-ab-result.schema.json](../../benchmarks/schema/e047-trace-ab-result.schema.json),
а `results/summary.json` проверяется схемой
[e053-experiment-summary.schema.json](../../benchmarks/schema/e053-experiment-summary.schema.json).
Общий prompt хранится без завершающего перевода строки; SHA-256 закреплён в
контракте. Для совместимости используется `n_predict=4`, для основного
сравнения — `n_predict=32`; оба профиля имеют один прогрев, пять измерений и
чередуются между вариантами.

Декодирование детерминированное: seed 123, temperature 0, context 512,
batch/ubatch 512, threads 8, flash attention on, mmap on, KV-cache F16.
Скорость сравнивается в token/s, visible char/s и UTF-8 byte/s; токены разных
токенизаторов напрямую не считаются единицей качества. Качество оценивается
отдельно по русскоязычной рубрике контракта, а точное совпадение token IDs
допустимо только внутри одной модели и варианта.

Память учитывается четырьмя раздельными классами: алгоритмические
`logical_bytes`, уникальные байты весов `unique_weight_bytes`, прямо измеренные
аппаратным счётчиком `observed_direct_ddr_*` и расчётные `inferred_ddr_*`.
У каждого класса обязательны собственные method/source/confidence; inferred
байты запрещено называть observed.

Каждый выполненный `run_id` обязан присутствовать в `summary`,
`telemetry.jsonl` и `trace.jsonl`. Прямые DDR-счётчики должны совпадать по
байтам, методу и источнику. Если trace разбивает итог запуска по операциям,
части явно помечаются `aggregation_relation.kind=partition_of_run_total`,
ссылаются на run total из `summary`, а валидатор проверяет их точную сумму.

Перед стартом необходимо выполнить preflight всех локальных веток и remote-
tracking refs. Исходный read-only манифест находится в
`data/branch-preflight.json`, а его выводы описаны в
`hypothesis-preflight.md`. Контракт v3 фиксирует точные commit SHA всех
неактивных refs, базовый commit и хеш stage-0 Git index по
`path/mode/blob OID`; удалённый tracked-файл нельзя пропустить только потому,
что его уже нет в рабочем каталоге. Валидатор независимо перечитывает
фактические refs и корни экспериментов: изменение, добавление или удаление ref
делает preflight недействительным. Кандидаты
дубликатов группируются по ID эксперимента, корневому каталогу и ref, а решение
повторно вычисляется из нового сканирования, а не из сохранённых `refs`.

После запуска каталог обязан содержать неизменённые
`raw/stdout.log`, `raw/stderr.log`, `raw/telemetry.jsonl` и `raw/trace.jsonl`.
Каждый такой файл должен быть перечислен с SHA-256 в `data/manifest.json`.
Для raw-файла также фиксируется capture metadata; telemetry и trace должны
быть непустыми валидными JSONL с событиями заявленной схемы.
Выполненный запуск обязан связать `results/summary.json` с результатом trace A/B:
ровно пять пар с tracing off и пять с tracing on. Для каждой пары проверяются
одинаковый workload, token stream, run/pair/mode, существование raw-файла,
capture metadata и SHA-256 из manifest.
Статусы `passed`, `failed`, `rejected` и `blocked` одинаково валидны: провал
и отклонение — результаты исследования, а не причина потерять трассу.
