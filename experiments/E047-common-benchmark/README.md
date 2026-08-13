# E047-common-benchmark: единый бенчмарк

Статус: `planned`. Модель на плате ещё не запускалась.

Цель — воспроизводимо сравнить Bonsai и LFM2.5 на Orange Pi Zero 3W с
фиксацией скорости, качества, памяти и полного трассирования.

Контракт: [e047-common-benchmark.example.json](../../benchmarks/contracts/e047-common-benchmark.example.json),
его проверяемая схема: [e047-common-benchmark.schema.json](../../benchmarks/schema/e047-common-benchmark.schema.json).
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

Перед стартом необходимо выполнить preflight всех локальных веток и remote-
tracking refs. Исходный read-only манифест находится в
`data/branch-preflight.json`, а его выводы описаны в
`hypothesis-preflight.md`.

После запуска каталог обязан содержать неизменённые
`raw/stdout.log`, `raw/stderr.log`, `raw/telemetry.jsonl` и `raw/trace.jsonl`.
Каждый такой файл должен быть перечислен с SHA-256 в `data/manifest.json`.
Статусы `passed`, `failed`, `rejected` и `blocked` одинаково валидны: провал
и отклонение — результаты исследования, а не причина потерять трассу.
