# E055 — cache-hot/cold Q1 microbenchmark

Статус: **STAGE 1 IMPLEMENTED; target ещё не запускался**.

E055 — ограниченный эксперимент для ответа на вопрос: где теряется время
stock Q1_0 4×4 на A733 — в чтении carrier-данных/кэшах или в распаковке
знаков и DOTPROD. Это **не** запуск Bonsai-27B и не новый результат
tokens/s. До прохождения этого gate запрещён полный Bonsai `n_predict=32`.

## Гипотеза и границы

Старые ветки уже проверили на целевой плате 4×8, простые `PRFM`, повторное
использование Q8, whole-K pair, register-LUT, scheduler и разные варианты
repack. E055 не меняет kernel и не предлагает ещё один prefetch. Он меняет
только контролируемые условия наблюдения и разлагает одну и ту же работу на
три сравнимых по layout контрольных режима.

Перед началом выполнен read-only preflight всех локальных и
remote-tracking refs. Полный снимок: [`data/branch-preflight.json`](data/branch-preflight.json).
В нём зафиксированы commit SHA 17 refs и решение по дубликату
`E055-Q1-HOT-COLD`.

## Что именно измеряется

В harness включён исходный E039 fixture и без изменений используется
`native_simd_group`: native `block_q1_0x4` (72 B) + четыре
`block_q8_0` (4×34 B) на один carrier block, то есть 208 B и 128 Q1
значений. Scalar oracle из E039 проверяет bit-exact результат. Веса модели
и GGUF в репозиторий не попадают.

| Режим | Что делает | Что он не доказывает |
|---|---|---|
| `packed_stream` | читает все Q1/Q8 carrier bytes в stock traversal и считает checksum | не измеряет DDR напрямую |
| `unpack_scale` | выполняет LUT sign-unpack в регистрах, читает Q8 и scale, считает checksum; SDOT/FMA нет | не является заменой Q1 GEMV |
| `full_dotprod` | точный stock E039 4×4 NEON/DOTPROD + FP32 accumulation | не является полным decode графом |

Checksum ненулевой и публикуется. Все управляющие функции `noinline,
noclone, used`; это не даёт компилятору удалить измеряемую работу.

## Hot/cold protocol

* `hot_repeat`: fixture и golden прогреваются до PMU marker; затем один
  процесс повторяет тот же carrier budget `250 ms` (или заданное число
  итераций).
* `cold_conditioned`: перед marker harness выделяет отдельный thrash buffer,
  записывает и затем проверяет **каждое 64-byte line**. После успешной
  проверки PMU включает counters и выполняется ровно одна target traversal.
  Поэтому thrash не входит в PMU окно. Поле называется `cold_conditioned`, а
  не `cache_miss_proven`: software write/read доказывает, что все строки
  тронуты, но не может само по себе доказать состояние каждого аппаратного
  cache set.

Граница измерения совместима с E049c: child пишет `S` в fd 9, ждёт `A` в
fd 8, выполняет workload и пишет `E` в fd 9. Подготовка, golden и cold
conditioning выполняются до `S`.

## Матрица target gate

Рабочий набор округляется вверх до полного 208-byte carrier block:

`64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB, 4 MiB, 12.5 MiB`.

Сначала запускаются CPU 0 (A55) и CPU 6 (A76), затем — только если
модель объясняет минимум 4% ускорения Q1, один bounded all-core
production-shape control. Для каждой ячейки нужны минимум пять чередующихся
пар/повторов, группы E049c `core`, `cache`, `memory`, и
`time_running/time_enabled = 1.0`. PMU event values — **счётчики событий,
не байты**. Нельзя умножать refill/access count на размер cache line и
называть результат DDR bandwidth.

В каждом raw sample сохраняются:

* exact command, CPU, compiler/build SHA и source hashes;
* requested/actual bytes, block count, logical Q1/Q8 bytes и dot products;
* elapsed time, calls/s, first-call time, non-zero checksum;
* golden result и cold-conditioning proof;
* grouped E049c PMU JSON без переименования counters в bytes.

Провальные, отклонённые и blocked samples также публикуются и входят в
manifest; они не превращаются в нули и не удаляются.

## Stage 1 reproducibility

Host contract и adversarial static gates:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_e055_q1_hotcold tests.test_e055_harness_contract -v
```

Cross-build без SDK и модели:

```text
aarch64-linux-gnu-g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -march=armv8.2-a+dotprod tooling/e055_q1_hotcold.cpp \
  experiments/E039-q1-pair-wholek/e039_q1_pair_wholek.S \
  -o /tmp/e055_q1_hotcold-aarch64
```

QEMU smoke проверяет ровно 18 scalar-vs-stock golden cases:

```text
qemu-aarch64 -L /usr/aarch64-linux-gnu \
  /tmp/e055_q1_hotcold-aarch64 --self-test
```

Ожидается `golden_pass=true`, `golden_cases=18`. Это не target performance
evidence; QEMU не используется для оценки A733 cache/PMU.

### Сохранённый провал и исправление

Первый QEMU smoke был **REJECTED**, а не удалён: `K=128, pattern=1,
row=1` дал `scalar=0x1.48p+7`, `simd=0x0p+0`. Причина была в harness —
новый `main` не скопировал E039 `make_table()` в глобальную
`g_table_q1_signs`, поэтому register unpack использовал нулевую LUT. Это не
было аппаратным или математическим результатом. Команда, stderr и объяснение
сохранены в [`data/failure-qemu-uninitialized-lut.txt`](data/failure-qemu-uninitialized-lut.txt).
После исправления и повторного cross-build self-test прошёл все 18 cases.

## Анализ после target runs

`tooling/e055_q1_hotcold.py` проверяет sample contract, вычисляет только
сравнимые paired deltas и строит осторожную directional inference:

* cold/hot ratio — cache sensitivity signal;
* full/unpack и full/stream — decomposition ratios, не speedup между
  несопоставимыми режимами;
* `memory_cache_sensitive` допускается только если во всех участвующих
  control cells есть минимум пять samples и cold penalty materially выше;
* иначе verdict — `unpack_compute_sensitive` или
  `insufficient_control_cells`.

Текущая Stage 1 намеренно не содержит target raw и не делает bottleneck
вывод. После target gate добавляются `raw/*.json`, PMU joins, CSV/PNG с
несколькими размерами и режимами и `data/summary.json`.

## Неизменённые ограничения

E055 не утверждает прямой DDR traffic, не изменяет OPP/DDR/thermal policy,
не включает NPU и не запускает модель. Если результат не предсказывает хотя
бы 4% ускорение Q1, следующий оптимизационный код не создаётся: сначала
публикуется отрицательный результат и возвращаемся к полной per-token
трассе E048/E049b.
