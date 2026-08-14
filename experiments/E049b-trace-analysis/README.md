# E049b — разбор пяти полных E048-трасс

Статус: **TRACE_ONLY — опубликованный post-processing, без запуска платы**.

E049b не меняет llama.cpp и не заявляет ускорение. Он повторно разбирает пять
принятых `trace-on` прогонов E048, сжатых детерминированным zstd, чтобы
получить воспроизводимую картину времени по токену, узлу, слою и типу работы.
Исходная ревизия E048 — `f5bd54eebbfc12c7399a64fd0f64ab56fa9172f5`.

## Что проверено

Скрипт `tooling/e049b_trace_analysis.py`:

1. читает `data/compat-ab-packed-manifest.tsv`;
2. проверяет SHA-256 и размеры всех 95 сжатых и распакованных артефактов;
3. распаковывает пять `pair01..pair05-on-e048-trace.jsonl.zst`;
4. проверяет схему E048, отсутствие переполнения ring и парность
   `token_begin/token_end`;
5. агрегирует worker-события по `(token_seq, node_index)`;
6. сохраняет per-layer/per-op данные в JSON и CSV;
7. строит только два многозначных графика — stacked latency по пяти runs и
   XY `logical bytes → wall latency` для top layer/op.

Команда полного воспроизведения (включая manifest gate):

```bash
MPLCONFIGDIR=/tmp/e049b-mpl \
  python3 tooling/e049b_trace_analysis.py --repo-root .
```

Unit-тесты аналитики:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -p 'test_e049b_trace_analysis.py' -v
```

## Термины и правило времени

* **Worker** — один поток CPU, который выполняет часть узла графа.
  Рабочие интервалы разных worker-ов перекрываются.
* **Wall interval узла** — `min(worker.start_ns)` до
  `max(worker.post_barrier_end_ns)`. В итоговую latency нельзя складывать
  длительности всех worker-ов: это посчитало бы параллельную работу несколько
  раз. Поле `worker_sum_duration_ns` оставлено только как диагностический
  контраст.
* **Barrier/sync** — интервал от `end_ns` до `post_barrier_end_ns`, то есть
  ожидание синхронизации после вычисления.
* **Fused node** — событие с `fused_nodes > 1`. Оно не разворачивается
  искусственно: в JSON остаётся семейство `fused:<OP>`.
* **Logical bytes** — сколько байт описывает текущий Q1/Q8 kernel в своей
  модели доступа. Это не означает, что каждый байт реально ушёл в DDR:
  возможны cache hits и повторное использование.
* **Observed DDR bytes** — физически измеренный счётчиком DDR трафик. В E048
  прямого счётчика не было, поэтому эти поля намеренно остаются `null`.
* **Steady token** — событие с `token_begin.steady_decode=true`; в каждом из
  пяти файлов это `token_seq=3,4,5`. Один sample run — представитель этих
  трёх токенов, выбранный как медианный по wall-time, чтобы stacked-столбец
  раскладывался точно в token wall-time. Итоговые `median/p10/p90/CV` считают
  пять run samples, а не пять отдельных случайных точек.

## Основной результат

Для representative steady token (медиана пяти run samples):

| Категория | Wall-time, мс | Доля token wall-time |
|---|---:|---:|
| Q1 GEMV (объединение перекрывающихся kernel spans) | **676.796** | **75.40%** |
| F32→Q8 | 0.000 | 0% |
| Прочие вычисления | 183.364 | 20.43% |
| Barrier/sync | 36.555 | 4.07% |
| Неатрибутированный зазор | 0.853 | 0.10% |
| **Итого token wall-time** | **897.562** | **100%** |

Внутренняя вариативность wall-time trace-on мала: p10 `891.130` мс,
p90 `901.779` мс, CV `0.543%`. Q1 GEMV — главный наблюдаемый участок
времени. Это не доказывает, что причиной является именно пропускная
способность RAM: в том же интервале смешаны чтение packed-весов, распаковка и
MAC, а прямого DDR-счётчика нет.

В accepted E048 raw capture записано `209 760` steady Q1 events. Фактический
variant трассы — `q1_0_4x4_q8_0`; событий `q1_0_4x8_q8_0` в этих пяти raw-файлах
нет. Анализатор поддерживает обе категории и не переименовывает фактический
4x4 в 4x8.

### Логический ledger на один steady-токен

| Поток | Байты/токен | Decimal | Статус |
|---|---:|---:|---|
| Packed Q1 weight read | `3 603 087 360` | `3.603 GB` (`3.356 GiB`) | logical descriptor |
| Q8 activation logical read | `6 805 831 680` | `6.806 GB` (`6.338 GiB`) | повторное logical чтение |
| Q8 activation unique descriptor | `105 039 872` | `105.040 MB` | не доказан на уровне эксперимента |
| Observed DDR read/write | `null` | — | PMU не был доступен |

Q8 logical read намного больше unique descriptor, потому что один и тот же
вектор активаций логически используется несколькими output-группами. Без
`allocation_id/base/offset/length` нельзя превращать этот descriptor в
экспериментальный unique DDR traffic.

### Отдельная фаза F32→Q8

В пяти steady capture нет ни одного `phase_worker` события
`f32_to_q8` (`f32_to_q8_status=not_observed_in_e048_capture`). Ноль в таблице
означает **не наблюдалось**, а не «фаза доказанно бесплатна» или «её нет в
модели». Следующий аппаратный trace должен обеспечить корректный seam для
этой фазы либо явно подтвердить, что вход уже Q8.

## Сопоставление с trace-off

Для контекста на stacked-графике пунктиром нанесён trace-off `eval/3` из
E048 — это примерно один decode-run по внутреннему summary timer. Его медиана
`1199.823` мс, но sample `pair03` содержит `1751.290` мс. Это другая область
таймера, чем per-token E048 wall interval, поэтому линия является только
диагностическим A/B ориентиром и помечена `TRACE_ONLY`; из неё нельзя вывести
ускорение `tokens/s`.

Графики намеренно сравнивают пять сопоставимых trace-on runs, а не рисуют
одинокую точку:

* [stacked category latency](charts/e049b-category-latency-stack.png)
* [top layer/op XY: logical bytes против wall latency](charts/e049b-top-layer-op-xy.png)

Все 681 групп `layer/op/family`, включая малые операции ниже порога
визуального графика, находятся в `data/e049b-summary.json` и
`data/e049b-layer-op.csv`. График top layer/op ограничен двумя представителями
семейства и latency ≥ `0.18 ms`, чтобы подписи оставались читаемыми; строки не
удаляются из машинных данных.

## Артефакты

* `data/e049b-summary.json` — схема, manifest gate, категории, per-layer/op
  stats, logical bytes и trace-off reference.
* `data/e049b-run-category.csv` — пять run samples для stacked-графика.
* `data/e049b-layer-op.csv` — все layer/op/family группы и их
  `median/p10/p90/CV`.
* `data/e049b-output-manifest.json` — SHA-256 и размеры сгенерированных
  machine/chart артефактов.
* `charts/*.png` — два графика, сгенерированные из этих данных.

## Научный вывод и следующий контроль

E049b локализует **что занимает время**: примерно три четверти steady token
wall-time находятся внутри Q1 GEMV. Он не отвечает на вопрос **почему** —
RAM read или unpack+MAC — потому что E048 записывал logical ledger, но не
физические DDR транзакции и не разделял внутренние подфазы Q1 kernel.

Следующая гипотеза не должна повторять E009/E023 prefetch или E020/E022
NPU-тайлы. Нужен новый парный контроль: одинаковый packed stream из cold DDR,
из прогретого cache и с отдельно измеренной распаковкой/compute, плюс
доступный DDR/NSI PMU. Только разность этих контролей позволит классифицировать
узкое место как bandwidth-limited или unpack/MAC-limited.
