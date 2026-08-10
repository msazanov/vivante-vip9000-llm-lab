# План реализации генератора графиков бенчмарков

> **Для агентных исполнителей:** ОБЯЗАТЕЛЬНЫЙ ПОДНАВЫК: используйте
> `superpowers:subagent-driven-development` для последовательного выполнения
> плана. Каждый шаг отслеживается флажком `- [ ]`.

**Цель:** построить из сохранённых CPU/NPU evidence один детерминированный,
наглядный и корректно отображаемый GitHub SVG, не скрывая несопоставимые,
неполные и непроверенные результаты.

**Архитектура:** один stdlib-only CLI читает канонический JSONL ledger и
VIP9000 summary JSON, валидирует их fail-closed, формирует неизменяемую модель
`ChartData` и рендерит две горизонтальные панели SVG. Генератор не изменяет
evidence; Markdown только встраивает проверенный артефакт и сворачивает широкую
таблицу. Профайлер остаётся независимым и проходит отдельный live-smoke.

**Технологии:** Python 3 standard library (`argparse`, `dataclasses`, `hashlib`,
`json`, `math`, `os`, `pathlib`, `tempfile`, `xml.etree.ElementTree`),
`unittest`, SVG 1.1, Markdown/HTML `<details>`.

## Общие ограничения

- Никаких matplotlib, pandas, Pillow, JavaScript, сетевых запросов или новых
  runtime-зависимостей.
- Пользовательская документация, заголовки, легенды, пояснения и ошибки CLI —
  на русском языке; имена файлов, JSON-полей, CLI-флагов и `data-*` атрибутов
  остаются стабильными английскими идентификаторами.
- CPU throughput и NPU latency всегда находятся в разных панелях и единицах;
  device latency никогда не пересчитывается в LLM tok/s.
- В график попадают только сопоставимые cohort; failed, non-resident,
  metric-less и singleton записи не влияют на шкалу и учитываются в подписи.
- Единственный статус со сплошной обводкой — точное значение `qualified`;
  остальные отображаемые статусы имеют пунктирную обводку и видимый суффикс.
- SVG имеет ширину 1200, минимальную высоту 900 и добавляет 72 пикселя на
  каждую отображаемую запись сверх первых трёх.
- SVG кодируется UTF-8, использует LF, завершается одним `\n`, не содержит
  timestamp, hostname, абсолютный путь или случайный ID.
- `--check` ничего не записывает: 0 — актуально, 1 — отсутствует/устарело,
  2 — входные данные некорректны. `--force` публикует через fsync временного
  файла, `os.replace` и fsync родительского каталога.
- Все изменения поведения выполняются строго RED → GREEN → REFACTOR. До
  production-кода соответствующий тест должен упасть по ожидаемой причине.
- Исходные 78 тестов, тесты профайлера и `record_model_result.py` должны
  остаться зелёными.

---

### Task 1 — Строгий сбор и классификация CPU/NPU evidence

**Файлы:**

- Создать: `tests/test_generate_benchmark_chart.py`
- Создать: `tooling/generate_benchmark_chart.py`

**Интерфейсы:**

- Вход: JSONL `Path` и каталог summary `Path`.
- Выход: `collect_chart_data(ledger: Path, results_dir: Path) -> ChartData`.
- Типы:

```python
class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Series:
    panel: str
    cohort_id: str
    run_id: str
    run_label: str
    metric: str
    status: str
    repository_commit: str
    temperature_c: float
    p10: float
    median: float
    p90: float


@dataclass(frozen=True)
class ChartData:
    cpu: Sequence[Series]
    npu: Sequence[Series]
    omissions: Sequence[tuple[str, int]]
    out_of_scope: int
```

- Порядок `omissions`: `failed_status`, `missing_metrics`,
  `non_resident`, `singleton_cpu_cohort`.

- [ ] **Шаг 1: написать RED-тест импорта и реального cohort**

Тест загружает модуль через `importlib.util`, вызывает реальный ledger и
summary-каталог, затем проверяет литеральные результаты:

```python
data = chart.collect_chart_data(
    ROOT / "benchmarks/results/model-runs.jsonl",
    ROOT / "benchmarks/results",
)
self.assertEqual(
    sorted({row.run_id for row in data.cpu}),
    [
        "bonsai27b-q1-cpu-a55-pp512-tg128-001",
        "bonsai27b-q1-cpu-a76-pp512-tg128-003",
    ],
)
self.assertEqual(len(data.cpu), 4)
self.assertEqual(len(data.npu), 2)
self.assertEqual({row.median for row in data.npu}, {2.803, 2.846})
self.assertEqual(data.out_of_scope, 6)
```

- [ ] **Шаг 2: подтвердить RED**

Выполнить:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart.BenchmarkChartTests.test_real_evidence_selects_comparable_cpu_and_resident_npu -v
```

Ожидается ошибка импорта из-за отсутствующего
`tooling/generate_benchmark_chart.py`, а не ошибка fixture.

- [ ] **Шаг 3: реализовать минимальные типы, чтение и cohort**

Создать модуль с константами допустимых полей и точными функциями
`collect_chart_data(ledger: Path, results_dir: Path) -> ChartData`,
`load_cpu_series(ledger: Path) -> tuple[list[Series], Counter[str]]`,
`load_npu_series(results_dir: Path) -> tuple[list[Series], Counter[str], int]`,
`cpu_signature(row: dict[str, object]) -> Sequence[object]`,
`npu_signature(row: dict[str, object]) -> Sequence[object]`,
`cohort_id(signature: Sequence[object]) -> str` и
`validate_statistics(value: object, location: str) -> tuple[float, float, float]`.

Allow-lists текущих schema задаются буквально:

```python
CPU_CONFIGURATION_FIELDS = {
    "backend", "partition", "context", "batch", "ubatch", "threads",
}
CPU_SOFTWARE_FIELDS = {
    "repository_commit", "runtime", "runtime_commit", "compiler",
    "sdk", "driver", "kernel", "command",
}
CPU_WORKLOAD_FIELDS = {
    "tokenizer", "prompt_suite", "prompt_tokens", "generated_tokens",
    "seed", "deterministic", "warmup_iterations",
}
NPU_ASSET_FIELDS = {
    "vpm_run", "network_binary.nb", "input_0.dat",
    "libNBGlinker.so", "libVIPhal.so",
}
NPU_TARGET_FIELDS = {
    "board", "kernel", "device", "module", "driver_abi",
    "driver_software", "cid", "device_count", "logical_core_count",
}
NPU_RESIDENT_CONFIGURATION_FIELDS = {
    "repository_commit", "profiler_guard_affinity", "host_workload_affinity",
    "device_core", "npu_hz", "loops", "warmup_loops", "measured_loops",
    "profiler_interval_ms", "preload", "npd", "bypass_output", "command",
}
```

`cohort_id` сериализует signature через
`json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`
и возвращает первые 12 символов SHA-256. CPU signature включает model id/hash,
runtime/runtime_commit/compiler/driver/kernel/sdk, полный canonical workload и
context/batch/ubatch; `threads`, `partition`, `command` и provenance
`repository_commit` в signature не входят. NPU signature включает пять полей
`NPU_ASSET_FIELDS`, девять полей `NPU_TARGET_FIELDS`, а также
`device_core`, `loops`, `warmup_loops`, `measured_loops`, `preload`, `npd` и
`bypass_output`. `npu_hz` и обе affinity строки остаются видимыми переменными
label, а provenance, profiler interval и command в signature не входят.

Для NPU сначала валидируются только schema, непустой status, `run_id`, его
уникальность и совпадение с basename каталога. Затем failed запись учитывается
как `failed_status`, а запись без обоих top-level resident statistics blocks и
без `measured_loops` — как `non_resident`. Только resident запись проходит
строгие `NPU_ASSET_FIELDS`, `NPU_TARGET_FIELDS` и
`NPU_RESIDENT_CONFIGURATION_FIELDS`; поэтому legacy `assets/network`,
`core_index`, `loop_count`, `show_top5` и `save_txt` из сохранённых
single/output evidence не считаются ошибкой и не интерпретируются.

- [ ] **Шаг 4: подтвердить GREEN для реальных данных**

Повторить команду шага 2. Ожидается `OK` и один пройденный тест.

- [ ] **Шаг 5: написать RED-тесты fail-closed границ**

Через `tempfile.TemporaryDirectory()` и полноценные synthetic JSON/JSONL
fixtures добавить отдельные тесты, которые ловят конкретные поломки:

```python
with self.assertRaisesRegex(chart.EvidenceError, "duplicate CPU run_id"):
    chart.collect_chart_data(ledger, results_dir)

with self.assertRaisesRegex(chart.EvidenceError, "samples.*repetitions"):
    chart.collect_chart_data(ledger, results_dir)

with self.assertRaisesRegex(chart.EvidenceError, "run_id.*directory"):
    chart.collect_chart_data(ledger, results_dir)

with self.assertRaisesRegex(chart.EvidenceError, "duplicate NPU run_id"):
    chart.collect_chart_data(ledger, results_dir)

with self.assertRaisesRegex(chart.EvidenceError, "measured_loops"):
    chart.collect_chart_data(ledger, results_dir)
```

Также отдельные tests проверяют `NaN`, отрицательное значение, нарушенный
`p10 <= median <= p90`, неверный линейно-интерполированный CPU quantile,
частичный resident NPU statistics block, неизвестное resident execution-поле,
а также успешную классификацию реальных single/output variants как
`non_resident`. Fixture с несуществующим `raw_result="../../sentinel.json"`
должен собраться без попытки открыть этот data-provided path.

- [ ] **Шаг 6: подтвердить RED одного нового правила**

Запустить focused class; первый ещё не реализованный boundary должен упасть с
отсутствующим `EvidenceError`, а реальный cohort-тест остаться зелёным:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart.BenchmarkChartValidationTests -v
```

- [ ] **Шаг 7: реализовать валидацию и omission boundary**

Реализовать literal allow-lists текущей v1 schema, проверку required identity,
типов без принятия `bool` как `int`, `math.isfinite`, sample count и CPU
quantile через позицию `(count - 1) * q` с
`math.isclose(rel_tol=1e-9, abs_tol=1e-12)`.

Failed CPU/NPU обрабатываются до metric validation. Непроваленный CPU с нулём
repetitions и полностью пустыми throughput metrics — `missing_metrics`;
частичный набор fatal. Непроваленный NPU без обоих resident blocks и без
`measured_loops` — `non_resident`; положительный `measured_loops` с
отсутствующим/частичным statistics block fatal.

- [ ] **Шаг 8: подтвердить GREEN и выполнить рефакторинг**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart -v
```

Ожидается полный PASS нового файла. Удалить дублирование только после GREEN и
повторить команду.

- [ ] **Шаг 9: локальный commit задачи**

```bash
git add tooling/generate_benchmark_chart.py tests/test_generate_benchmark_chart.py
git commit -m "feat: validate benchmark evidence for charts"
```

---

### Task 2 — Детерминированный наглядный SVG и безопасный CLI

**Файлы:**

- Изменить: `tests/test_generate_benchmark_chart.py`
- Изменить: `tooling/generate_benchmark_chart.py`
- Создать при проверке: `benchmarks/charts/benchmark-overview.svg`

**Интерфейсы:**

- Потребляет: `ChartData` из задачи 1.
- Производит точные функции `render_svg(data: ChartData) -> bytes`,
  `nice_axis(maximum: float) -> tuple[float, Sequence[float]]`,
  `publish_atomic(output: Path, payload: bytes) -> None` и
  `main(argv: list[str] | None = None) -> int`.

- [ ] **Шаг 0: RED/GREEN для температурной метаинформации и производных величин**

Сначала расширить `Series` полем `temperature_c: float`. До изменения parser
добавить RED-assertions реальных значений `69.089`, `63.612` и `35.650`.
CPU читает `evidence.thermal_peak_millidegrees_c`, resident NPU —
`profiling.npu_peak_millidegrees_c`; оба значения обязательны для отображаемой
строки, не принимают `bool`, конечны, неотрицательны и делятся на 1000.

Отдельным RED-тестом зафиксировать, что нулевая CPU decode median или нулевая
NPU host/device median завершается `EvidenceError`: график обязан безопасно
вычислять `1 / decode_tps` секунд на токен и `1000 / latency_ms` инференсов в
секунду. После минимального GREEN повторить весь parser test file.

- [ ] **Шаг 1: написать RED-тест структуры и русских подписей**

На synthetic `ChartData` вызвать `render_svg`, распарсить результат через
`xml.etree.ElementTree.fromstring` и проверить:

```python
self.assertIn("CPU: пропускная способность LLM — больше лучше", text)
self.assertIn("VIP9000: задержка резидентного запуска — меньше лучше", text)
self.assertIn("токенов/с", text)
self.assertIn("мс", text)
self.assertEqual(len(root.findall(".//*[@data-role='median-bar']")), 4)
self.assertEqual(len(root.findall(".//*[@data-role='whisker']")), 4)
self.assertEqual(len(root.findall(".//*[@data-role='whisker-cap']")), 8)
self.assertEqual(len(root.findall(".//*[@data-role='median-label']")), 4)
```

Fixture содержит label `A55 & A76 <probe>`; XML должен распарситься, а текст
после парсинга должен восстановить исходные символы. Каждая metric group имеет
точные `data-panel`, `data-cohort-id`, `data-run-id`, `data-metric` и
`data-status`.

Тест также проверяет точные подписи производных величин и температуры:

```python
self.assertIn("0.650 ток/с · 1.539 с/ток", text)
self.assertIn("2.846 мс · 351.37 инф/с", text)
self.assertIn("пик CPU 69.1 °C", text)
self.assertIn("пик NPU 35.6 °C", text)
```

- [ ] **Шаг 2: подтвердить RED рендера**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart.BenchmarkChartRenderTests.test_svg_has_visible_structural_contract -v
```

Ожидается `AttributeError` об отсутствующем атрибуте `render_svg`.

- [ ] **Шаг 3: реализовать минимальный renderer**

Реализовать 1200 px horizontal-bar layout. Высота вычисляется как
`max(900, 900 + 72 * (unique_run_count - 3))`. Для каждой metric group вывести
один `rect[data-role=median-bar]`, один горизонтальный
`line[data-role=whisker]`, два вертикальных cap и одну трёхзнаковую подпись.
Нулевая линия и ticks строятся независимо для CPU и NPU.

CPU prompt label показывает только `median` в `ток/с`; CPU decode label —
`median` и `1 / median` в `с/ток`. NPU host/device label — `median` в `мс` и
`1000 / median` в `инф/с`. Основная величина и `с/ток` имеют три знака после
точки, `инф/с` — два, температура — один. У каждого run label есть отдельная
строка `пик CPU … °C` или `пик NPU … °C`; температура не участвует в шкале.

`nice_axis` вычисляет `raw_step = maximum / 5`, степень 10 и первый множитель
из `(1, 2, 5, 10)`, который не меньше raw step; ceiling — ближайшее верхнее
кратное. При `maximum <= 0` выбрасывается `EvidenceError`.

- [ ] **Шаг 4: подтвердить GREEN структуры**

Повторить команду шага 2. Ожидается PASS.

- [ ] **Шаг 5: написать RED-тесты детерминизма, роста и статусов**

Добавить независимые assertions:

```python
self.assertEqual(chart.render_svg(data), chart.render_svg(data))
self.assertEqual(root.attrib["width"], "1200")
self.assertEqual(root.attrib["height"], "972")  # четыре unique run
self.assertIn("stroke-dasharray", unqualified_bar.attrib)
self.assertNotIn("stroke-dasharray", qualified_bar.attrib)
self.assertNotIn(str(ROOT), payload.decode("utf-8"))
self.assertTrue(payload.endswith(b"\n"))
```

Отдельно проверить literal p10/median/p90 geometry, `<title>`/`<desc>`, legend,
omission counts, видимые NPU cohort headers/separators, NPU caveat о
setup/correctness, стабильный порядок cohort/run/metric, полный видимый run ID,
`data-repository-commit` и footer с точными относительными источниками
`benchmarks/results/model-runs.jsonl` и `benchmarks/results/*/summary.json`.

- [ ] **Шаг 6: подтвердить RED динамической высоты или статуса**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart.BenchmarkChartRenderTests -v
```

Ожидается хотя бы один assertion failure на ещё не реализованном поведении.

- [ ] **Шаг 7: завершить renderer без CLI**

Реализовать оставшиеся visual assertions шага 5: русские legend/caveat/footer,
cohort separators, accessibility, status style, provenance metadata,
динамическую высоту и полностью детерминированный порядок/формат bytes. CLI в
этом шаге не добавлять.

- [ ] **Шаг 8: RED/GREEN CLI-контракт**

Для каждой CLI-ветки сначала добавить один subprocess RED-тест, запустить его,
реализовать только эту ветку и подтвердить GREEN. Literal defaults:
`--ledger benchmarks/results/model-runs.jsonl`,
`--results-dir benchmarks/results`,
`--output benchmarks/charts/benchmark-overview.svg`.

Проверяемые ветки: первый default write = 0, повторный = 2;
`--check` current = 0, stale/missing output = 1, malformed input = 2;
`--force` заменяет inode/content и не оставляет sibling temp; совместное
`--force --check` = argparse exit 2. Malformed input не должен создавать
отсутствующий output parent. Сначала запустить каждый новый тест до реализации
нужной ветки и увидеть ожидаемый failure, затем реализовать минимальную ветку и
повторить до PASS.

После RED каждого теста добавить argparse flags `--ledger`, `--results-dir`,
`--output`, mutually-exclusive `--force`/`--check` и только требуемую ветку
`main`. Сбор/рендер bytes всегда завершается до создания output parent. Default
mode использует exclusive create, `--check` сравнивает bytes без записи,
`--force` использует sibling temporary file, flush/fsync, `os.replace`, fsync
каталога и удаление temp при исключении. `EvidenceError` и `OSError` печатаются
как `ошибка: <причина>` без raw payload.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_generate_benchmark_chart.BenchmarkChartCliTests -v
```

- [ ] **Шаг 9: сгенерировать реальный SVG и проверить freshness**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/generate_benchmark_chart.py --force
PYTHONDONTWRITEBYTECODE=1 python3 tooling/generate_benchmark_chart.py --check
```

Ожидается exit 0 у обеих команд. Два последовательных `--force` дают один и
тот же SHA-256.

- [ ] **Шаг 10: локальный commit задачи**

```bash
git add tooling/generate_benchmark_chart.py tests/test_generate_benchmark_chart.py \
  benchmarks/charts/benchmark-overview.svg
git commit -m "feat: render deterministic benchmark overview"
```

---

### Task 3 — Русская документация и безопасное встраивание в model card

**Файлы:**

- Изменить: `docs/superpowers/specs/2026-08-10-benchmark-chart-generator-design.md`
- Изменить: `README.md`
- Изменить: `tooling/README.md`
- Изменить: `benchmarks/models/bonsai-27b.md`
- Изменить: `tests/test_record_model_result.py`

**Интерфейсы:**

- SVG-link из model card: `../charts/benchmark-overview.svg`.
- Marker contract остаётся ровно одним
  `<!-- MODEL_RESULTS_START -->`/`<!-- MODEL_RESULTS_END -->` внутри details.

- [ ] **Шаг 1: написать characterization-регрессию recorder + details**

Добавить card fixture с `<details>`, `<summary>Полная таблица запусков</summary>`
и обоими markers внутри. Записать новый валидный result через реальный CLI и
проверить, что строка добавлена перед END marker, details закрыт, а markers
по-прежнему встречаются ровно по одному.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_record_model_result.RecordModelResultTests.test_valid_result_updates_table_inside_details -v
```

До изменения production card тест обязан пройти: recorder уже ищет markers по
содержимому, независимо от `<details>`. Это characterization существующего
контракта, а не новый production behavior; если тест падает, остановить задачу
и диагностировать несовместимость до изменения документа.

- [ ] **Шаг 2: встроить chart и свернуть таблицу**

В `benchmarks/models/bonsai-27b.md` перед Recorded results добавить:

```markdown
## Наглядный обзор измерений

![Сравнение пропускной способности CPU и резидентной задержки VIP9000](../charts/benchmark-overview.svg)

График строится только из сохранённых evidence. Непроверенный статус означает
наблюдение производительности, а не подтверждение качества модели.

<details>
<summary>Полная таблица запусков с добавлением новых строк</summary>
```

Сразу после открывающего блока перенести существующий раздел
`## Recorded results`, его два абзаца, заголовок таблицы, оба markers и все
шесть строк байт-в-байт. Сразу после `<!-- MODEL_RESULTS_END -->` добавить
отдельную строку `</details>`.

- [ ] **Шаг 3: перевести feature-документацию и добавить команды**

Полностью перевести утверждённый design spec на русский, не меняя технических
правил. В `tooling/README.md` добавить русскоязычный раздел с командами
`--force` и `--check`, exit codes и пояснением источников. В `README.md`
добавить русскоязычный раздел «Текущие графики» со ссылками на SVG и model
card. Не переводить задним числом документы вне текущей feature.

- [ ] **Шаг 4: проверить recorder и ссылки**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_record_model_result -v
python3 -c 'from pathlib import Path; import xml.etree.ElementTree as ET; ET.parse(Path("benchmarks/charts/benchmark-overview.svg")); card=Path("benchmarks/models/bonsai-27b.md").read_text(); assert card.count("<!-- MODEL_RESULTS_START -->")==1; assert card.count("<!-- MODEL_RESULTS_END -->")==1; assert "../charts/benchmark-overview.svg" in card'
```

Ожидается PASS и exit 0.

- [ ] **Шаг 5: локальный commit задачи**

```bash
git add README.md tooling/README.md benchmarks/models/bonsai-27b.md \
  tests/test_record_model_result.py \
  docs/superpowers/specs/2026-08-10-benchmark-chart-generator-design.md
git commit -m "docs: publish Russian benchmark chart workflow"
```

---

### Task 4 — Наглядная QA-проверка, регрессия профайлера и публикация

**Файлы:**

- Проверить: `benchmarks/charts/benchmark-overview.svg`
- Проверить: весь `tooling/` и `tests/`
- Обновить durable notes: Beads `orange-RAG-66i`

**Интерфейсы:**

- Визуальные PNG — временные QA-артефакты под `/tmp`, в Git не входят.
- Remote base перед публикацией — актуальный head `codex/profiling-foundation`;
  ref обновляется только fast-forward (`force: false`).

- [ ] **Шаг 1: полный свежий verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tooling/*.py tests/*.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 tooling/generate_benchmark_chart.py --check
```

Ожидается 0 failures и exit 0 у freshness-check.

- [ ] **Шаг 2: live-smoke профайлера**

Запустить `profile_command.py` на 80 ms stdlib child в новом `/tmp` run-dir.
Проверить `metadata.exit_code == 0`, `profiler_error is None`, последовательные
events `profiler_start`, `profiler_complete`, ненулевой и совпадающий с
metadata счётчик telemetry. Этот smoke доказывает живой profiler path на host;
он не заменяет уже сохранённый board evidence A55/NPU.

```bash
smoke_root="$(mktemp -d /tmp/vip9000-chart-profile-smoke.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 python3 tooling/profile_command.py \
  --output-dir "$smoke_root/profile-smoke-001" \
  --run-id profile-smoke-001 --interval-ms 10 --label live-smoke \
  -- python3 -c 'import time; time.sleep(0.08)'
SMOKE_ROOT="$smoke_root" python3 -c 'import json, os; from pathlib import Path; p=Path(os.environ["SMOKE_ROOT"])/"profile-smoke-001"; m=json.loads((p/"metadata.json").read_text()); phases=[json.loads(x) for x in (p/"phases.jsonl").read_text().splitlines()]; telemetry=(p/"telemetry.jsonl").read_text().splitlines(); assert m["exit_code"]==0 and m["profiler_error"] is None; assert [x["event"] for x in phases]==["profiler_start","profiler_complete"]; assert len(telemetry)==m["telemetry_samples"] and len(telemetry)>0'
```

- [ ] **Шаг 3: отрендерить и визуально проверить два масштаба**

```bash
rsvg-convert --width 1200 benchmarks/charts/benchmark-overview.svg \
  --output /tmp/vip9000-benchmark-overview-1200.png
rsvg-convert --width 600 benchmarks/charts/benchmark-overview.svg \
  --output /tmp/vip9000-benchmark-overview-600.png
```

Открыть оба PNG через image viewer. Проверить без догадок: обе панели видны,
подписи не перекрываются, p10–p90 end caps различимы, единицы и unqualified
предупреждения читаемы, числа 1.583/0.650, 1.445/0.726 и 2.846/2.803 видимы.
Дополнительно видимы производные `1.539/1.377 с/ток`,
`351.37/356.76 инф/с` и пики температуры `69.1/63.6/35.6 °C`.

- [ ] **Шаг 4: независимый GPT-5.6 Luna review**

Reviewer получает plan, task reports и полный diff. Требуются два явных
вердикта: spec compliance и code quality. Любой Critical/Important finding
возвращается первоначальному implementer в fix-loop и проходит scoped
re-review.

- [ ] **Шаг 5: синхронизировать GitHub и проверить remote**

Создать remote blobs/tree/commit только для изменённых файлов поверх свежего
head, обновить `codex/profiling-foundation` с `force: false`, затем повторно
получить commit и SVG/model-card через GitHub connector. Обновить draft PR #7
русскоязычным пунктом о графике и фактическими verification counts.

- [ ] **Шаг 6: закрыть feature-bead после доказанной публикации**

Добавить в `orange-RAG-66i` commit SHA, число тестов, SVG SHA-256, пути PNG QA
и Luna verdict; закрыть issue только после успешной remote-проверки. Задачи
VIP9000 custom-op (`orange-RAG-16p.5`) и GPU baseline (`orange-RAG-16p.6`)
остаются отдельными следующими исследовательскими треками.
