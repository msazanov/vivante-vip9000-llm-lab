# E049d — окно ARM PMU для трёх steady decode

Статус: **REJECTED на marker gate; серия 5×3 не запускалась**.

Цель эксперимента — измерить аппаратными ARM PMU events ровно три
steady-state single-token graph computation Bonsai-27B, исключив загрузку
модели, prompt eval и первый single-token decode. Счётчики должны были
сниматься отдельными группами `core`, `cache` и `memory`, по пять повторов.
Этот этап до них не дошёл: обязательный `n=4` marker gate не получил `E`.

## Термины

- **single-token decode** — один проход графа, который обрабатывает один новый
  токен после prompt;
- **steady decode** — повторяющаяся часть генерации после первого такого
  прохода;
- **S / ACK / E** — child пишет `S`, parent атомарно включает группу PMU и
  отвечает `ACK`, после измеряемого окна child пишет `E`;
- **sample_valid** — общий gate результата. Поддержанные counters и ratio 1.0
  недостаточны, если граница `E` отсутствует.

## Preflight веток

До изменения кода выполнен read-only all-branch preflight. Инструмент требует
суффикс ID через дефис, поэтому первая команда с literal `E049d` была
отклонена: `ValueError: invalid experiment_id: 'E049d'`. Повтор с canonical
ID `E049-STEADY-PMU` проверил **16 refs**, включая **10 remote-tracking refs**,
и вернул `duplicate_decision=no_duplicate`, `candidate_count=0`. Полный
снимок: [`data/branch-preflight.json`](data/branch-preflight.json).

## Marker patch

[`e049d-steady-pmu.h`](../../patches/llama.cpp/38c66ad/e049d-steady-pmu.h)
содержит per-context state machine, а
[`e049d-steady-pmu.patch`](../../patches/llama.cpp/38c66ad/e049d-steady-pmu.patch)
ставит вызовы непосредственно до и после `graph_compute` в
`process_ubatch`.

Контракт:

1. без `LLAMA_E049D_STEADY_PMU=1` helper не обращается к fd 8/9 и не создаёт
   измерительное окно;
2. prompt/microbatch с `n_tokens != 1` игнорируется;
3. первый `n_tokens == 1` пропускается;
4. перед следующим проходом пишется один байт `S` в fd 9 и ожидается один
   байт `A` из fd 8;
5. после трёх успешных измеряемых `graph_compute` пишется один байт `E`;
6. нет файлового trace, printf/logging, allocation или иной I/O кроме этих
   двух marker writes и одного ACK read.

Host TDD исполняет настоящий helper с pipe на фиксированных fd. Последовательность
`[4, 1, 1, 1, 1, 1]` даёт measured flags `[0, 0, 1, 1, 1, 0]` и ровно
`S → ACK → E`; без env та же программа проходит без marker-fd.

Baseline snapshot был скопирован hardlink-ами в отдельный каталог. Перед
patch два изменяемых файла были заменены отдельными inode. SHA baseline до и
после совпали byte-for-byte; raw находится в `baseline-*.sha256`.

Изолированный build:

```text
Release; armv8.2-a+dotprod; GGML_CPU_REPACK=ON; GGML_OPENMP=ON;
GGML_NATIVE=OFF; LLAMA_BUILD_TESTS=OFF; LLAMA_BUILD_SERVER=OFF
```

SHA patched `llama-completion`:
`4515143ac8d5e9fdc762f938c2ead493cbe1d2377f1f5cf82ac37eaf56b8fbf8`.
Модель в git не добавлена; её target SHA:
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.

## Обязательные gates

Во всех валидных model runs использованы E047 prompt/settings:

```text
prompt="Explain the A733 memory bottleneck in one short sentence."
ctx=512, batch=512, ubatch=512, n=4, seed=123, temperature=0,
top-k=0, top-p=1, min-p=0, repeat-penalty=1, repeat-last-n=0,
threads=8, threads-batch=8, affinity=0xff, taskset=0-7,
flash-attention=on, mmap=on, ngl=0, KV=f16
```

| Gate | Результат | Tokens | Thermal |
|---|---|---|---|
| patched, env отсутствует, `n=1` | PASS: model load/generation RC=0, fd 8/9 не требуются | generated `[271]` | max 65.162 °C, abort=false |
| unpatched stock `llama-completion`, `n=4` | PASS: RC=0 | generated `[271, 248068, 198, 8160]` | max 65.162 °C, abort=false |
| patched + env + core PMU, `n=4` | **REJECTED**: launcher exit 2, `sample_valid=false`, `S=true`, `ACK=true`, `E=false`, `failure_reason=sync_pipe_closed` | generated точно совпали со stock | max 65.100 °C, tripped=false |

Token-capture raw у stock и marker run byte-identical, SHA-256
`12aa19bdb7d3920cc33ffc097ea685539ca85aff060ecdb2f3b2c1cf0af718b5`.
То есть patch не изменил четыре generated token в этом gate, но измерительное
окно не закрылось.

## Почему gate отклонён

Stock stderr фиксирует только **3 eval runs** для `n=4`. Наиболее узкое
объяснение, совместимое с исполняемым state machine и raw: warmup не был тем
`n_tokens == 1` вызовом `process_ubatch`, который должен был поглотить skip.
Helper пропустил первый реальный decode, открыл `S` перед следующим, после
чего до завершения процесса осталось лишь два подходящих graph computation.
Третий measured graph не произошёл, поэтому `E` не был записан.

Это вывод о несовместимости текущей границы с `n=4`, а не профиль Bonsai.
Исправление marker semantics или переход к `n=5` — отдельная гипотеза и в
этой публикации не выполнялись.

## Счётчики rejected run — только raw failure evidence

Launcher успел прочитать следующие значения:

| Event | Raw count | running ratio |
|---|---:|---:|
| `cpu_cycles` | 35,619,769,852 | 1.0 |
| `instructions` | 45,853,853,531 | 1.0 |
| `stall_backend` | 10,162,132,275 | 1.0 |

Эти числа **INVALID** для анализа: общий `sample_valid=false`, `E=false`.
Они не делятся на три токена, не сравниваются между группами и не дают
bottleneck interpretation. `cache` и `memory` вообще не запускались. Events —
счётчики событий, **не DDR bytes**.

Следовательно, E049d не подтверждает memory-bound или compute/unpack-bound
характер Bonsai и не заявляет оптимизацию либо ускорение.

## Провальная попытка оснастки

До корректного stock reference ошибочно был вызван интерактивный `llama-cli`.
Он печатал повторяющийся prompt `>` до заполнения target filesystem и не
создал `exit.txt`. Попытка классифицирована `INVALID_HARNESS`; её token IDs не
используются. Исходный stdout: 481,722,368 bytes, SHA-256
`093ac04c203b7ad0d49e742db93d7cbf9dafd44752aa8726c722c8f62b4e8607`.
Он потоково сохранён как `raw/failed-llama-cli/stdout.log.zst`; распаковка
повторно дала исходный SHA. После этого только подтверждённая target-копия
stdout была удалена, остальные raw сохранены. Thermal JSONL этой попытки также
неполон: 831 строка валидна, строка 832 оборвана (`Unterminated string`),
поэтому он публикуется как failure evidence, а не telemetry sample.

## Безопасность и состояние платы

- thermal limit каждого засчитанного run: 85 °C, срабатываний нет;
- governors не менялись; после gate `policy0=ondemand`, `policy6=ondemand`;
- OPP, DDR и thermal policy не изменялись, пакеты не устанавливались;
- root использовался parent launcher-ом для PMU; workload сбрасывался до
  `orangepi:orangepi`;
- proprietary SDK/NBG, model payload и пароли в репозиторий не добавлены.

Точные SHA, исходные состояния governors/frequency/temperature и commit
находятся в [`raw/target-provenance.txt`](raw/target-provenance.txt). Все
артефакты перечислены в `data/manifest.tsv`.
