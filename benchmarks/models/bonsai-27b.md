# Bonsai 27B — накопленная таблица экспериментов

Pinned модель: `Bonsai-27B-Q1_0.gguf`, SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
Основная метрика — steady-state decode, токенов/с. `rejected` означает, что
вариант измерен, но не превосходит pinned CPU reference `0.888039 ток/с`.

| Эксперимент | Идея / режим | Скорость, ток/с | Изменение к reference | Качество | Статус |
|---|---|---:|---:|---|---|
| E026 | CPU native, F16 KV, flash on, performance governor | **0.888039** | — | throughput baseline | reference |
| E029 | `-mtune=cortex-a55` | 0.856562 | −3.54% | веса не менялись | rejected |
| E030 | inline pair: повторное использование Q8 для двух Q1-групп | 0.559322 | −37.02% | exact stdout | rejected |
| E031 | pair + `noinline` single-half helper | 0.576260 | −35.11% | exact stdout | rejected |
| E032 | один `noinline` helper на четыре Q8-блока (block4), `strict=0` | 0.958360 | +7.92% к E026; −1.02% к E033 | exact stdout | rejected ниже E033 |
| E033 | native GEMV, all-core `strict=0` scheduler | **0.968236** | **+9.03%** | exact stdout | лучший подтверждённый CPU режим |
| E034 | screen 4–8 workers, затем 8 workers full gate, `strict=0` | 0.965594 | −0.27% к E033 | не меняет веса | rejected ниже E033 |
| E035 | scheduler `poll=50`, повторный full gate | 0.972497 | +0.44% к E033 | exact stdout | принятая настройка; новый рекорд не подтверждён |
| E044 | [`big-only PRFM` screen](../../experiments/E044-cluster-prfm/README.md) ([график](../../experiments/E044-cluster-prfm/generated/e044_screen_throughput.png)) | **1.018629** (только `n=8/r=1` screen) | +14.70% к E026; ranking-only | screen без thermal abort; full golden не получен | **PROMISING screen; UNQUALIFIED full, без production tok/s** |
| E045a | [DSpark preflight](../../experiments/E045-dspark-stock-oom/README.md) ([график OOM](../../experiments/E045-dspark-stock-oom/generated/e045_memory_oom.png)); `-c` пропущен | — | не применимо | качество не запускалось | **CONFIGURATION_FAIL**: kernel OOM на полном контексте, не capacity-доказательство |
| E045e | [DSpark functional smoke](../../experiments/E045-dspark-stock-oom/README.md) ([график](../../experiments/E045-dspark-stock-oom/generated/e045_speculation_smoke.png)) | 0.263 (2 фактических токена; decode-phase timer) | не сопоставимо с E026/E035 | `drafted=8`, `accepted=0`, acceptance **0%**; golden не запускался | **SMOKE_ONLY**, не benchmark |
| E045f | [DSpark target-only phase control](../../experiments/E045-dspark-stock-oom/README.md) ([сравнительный график](../../experiments/E045-dspark-stock-oom/generated/e045_speculation_comparison.png)) | 1.120 eval tok/s (1 токен; phase-only) | не сопоставимо с E045e: другой timer | видимый prefix `<think>`; token IDs и golden отсутствуют | **DIAGNOSTIC_ONLY**, не benchmark |

## Последний подтверждённый production-результат

E035 сейчас является принятой production-настройкой: `poll=50` дал полный gate
0.972497 ток/с и exact stdout. Разница +0.44% к E033 находится в естественном
разбросе, поэтому это не считается статистически устойчивым новым рекордом.
E033 остаётся базовым доказанным scheduler-скачком: ослабление жёсткой
round-robin привязки потоков до `strict=0` дало 0.968236 ток/с при совпавшем
quality gate. E032 математически точен, не расширяет Q1-веса и не троттлился,
но отдельный block4 helper дал 0.958360 ток/с — немного хуже одного scheduler
изменения. E034 показал, что уменьшение worker-пула не помогает, а E035
подтвердил `poll=50` (0.972497) без нового статистически устойчивого рекорда.
Подробные карточки: [`experiments/E032-q1-block4-reuse/README.md`](../../experiments/E032-q1-block4-reuse/README.md),
[`experiments/E033-scheduler-strict0/README.md`](../../experiments/E033-scheduler-strict0/README.md),
[`experiments/E034-worker-count-strict0/README.md`](../../experiments/E034-worker-count-strict0/README.md)
и [`experiments/E035-poll-screen/README.md`](../../experiments/E035-poll-screen/README.md).

Каждая строка должна оставаться append-only: новый запуск получает новый ID,
а rejected/unqualified результаты не удаляются и не превращаются в «оценку».

## Дополнение: E044/E045 (2026-08-13)

E044 дал обнадёживающий короткий ranking: cluster-aware `PRFM` на двух A76
достиг `1.018629 ток/с` в одном `n=8/r=1` screen. Однако все три sustained
full-попытки завершились reset платы до throughput и quality/golden; поэтому у
E044 нет production-скорости, а запись остаётся `PROMISING screen /
UNQUALIFIED full`. Температурный guard `85 °C` не сработал, причина reset не
установлена. Графики и нормализованные данные находятся в
[`E044`](../../experiments/E044-cluster-prfm/README.md).

E045 подтвердил, что официальный DSpark-путь можно загрузить с ограниченным
`-c 512`, но пока дал только диагностику. E045a — ошибка конфигурации полного
контекста: kernel OOM при пропущенном `-c`, без валидной генерации. В E045e
DSpark завершил functional smoke со скоростью `0.263 ток/с`, предложил 8
токенов и принял 0 (`0%`). E045f — отдельный target-only eval-phase контроль
`1.120 ток/с`; его timer не совпадает с decode-phase timer E045e, поэтому
отношение `1.120/0.263` не является speedup. В E045f сохранён только видимый
prefix `<think>`, без token IDs и golden. Полный разбор и графики — в
[`E045`](../../experiments/E045-dspark-stock-oom/README.md).
