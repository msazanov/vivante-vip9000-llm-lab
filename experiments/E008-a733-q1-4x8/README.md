# E008 — A733 Q1 dotprod: packed layout 4x4 против 4x8

Статус: изолированный operator A/B. Это не заявка на ускорение полной модели,
пока не пройдены real-tensor golden, повторяемость и `tg32` gate.

## Гипотеза

PrismML `llama.cpp` содержит две ARM NEON реализации Q1_0×Q8_0 GEMV:
`ggml_gemv_q1_0_4x4_q8_0` и `ggml_gemv_q1_0_4x8_q8_0`. Обе имеют ветку
`__ARM_FEATURE_DOTPROD`, но selector выбирает packed interleave 4x8 только при
наличии `i8mm`; A733 сообщает dotprod, но не i8mm, и поэтому всегда получает
4x4.

4x8 читает восемь sign bytes более последовательно, собирает Q8 activation
через четыре 8-byte loads и использует два независимых dot accumulators. Это
может уменьшить lane-extract/dependency overhead на Cortex-A55/A76. Может и
ухудшить результат из-за лишних `vcombine` и register pressure — поэтому
выбор не делается по чтению кода, а измеряется.

## Единственное изменение

`a733-q1-dotprod-4x8.patch` меняет только возвращаемый tensor trait внутри
ветки `GGML_TYPE_Q1_0 + NEON + dotprod`: `q1_0_4x4_q8_0` на уже существующий
`q1_0_4x8_q8_0`. Математика Q1, Q8 quantization, thread pool и GGUF не
меняются. Expanded копия весов не создаётся: меняется один resident packed
CPU_REPACK layout, а не logical `Q1_0` и не байты в GGUF.

## Fail-closed matrix

1. Применить patch к отдельному clean worktree pinned commit `38c66ad`; control
   checkout и binary не заменять.
2. Собрать с исходными `armv8.2-a+dotprod`, `GGML_CPU_REPACK=ON`.
3. На реальном `blk.0.ffn_gate.weight`, `M=17408`, `K=5120`, выполнить один
   warmup и 50 repetitions, all-core.
4. Требовать exact byte-for-byte output относительно control Q1_0×Q8_0 и
   одинаковый Q8 activation hash. Любое расхождение отклоняет кандидат.
5. Full-model `tg32 r3` разрешён только при operator median gain `>=2%` и без
   thermal/frequency/swap faults. `tg128` и model golden — только после
   положительного `tg32`.

Operator milliseconds никогда не переводятся в выдуманные токены/с. Итоговая
метрика проекта остаётся full-model decode Bonsai при exact quality.

## Результат на A733

Patch применён к отдельному clean worktree pinned commit `38c66ad`; собрана
отдельная `armv8.2-a+dotprod` библиотека. Все четыре candidate runs выдали
точно тот же output SHA-256, что control:
`359f62fb43af0150cd2043b26c582e97562aaaf2f8a7134ce34ed144dbf1685f`.
Q8 activation SHA также совпал:
`0c72c165dfe0f132870616f777c165421014249d65f8c1183bcc19bbc6fa4f06`.
Следовательно, layout сохраняет математику Q1_0×Q8_0.

Чередующийся A/B, каждый run `warmup=1`, `iterations=50`:

| Пара | Control 4x4 median | Candidate 4x8 median |
|---:|---:|---:|
| 1 | 3.850 ms | 4.238 ms |
| 2 | 3.198 ms | 4.158 ms |
| 3 | 3.712 ms | 7.253 ms |
| 4 | 3.743 ms | 3.987 ms |
| **Median-of-medians** | **3.727 ms** | **4.198 ms** |

Candidate хуже на **12.62%** по median-of-medians. Один 4x8 run имеет большой
выброс, но даже лучший candidate run `3.987 ms` медленнее cohort control
median `3.727 ms`. Значит более последовательный 8-byte interleave не
компенсирует лишние `vcombine`/register dependencies на A733. `tg32` не
запускается: operator gate `>=2%` не пройден.

Первая попытка увеличить run до 500 repetitions корректно завершилась кодом 2:
baseline runner fail-closed принимает только `warmup=1`, `iterations=50`.
Failed directory сохранён, но не включён в таблицу.

Ограничение provenance: переиспользованный E004 runner печатает compile-time
строку `CPU_REPACK_Q1_0_4x4`, хотя linked candidate library содержит
единственное selector-изменение на 4x8. Поэтому machine-readable candidate
rows помечаются unqualified; фактический layout доказывается patch hash/diff,
изолированным build path и exact output, но не полем старого runner. Для
положительного кандидата потребовалась бы отдельная динамическая identity.
