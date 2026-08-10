# E009 — A733 Q1_0×Q8_0: prefetch packed weights

Статус: минимальный operator A/B; full-model результат не заявляется до
прохождения exact golden и performance gate.

## Гипотеза

Победивший ARM kernel `ggml_gemv_q1_0_4x4_q8_0` последовательно читает
`block_q1_0x4`. Один interleaved block занимает 72 байта: четыре FP16 scales и
64 байта sign payload. В hot loop нет явного `prfm`; disassembly принятой
сборки также не содержит prefetch для этой последовательности.

Patch делает read-prefetch на четыре blocks, то есть примерно на `4 × 72 =
288` байт вперёд. Q8 activation имеет всего 5,440 байт для K=5120, а таблица
распаковки знаков — 2 KiB; обе повторно используются и не prefetch-ятся.
Меняется только ожидание packed weight stream, не арифметика и не layout.

## Ограничение изменения

`a733-q1-prefetch4.patch` добавляет один guarded
`__builtin_prefetch(..., read, high-locality)` только во внутренний block loop
Q1 4x4. GGUF, Q1/Q8 scales, thread pool и CPU_REPACK layout не меняются.
Expanded weight buffer не создаётся; resident packed bytes остаются теми же.

## Gate

1. Отдельный clean worktree на commit `38c66ad`, исходная
   `armv8.2-a+dotprod` сборка как control.
2. Real `blk.0.ffn_gate.weight`, all-core, warmup 1, 50 iterations;
   чередующийся control/candidate cohort.
3. Exact byte-for-byte Q1_0×Q8_0 output и одинаковый Q8 activation hash.
4. Full-model `tg32 r3` только если candidate operator median-of-medians
   быстрее control на `>=2%`. Иначе ветка закрывается без `tg128`.

Operator latency не переводится в tokens/s. Даже успешный prefetch становится
ускорением Bonsai только после полного decode и model golden.

## Результат на A733

Candidate собран из отдельного clean worktree `38c66ad`. Disassembly доказал,
что compiler не удалил hint и выпустил именно:

```text
prfm pldl1keep, [x0, #288]
```

Четыре control/candidate пары запускались по очереди на реальном
`blk.0.ffn_gate.weight`, all-core, warmup 1, 50 iterations. Каждый candidate
output byte-for-byte совпал с control; output SHA-256
`359f62fb43af0150cd2043b26c582e97562aaaf2f8a7134ce34ed144dbf1685f`.

| Пара | Control median | Prefetch +288 B median |
|---:|---:|---:|
| 1 | 3.417 ms | 3.862 ms |
| 2 | 3.444 ms | 7.248 ms |
| 3 | 6.723 ms | 4.160 ms |
| 4 | 3.147 ms | 4.004 ms |
| **Median-of-medians** | **3.430 ms** | **4.082 ms** |

Явный L1 prefetch увеличил latency на **19.00%**, то есть снизил operator
speed примерно на **15.96%**. Отдельные runs шумны, но три из четырёх pairs
проиграны, а median-of-medians далеко от gate `>=2%` в нужную сторону.

Вероятное объяснение: последовательный weight stream уже распознаётся hardware
prefetcher, а `pldl1keep` создаёт лишнее давление на L1 и вытесняет Q8
activation либо 2 KiB sign lookup table. Candidate отклонён; `tg32` не
запускается.
