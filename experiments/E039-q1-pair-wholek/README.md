# E039 — whole-K packed-Q1 pair на A733

Дата: 2026-08-12
Плата: Orange Pi Zero 3W / Allwinner A733, Linux `6.6.98-sun60iw2`, `aarch64`.
Модельный контекст: Bonsai-27B-Q1_0, pinned SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.

Статус: **golden PASS; microgate REJECT для production-интеграции**.

## Идея и терминология

E039 проверяет CPU-ядро для двух соседних packed Q1_0 групп, использующих один Q8_0 activation stream. `whole-K` означает, что один вызов проходит все `K/128` блоков и сохраняет восемь FP32 аккумуляторов до финальной записи; expanded `+1/-1` tensor в DDR не создаётся.

- `Q1_0×Q8_0` — packed sign weights и 8-битные activations с FP16 scales.
- `sdot`/DOTPROD — AArch64 NEON dot-product для четырёх пар INT8.
- `golden` — независимый scalar oracle плюс native NEON baseline; `bit-exact=yes` означает совпадение битов FP32.
- `microgate` — локальное сравнение стоимости ядра, не end-to-end Bonsai.

Гипотеза общего Q8 reuse не прошла по скорости: helper оказался медленнее уже оптимизированного native SIMD.

## Исходники и hashes

В репозиторий включены реальные исходники v2:

- [e039_q1_pair_wholek.S](e039_q1_pair_wholek.S);
- [e039_wholek_harness.cpp](e039_wholek_harness.cpp).

Ожидаемые SHA-256:

```text
3338df3dfedf683247ef0da3975db371667add1c2552f77219718a444687143f  e039_q1_pair_wholek.S
4bbd8841069bef4b6e29f99b1b31dc7e4d1e67a15e0ec644282ef9ca7e5b567e  e039_wholek_harness.cpp
```

Target-only ELF/object намеренно не коммитятся. Их hashes зафиксированы в `results/e039_target_summary.json`.

## Target методика

При stock DDR raw `0x54`:

```text
profile_command.py --interval-ms 50
  -> thermal_exec_guard.py --limit-mc 85000 --interval-ms 100
  -> taskset -c 0-7 e039_wholek_harness
```

Обе CPU policy временно переключались `ondemand -> performance` и через `trap` возвращались в `ondemand`. DDR PLL/raw, MMIO, SMC, NPU, GPU, modules и вентилятор не изменялись. На плате отсутствуют читаемые `/sys/class/devfreq/dmc` и `/sys/kernel/debug/clk`.

Профилировщик сохранил stdout/stderr/metadata/telemetry/phases и thermal trace. `RUN_RC=0`, child/guard status 0, signal отсутствует, thermal abort не было. Статический AArch64 ELF ожидаемо даёт `ldd: not a dynamic executable`.

Полная read-only копия target evidence: `/tmp/e039-target-results/e039-wholek-target-20260812-001/`.

## Golden

Для `K=128,256,5120` и шести паттернов все 18 CASE дали:

```text
scalar_vs_simd_bit_exact=yes
paired_vs_scalar_bit_exact=yes
paired_vs_simd_bit_exact=yes
```

Итог:

```text
PASS E039 whole-K packed-Q1 oracle: K=128,256,5120 patterns=6 no mutation
```

Входные b0/b1/q8 не мутировались.

## Microgate

Сравнение — paired assembly против двух native NEON/DOTPROD calls; median четырёх чередующихся раундов:

| K | paired ASM | 2× native SIMD | paired/native | решение |
|---:|---:|---:|---:|---|
| 128 | 180.1 ns | 170.1 ns | 1.05879× | медленнее 5.88% |
| 256 | 276.3 ns | 250.1 ns | 1.10476× | медленнее 10.48% |
| 5120 | 3945.4 ns | 3448.4 ns | 1.14412× | медленнее 14.41% |

```text
MICROGATE_SUMMARY K=128 paired_median_ns=180.1 two_native_simd_median_ns=170.1 sink=-0x1.661c3cp+41
MICROGATE_SUMMARY K=256 paired_median_ns=276.3 two_native_simd_median_ns=250.1 sink=-0x1.6351d2p+55
MICROGATE_SUMMARY K=5120 paired_median_ns=3945.4 two_native_simd_median_ns=3448.4 sink=0x1.2467e6p+56
```

Вывод: корректность на фактическом A733 подтверждена, но ускорения нет. E039 в llama.cpp не интегрировать. Следующий кандидат должен уменьшать реальные memory stalls полного GEMV, а не только объединять два локальных вызова.

## Thermal и governors

Во время guard samples максимумы: CPU little `39.370 °C`, CPU big `37.448 °C`, DDR `36.518 °C`, NPU `36.518 °C`, GPU `36.766 °C`, skin `31.874 °C`; лимит был `85 °C`.

Governors: before `policy0/policy6=ondemand`, during `performance`, after restore `ondemand`. NPU после теста: `1008000000 Hz`, governor `performance`.

Полный русский отчёт target-run: `/tmp/e039_target_report.md`, SHA-256 `23cd06aa43d355788dfcbe5b0d16a2a81ea3cd6e12cb50931c4f096170615f16`.
