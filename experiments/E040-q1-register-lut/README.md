# E040 — регистровая nibble-LUT для packed Q1 на A733

Дата: 2026-08-12
Плата: Orange Pi Zero 3W / Allwinner A733 (sun60iw2)
Ядро: 6.6.98-sun60iw2, aarch64
Модельный контекст: Bonsai-27B-Q1_0, pinned SHA-256 17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0

Статус: **golden PASS; performance-gate REJECT; production REJECT**.

## Цель

Проверить, можно ли заменить табличное разворачивание packed Q1-знаков на
регистровую nibble-LUT и выиграть время без создания expanded +1/-1-тензора
в DDR.

Ядро e040_q1_group_wholek обрабатывает одну native block_q1_0x4 output group
по всему K. Оно читает четыре native block_q8_0 на каждый Q1-блок,
разворачивает четыре nibble через 64-байтную LUT только в NEON-регистрах и
использует AArch64 DOTPROD sdot.

Важно: baseline timing — **один native SIMD group**, то есть один вызов
native_simd_group с теми же входами и тем же whole-K циклом. Это не сравнение
с двумя native вызовами.

## Exact-gate

На target-run-002 прошли:

    PASS early-exit nb=0/high-bit with null inputs
    PASS red-zone/canary dst/q1/q8/lut nb=1
    PASS exhaustive packed-Q1 byte values=256 offsets=64
    PASS E040 register-nibble Q1 whole-K: exact/oracle/native/no-mutation

Проверены K=128, 256 и 5120, шесть шаблонов данных, exhaustive-перебор 256
значений байта Q1 на 64 смещениях, ранний выход с nullptr, canary/red-zone и
отсутствие мутации входов. Вывод E040 совпал побитно с независимым scalar
oracle и native NEON/DOTPROD baseline.

## Target microgate

Квалифицированный запуск использовал четыре чередующихся раунда по 25 ms
для каждого пути:

| Раунд | Порядок | E040 LUT, ns/call | Один native SIMD group, ns/call |
|---:|---|---:|---:|
| 0 | E040-first | 4509.0 | 1742.1 |
| 1 | native-first | 4508.5 | 1741.2 |
| 2 | E040-first | 4505.7 | 1740.9 |
| 3 | native-first | 4505.5 | 1740.4 |
| median | — | **4507.1** | **1741.1** |

    candidate latency / baseline latency = 2.58865085x
    candidate speed / baseline speed      = 0.38630161x
    speed regression                      = -61.36984%
    latency regression                    = +158.86509%

Регистровая LUT битово корректна, но дополнительные TBL, индексация и
регистровые зависимости существенно дороже штатного native пути. E040 не
подключать к repack.cpp или llama.cpp; end-to-end Bonsai прогон для этого
кандидата не оправдан после провала microgate.

## Методика и безопасность

Qualified-команда на плате:

    profile_command.py --interval-ms 50
      -> python3 thermal_exec_guard.py --limit-mc 85000 --interval-ms 100
      -> taskset -c 0-7 e040_green

CPU policy временно переключались ondemand -> performance и восстановлены:

    policy0/scaling_governor=ondemand
    policy6/scaling_governor=ondemand

DDR raw/PLL, NPU, GPU, MMIO, SMC, модули и вентилятор не изменялись.
Thermal guard завершился штатно, abort не было. В qualified trace было четыре
sample; максимумы:

| Зона | Максимум |
|---|---:|
| CPU little / cpul_thermal_zone | 38.192 °C |
| CPU big / cpub_thermal_zone | 38.192 °C |
| DDR | 35.898 °C |
| NPU | 35.712 °C |
| GPU | 36.084 °C |
| skin | 31.638 °C |

## Артефакты

В репозитории находятся только проверенные исходники:

- e040_q1_group_wholek.S — AArch64 assembly;
- e040_q1_group_harness.cpp — scalar/native golden, canary, exhaustive и
  timing harness;
- results/e040_target_summary.json — компактный sanitized результат.

SHA исходников:

    e040_q1_group_wholek.S     00bea3ec1573698d8b0156417df675c2013d6ebc4f03e55cef30fc4096ade3fd
    e040_q1_group_harness.cpp  2001de476b74019703858f9e2c58869ca9d7d3d25a777af79ced5ce199c5526c

Target-only ELF, objects, vendor SDK, NBG и модель намеренно не коммитятся.
Target staging SHA allowlist зафиксирован в JSON.

### Result-001: незапущенный prelaunch

Путь на плате: results/e040-register-lut-target-20260812-001/
Статус: **unqualified prelaunch failure**, run_rc=127.

Профилировщик получил PermissionError, потому что первая команда вызвала
thermal_exec_guard.py без python3. E040 binary не стартовал:
telemetry_samples=0, stdout пустой. Каталог сохранён отдельно и не смешивается
с квалифицированным запуском.

### Result-002: qualified target run

Путь на плате: results/e040-register-lut-target-20260812-002/
Read-only evidence: /tmp/e040-target-results/e040-register-lut-target-20260812-002/
Статус: **golden PASS, performance REJECT**, run_rc=0.

Ключевые hashes evidence:

    metadata.json         f5377f7afeb9d7306a7a4ff42c2c78e552b85e6d9106e72a02e75c243cccaca4
    stdout.log            88e7f19f0b664ca610637e0532af9438432fa3b385ef615ff2bf6849de54e699
    stderr.log            e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    thermal_guard.jsonl   cc9e0aad45c078923ba997b4f4f8656c69f924c71eb3fca997bdb63b0f91525e
    telemetry.jsonl       4947f57441d7c1fc2674c4853ec0a5b9ebfb407c676dcffc1f792424b7d7731f

Полный русский отчёт target-run: /tmp/e040_target_report.md,
SHA-256 bc9ff7311d6ac524c0d41da54ab0fa7f985fb3757421625c25cab72018a43c24.

## Следующий тест

E040 показывает, что изменение только схемы unpack внутри одного output group
не устраняет bottleneck. E039 уже показал, что простое объединение соседних
Q1-групп также не даёт выигрыша.

Следующий кандидат должен проверять более широкий full-GEMV путь: генерацию
адресов и расписание загрузок, реальное устранение memory stalls, параллелизм
по output rows/threads и удержание packed данных в cache/регистрах. Решение о
переносе в llama.cpp принимать только после standalone exact-gate и target
timing-gate.
