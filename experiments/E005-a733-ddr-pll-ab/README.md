# E005 — защищённый A733 DDR PLL A/B

Эксперимент E005 подготавливает exact PLL DDR candidate `raw 0x54 → 0x63`
(ожидаемый `510 → 600 MHz`). В текущем ходе runner запускается только с mock backend в unit tests;
реальная плата не записывалась, reboot не выполнялся, boot/sysfs/devmem не
изменялись.

## Что именно меняется

`PLL_DDR_CTRL` находится по адресу `0x02002020`. Runner принимает только
полное исходное слово `0xf9005400`, извлекает raw-поле bits 15:8 (`0x54`) и
строит только `0xf9006300`, сохраняя все остальные биты. Разрешена только
эта exact pair; `0x55`, `0x64`, другое исходное слово и любые другие адреса
отклоняются.

`DRAM_CLK_REG=0x02002c00` разрешён только для чтения и drift-контроля. Runner
проверяет enable/LDO/output/lock bits PLL, bounded readback и lock polling.
После любого завершения полный `0xf9005400` восстанавливается в `finally`; при
`SIGINT`/`SIGTERM`, thermal abort, register drift, lock loss, sampler failure,
timeout или workload failure запись также должна быть восстановлена.

## Почему raw не равен автоматически MHz

На проверенном target read-only snapshot raw `0x54` соседствует с
`pll-ddr=2040 MHz` и `dram0=510 MHz`. Поэтому старое обозначение
`raw84→2016` имеет возможный off-by-one: Linux-visible semantics, вероятно,
используют `(raw+1)×24 MHz`, что даёт для кандидата raw `0x63`:
`(0x63+1)×24/4 = 600 MHz` при том же `/4`. Это expected до readback, а не
измеренный результат. SDM/fractional PLL registers и точный source call path ещё
нужно подтвердить. Runner не объявляет raw write доказанным MHz speedup.

Отдельно `dram_para24` — не PLL N: vendor module декодирует его как четыре
5-битных `divider_minus_1`; `dram_para30` bit2 включает devfreq. Активный DT
не содержит `dram_para30`, поэтому обычный kernel DMC path сейчас unbound.

## Watchdog и восстановление

Apply обязан открыть `/dev/watchdog`, получить `WDIOC_GETSUPPORT`, проверить
`WDIOF_MAGICCLOSE=0x0100` и `WDIOF_KEEPALIVEPING=0x8000`, затем задать bounded
timeout через `WDIOC_SETTIMEOUT` и записать фактически возвращённый timeout в
event log. Между семплами (250 ms) отправляется `WDIOC_KEEPALIVE`.
Runner не полагается на watchdog sysfs: на target `CONFIG_WATCHDOG_SYSFS` не
включён, а `/dev/watchdog{,0}` root-only. При успешном завершении используется
magic-close `V`; при hang/потере userspace watchdog должен вернуть boot state.
Если magic-close не заявлен, apply fail-closed.

## Workloads и golden

Без JSON gate runner использует фиксированные argv:

```text
/usr/bin/taskset -c 0-5 /usr/bin/mbw -n 10 256
/usr/bin/taskset -c 0-5 /usr/local/bin/memtester 128M 1
```

Для первого approved raw63 run нужен также Q1 operator gate через повторяемый
`--workload-json`:

```json
{
  "name": "q1-short",
  "argv": ["/usr/local/bin/q1_cpu_operator_runner", "--iters", "1"],
  "timeout_s": 60
}
```

JSON строгий: только `name`, `argv`, `timeout_s`; executable должен быть
совпадать с pinned path (`/usr/bin/taskset`, `/usr/bin/mbw`,
`/usr/local/bin/memtester`, `/usr/local/bin/q1_cpu_operator_runner` или
`/usr/local/bin/llama-bench`); basename spoof и nested shell запрещены.
`shell=True` не используется.
Полный llama-bench допустим отдельным длинным A/B только после успешного
короткого gate. Watchdog и 250 ms telemetry sampler живут на всём workload
lifecycle.

## Команды (не выполнять на target без отдельного approval)

Read-only inspect/dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/a733_ddr_pll_guard.py inspect \
  --event-log /tmp/E005-inspect.jsonl
PYTHONDONTWRITEBYTECODE=1 python3 tooling/a733_ddr_pll_guard.py dry-run \
  --event-log /tmp/E005-dry-run.jsonl
```

Guarded apply-пример (намеренно не запускался в этом ходе):

```bash
python3 tooling/a733_ddr_pll_guard.py apply --apply \
  --expected-original 0xf9005400 --target-raw 0x63 \
  --confirm APPLY-A733-DDR-510-TO-600 \
  --workload-json /path/to/q1-short.json \
  --event-log /path/to/E005-events.jsonl
```

Команда не меняет boot config/DT и не пишет DRAM divider. Перед реальным run
нужны serial/physical recovery, внешний известный-good образ, проверка hashes,
thermal guard 85 °C, zero integrity errors и подтверждение deterministic Q1
golden. `baseline.json` специально отделяет past observations от paired
before/after measurements, которых пока нет.
