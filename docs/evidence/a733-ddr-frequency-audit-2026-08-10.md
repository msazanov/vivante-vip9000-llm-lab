# A733 DDR frequency / bandwidth audit — 2026-08-10

## Краткий вывод

**Проверенный факт.** На целевой системе Debian 12 работает kernel
`6.6.98-sun60iw2`, DT совместим с `xunlong,orangepi-4-pro` и
`arm,sun60iw2p1`, доступно 12,162,208 kB RAM. CPU topology — A55 policy 0
(6 ядер, до 1,794 MHz) и A76 policy 6 (2 ядра, до 2,002 MHz). NPU — отдельный
devfreq `3600000.npu` с 492/852/1008 MHz и текущими 1008 MHz; это не DDR.

**Проверенный факт.** `/boot/orangepiEnv.txt` включает `overlays=... dram-opp`,
а `/boot/boot.cmd` загружает и применяет `sun60i-a733-dram-opp.dtbo`. Overlay
задаёт `dram_para00 = 1200`, `dram_para24 = 0x0f0f0f0f` и два kernel OPP:
510,000,000 и 600,000,000 Hz. Он подменяет `operating-points-v2` для
`/dmcfreq@3120000`.

**Проверенный факт.** При этом сейчас нет DDR-ноды в `/sys/class/devfreq`.
Модуль `sun55iw3_devfreq` версии 2.0.0 загружен, но
`/sys/devices/platform/a020000.dmcfreq/driver` отсутствует и
`/sys/bus/platform/drivers/sunxi-dmcfreq/` не содержит привязанного устройства.
`clk_ddr` привязан к `sunxi-ddrclock`; платформа `dram` также не имеет
привязанного kernel-драйвера. Поэтому kernel userspace API не показывает и не
позволяет безопасно выбирать текущий DDR OPP.

**Проверенный факт / неизвестно.** Read-only `clk_summary` показывает текущие
`pll-ddr=2040 MHz`, `sdram=2040 MHz`, `dram0=510 MHz`; это доказанный
clock-controller readback. Но `dram_para00=1200` остаётся vendor DRAM-init
parameter, а 510 MHz не автоматически означает 1020 MT/s PHY I/O. Boot log с
полным training sequence и независимый PHY/data-rate readback не сохранены.
Нельзя писать «сейчас DDR = 1200 MHz» только по `dram_para00`.

**Гипотеза.** Начальная DDR частота и training задаются BootROM/U-Boot/firmware
до Linux; kernel DMC/devfreq предназначен для дальнейшего DFS, но на текущем
образе не bind-ится. Уверенность средняя: это следует из boot flow, DT,
непривязанного DMC и наличия DRAM-init строк в локальном образе, но не из
прямого U-Boot console log.

**Рекомендация.** DDR overclock пока не выполнять. Сначала получить baseline
и зафиксировать driver ownership; фактический controller rate уже доказан как
510 MHz. Затем сравнивать только vendor-supported OPP (510↔600) после проверки
parent/rate tuple и bind; лишь после отдельного approval обсуждать OPP выше
600 MHz. Простая запись governor не сработает при отсутствующем devfreq bind и
не является overclock-планом.

## Граница аудита и источники

**Проверенный факт.** Выполнены только чтения: `rg` по двум рабочим деревьям и
backup, `codegraph explore`, локальное чтение DT/boot/BSP-артефактов, а также
SSH-команды чтения `/proc`, `/sys`, `/boot`, `dmesg`, DTB/DTBO и метаданных.
Не выполнялись `sudo`, записи в sysfs/devmem, смена governor/частоты/напряжения,
перезагрузка, стресс-тест или inference workload. Shell history, credentials,
полные environment/config secrets не читались.

Основные первичные артефакты:

- target `/boot/orangepiEnv.txt`, `/boot/boot.cmd`;
- target `/boot/dtb-6.6.98-sun60iw2/allwinner/sun60i-a733-orangepi-zero3w.dtb`;
- target `/boot/dtb-6.6.98-sun60iw2/allwinner/overlay/sun60i-a733-dram-opp.dtbo`,
  SHA-256 `4a702d0bff118b64c40c3f095290d78a6d9aa551c1e35cf4955da88e0535a58d`;
- локальный `/home/random/a733_datasheet/A733_Datasheet_V0.93.pdf`;
- локальный vendor-header snapshot
  `/home/random/orangepi-kernel/headers/usr/src/linux-headers-6.6.98-sun60iw2/`;
- foundation evidence: `docs/hardware/orange-pi-zero-3w.md`,
  `docs/evidence/bonsai-q1-cpu-baselines-2026-08-09.md`,
  `docs/evidence/a733-cpu-optimization-matrix-2026-08-09.md`,
  `docs/evidence/vip9000-phase-profile-2026-08-10.md`.

## Что реально пытались менять раньше

### DDR-specific поиск

**Проверенный факт.** В tracked history `/home/random/orange-RAG` и
`/tmp/vip9000-profiling-foundation` нет коммитов/скриптов/логов, доказывающих
прошлую попытку DDR overclock, запись `devfreq`, изменение `dram-opp`,
`devmem`/CCU-регистров или добавление DDR OPP. `git log -S` по
`ddr/devfreq/overclock/LPDDR` дал только общие profile/history hits, без
эксперимента.

**Проверенный факт.** В `/home/random/orange-RAG/orangepi_profile.yaml` и
`orangepi_deployment_checklist.md` LPDDR5/12 GB и желаемый CPU governor
`performance` записаны как профиль/план. Это не журнал применения: на свежем
снимке CPU governors фактически `ondemand`.

**Прошлое наблюдение.** Самый близкий к DDR артефакт — уже установленный
`dram-opp` boot overlay. Он меняет DT OPP table и `dram_para00`, но в доступных
материалах нет даты/автора/результата A/B и нет доказательства, что kernel
выбрал какой-либо из двух OPP.

### Углублённая проверка старых артефактов (follow-up 2026-08-10)

**Сообщение владельца проекта (provenance, не проверенный факт).** Владелец
сообщает, что в старом Orange RAG уже меняли частоту памяти через бит или
множитель и остановились из-за сильного thermal throttling; теперь добавлено
серьёзное охлаждение и ветку можно продолжить. В сообщении нет даты, хеша
артефакта, точного регистра/бита, before/after readback или thermal log. Поэтому
это не заменяет первичный журнал эксперимента и не используется как доказанный
факт текущего состояния платы.

**Прошлое наблюдение (не independently verified).** Найден внешний
`/home/random/orange-pi-dram-overclock-kb.md` (mtime 2026-07-02, mode 600), на
который ссылается `.omo/plans/orange-rag.md:2178`. KB описывает именно старую
попытку, а не только план:

- clock tree заявлен как `HOSC 24 MHz → PLL_DDR → DDR CCU div → DMC`;
- для PLL DDR указаны CCU base `0x02002000`, CTRL register `0x02002020`,
  `N` в bits 15:8, `M` в bits 22:20, enable bit 31, lock bit 28 и output
  bit 27; для DRAM clock divider — `0x02002c00`, div bits 4:0;
- заявленный stock readback: CTRL `0xf9005400` (N=84), после OC:
  `0xf9005500` (N=85), при неизменных M=0, div=3; это соответствует
  заявленным PLL 2016→2040 MHz и controller DDR clock 504→510 MHz;
- метод назван как прямой Python `mmap` `/dev/mem`, изменение `N=84 + 0x100`
  до N=85; заявлены lock=1, ping watchdog >7 s, отдельный
  `memtester 128M 1=pass` и 11 минут стабильной работы;
- одновременно в error log KB есть отдельный случай зависания board при N=85:
  автор связывает его с timeout `sudo` pipe и возможно невооружённым watchdog,
  а не с доказанной DRAM data corruption; после этого рекомендован запуск
  скрипта локально на плате. Это объясняет, почему N=85 в KB одновременно
  помечен как «стабилен 11 минут» и имеет operational failure note.
- дальнейшие N=86/N=88 в самом KB отмечены как planned, а таблица N=90/95/100/105
  — как потенциал, не как проведённые испытания. Сам KB не содержит исполняемого
  скрипта, полного stdout/stderr, серийной консоли или хеша регистра readback.

Это впервые даёт конкретный исторический bitfield/multiplier candidate:
`N` (PLL multiplier/factor) в bits 15:8, шаг `0x100` в `0x02002020`; это не
доказывает, что изменение было сделано на текущем boot image или что оно было
причиной последующего throttling. В KB нет измерения thermal zone во время
N=85; сообщение владельца о thermal stop остаётся отдельной provenance.

**Проверенный факт / коррекция старого KB.** Текущий raw readback после reboot
`0xf9005400` содержит field 84, а не 85, при фактическом 2040/510. Поэтому
таблица KB «N=84 → 2016» / «N=85 → 2040» имеет вероятный off-by-one в
обозначении N: по Linux-visible semantics baseline — raw84 → 2040/510,
следующий raw85 ожидался бы как 2064/516, если формула `(raw+1)*24` верна.
Исторический факт, который сохраняется, — был заявлен write `0xf9005500`;
его raw value сейчас не активен. Не использовать старые обозначения N=84/N=85
как доказанные MHz без source-verified formula. Остаётся отдельная SDM/
fractional-PLL неопределённость: старый KB упоминает pattern registers
`0x02002028/0x0200202c`, но их текущий readback и clock-driver source не
получены; поэтому `(raw+1)*24` — сильная согласованная гипотеза, не окончательная
документация PLL semantics.

**Проверенный факт.** В tracked branches/tags, reflog, найденных dangling
commits, Beads notes/issues, имена untracked-файлов Orange RAG и
`/home/random/orangepi-backup` не найдено более первичного скрипта, лога или
before/after benchmark этого эксперимента. В частности, `orange-RAG-16p.2.11`
содержит только результат предыдущего аудита; поиск Beads по DDR/DRAM пуст.
В `orangepi-backup` обнаружены только NPU/GPU binary artifacts. Локальный
`/home/random/orangepi-kernel` — headers/deb snapshot без реализаций `ccu-ddr.c`,
`ccu-sun60iw2.c` или `sunxi-dmc.c`, поэтому call path нельзя восстановить из
исходника этого snapshot.

**Проверенный факт.** На самой плате read-only metadata текущего boot path:

| Артефакт | size / mtime | SHA-256 | Наблюдение |
|---|---:|---|---|
| `/boot/orangepiEnv.txt` | 208 / 2026-06-18 09:24:40 UTC | `997eaa824399b86870e5a0b0b6851e8795966c2a592498b2c3e7229a63004225` | содержит `dram-opp` |
| `orangepiEnv.txt.auto.20260618-092248` | 208 / 2026-06-18 09:22:48 UTC | тот же | совпадает с current |
| `orangepiEnv.txt.backup` | 199 / 2026-06-17 13:51:00 UTC | `62a09786c575c11f5fdc8eda59ef6b5904d0b34e4acc253530d759b9c29829a8` | старый вариант без найденного `dram-opp` |
| `orangepiEnv.txt.backup-lkg` | 199 / 2026-06-18 08:57:46 UTC | тот же | совпадает с backup |
| `sun60i-a733-dram-opp.dtbo` | 794 / 2026-06-18 09:55:11 UTC | `4a702d0bff118b64c40c3f095290d78a6d9aa551c1e35cf4955da88e0535a58d` | активный overlay |
| `sun60i-a733-orangepi-zero3w.dtb` | 213628 / 2026-04-07 04:26:47 UTC | `54ad198e9019a778d4ff936bd18808b68b828ec452491920aaf9c088f4bffca6` | текущий base DTB |
| `sun55iw3-devfreq.ko` | 30200 / 2026-06-18 11:21:45 UTC | `884c4be170b8c0caf2054cb9e24ab516c45da95cb22ffb01a86ef5086a325e38` | модуль на target |

`boot.scr` и его `backup`/`backup-lkg` также различаются hash (текущий
`ad66e729...`, backups `8d3d5617...`), но `boot.cmd` явно применяет overlay из
`overlays`; это метаданные, не доказательство фактического rate. Read-only
`dmesg`/`journalctl -k` от непривилегированного пользователя не предоставили
строк `sun55iw3`/`dmcfreq`/thermal, поэтому thermal stop не подтверждён журналом.

**Проверенный факт (новый read-only probe владельца).** Root read-only
`debugfs/clk_summary` на плате прямо показал `pll-ddr=2040000000 Hz`,
`sdram=2040000000 Hz` и `dram0=510000000 Hz`, причём `dram0` имеет parent
`pll-ddr`. Это устанавливает текущий clock-controller rate 510 MHz и
совместимо с историческим KB по выходному rate, но не доказывает, что старый
`/dev/mem` write всё ещё влияет на живую систему: тот же rate может быть
штатным bootloader/training result. `sdram` и `pll-ddr` — parent/PHY-related names,
`dram0` — controller clock; CPU/NPU clocks к этому readback не относятся.

**Проверенный факт (новый root read-only `/dev/mem` readback).** Для текущей
платы прочитаны без записи: `PLL_DDR_CTRL@0x02002020 = 0xf9005400` и
`DRAM_CLK_REG@0x02002c00 = 0x80000003`. Следовательно, после reboot старый
записанный raw `0xf9005500` не сохранился: текущий PLL field bits 15:8 равен
`0x54` (84), а divider field bits 4:0 равен 3 (`/(3+1)`). При этом debugfs
показывает parent 2040/child 510; значит, vendor/Linux clock semantics
почти наверняка используют `(raw_N + 1) * 24 MHz` (`(84+1)*24=2040`), а не
старую KB-интерпретацию `raw_N*24=2016`. Это согласование по двум readback,
но точная формула должна быть подтверждена clock-driver source.

### Vendor-module interpretation: `dram_para00`, `dram_para24`, PLL/divider

**Проверенный факт.** В strings/symbols AArch64 модуля target
`sun55iw3-devfreq.ko` найдены `sunxi_dmcfreq_probe`,
`sunxi_dmcfreq_get_cur_freq`, `dmcfreq_sun60iw2_data`,
`dev_pm_opp_of_add_table`, `devm_devfreq_add_device`, `dram_para[30]`,
`dram_para[24]`, `dram_para24`, `dram_div`, `change dram clock error!`, а также
OF match `allwinner,sun60iw2-dmc`. Это подтверждает намеренный kernel path:
probe читает `/dram`/параметр 24, ищет divider, добавляет OPP/devfreq и пытается
менять DRAM clock. Дизассемблированного vendor source/call path с адресами
регистров в доступном локальном tree нет; strings не показывают содержимое
битовых масок.

**Проверенный факт (восстановленный call path из disassembly владельца).**
`sunxi_dmcfreq_probe` сначала читает `dram_para30` как `tpr13`; если bit2 не
установлен, печатает `disable devfreq` и возвращает `-ENODEV`. Затем читает
`dram_para24` и извлекает четыре поля по 5 бит: bits 0, 8, 16, 24. Каждое
поле — `divider_minus_1`; для каждой точки rate вычисляется как
`parent_rate * 4 / (field + 1)` (в disassembly видны `and #0x1f`, `ubfx`, `+1`,
`lsl rate,#2`, `udiv`). Это устанавливает, что `dram_para24` — не PLL N,
а packed divider word для четырёх DMC rate points.

**Проверенный факт.** На активной плате `dram_para24=0x0f0f0f0f`: все четыре
поля равны 15, то есть divider=16. При parent `pll-ddr=2040 MHz` каждая
точка получается `2040*4/16=510 MHz`; это независимо согласуется с
`clk_summary` `dram0=510 MHz`. Известный vendor example `dram_clk=1800`,
`dram_div=0x11080503` в порядке low-to-high декодируется как делители
4/6/9/18 и точки 1800/1200/800/400 MHz (порядок зависит от rate array), что подтверждает
формулу, но не является экспериментом на этой плате.

**Проверенный факт.** В активном DT `dram_para30` отсутствует. Это объясняет,
почему модуль с alias `allwinner,sun60iw2-dmc` не публикует DDR devfreq и
почему `/sys/class/devfreq` не содержит DMC: guard bit2 не разрешает его
enable. `dram_para00=1200` при этом читается clock driver как `dram_clk:1200`
в dmesg (наблюдение `sunxi:ccu_ddr dram_clk:1200, dram_div:0xf0f0f0f`), но
текущий controller readback остаётся 510 MHz из `dram_para24` и parent
2040 MHz. Это разводит vendor init parameter, DMC enable gate и фактический
clock-controller rate.

**Проверенный факт.** В старом KB clock-divider описан как регистр
`0x02002c00` (DDR CCU base `0x02002000` + offset `0xc00`), `div` bits 4:0,
а текущий readback даёт `2040/510 = 4`. Следовательно, для текущего `dram0`
наблюдается деление parent на 4; это согласуется с raw field `3` и формулой
`/(DIV+1)`, заявленными KB. Однако это подтверждает только clock-summary и
историческую таблицу; этот регистр относится к clock-provider divider, а
`dram_para24` — к DMC rate-array и не должен автоматически отождествляться с
одним raw CCU register field.

**Неизвестно.** Точный способ, которым bootloader/CCU выставляет PLL `N` и
обеспечивает parent 2040 MHz, не восстановлен из исходника или U-Boot log.
Исторический KB отдельно и явно называет PLL `N` (bits 15:8) кандидатом для
фактического `/dev/mem` изменения; это отдельный механизм от `dram_para24`.

**Проверенный факт / оставшийся пробел.** `dmcfreq_sun60iw2_data` присутствует
как локальный объект модуля; disassembly восстановил его `dram_para30` gate и
`dram_para24` decoding, но headers-only snapshot не содержит исходного
инициализатора/комментариев `sunxi-dmc.c`. Не восстановлены только точные
структуры OPP array и ветви supplier error после gate; это не меняет вывода,
что текущий unbound path объясняется отсутствующим bit2.

**Проверенный факт.** Текущие DT/boot metadata, активный overlay и модуль
совместимы по имени с описанным DMC path, но runtime DMC остаётся unbound:
нет DDR `/sys/class/devfreq`, а `/sys/devices/platform/a020000.dmcfreq/driver`
отсутствует. Поэтому старое прямое `/dev/mem` изменение и поддержанный kernel
OPP A/B — разные эксперименты с разными владельцами clock.

### Что доказано о thermal stop, а что нет

**Сообщение владельца проекта.** Остановка старой ветки была связана с сильным
thermal throttling; новое охлаждение теперь установлено. Это принимается как
provenance для планирования, но не как измеренный DDR thermal log.

**Прошлое наблюдение.** Beads thermal work документирует CPU cooling epoch:
после установки extra cooler policy0/policy6 держали свои максимальные CPU
частоты, `cpufreq cooling0`, matching kernel throttle events не наблюдались.
Это подтверждает пользу нового охлаждения для CPU workload, но не доказывает
отсутствие DDR PHY/DMC thermal throttle в старом OC и не является заменой
`ddr_thermal_zone` time series.

**Неизвестно.** Не найдены thermal zone name/peak/time series, cooling-state
transition, kernel log или frequency trace, связывающие именно старый N=85
или другой bit/multiplier с throttling. Точная причина остановки (DDR PHY,
SoC rail, CPU/board sensor, watchdog/driver fault или сочетание) требует
восстановления первичного лога.

### CPU A/B, ошибочно принимаемые за memory tuning

**Прошлое наблюдение.** CPU baseline `bonsai27b-q1-cpu-a55-pp512-tg128-001`:
6 A55, decode median **0.726175 tok/s**, peak RSS 7308.152 MiB, swap delta 0,
DDR thermal median/peak 51.646/56.172 °C, cpufreq cooling state 0, без
matching throttle/OOM/accelerator событий. A76 run 003 дал 0.649836 tok/s
decode. Это сравнение CPU topology, не DDR A/B; частота DDR в этих runs не
менялась и не записывалась.

**Прошлое наблюдение.** NPU phase profile держал NPU на 1008 MHz и измерял
resident ShuffleNet H2D/run/D2H. Это отдельный NPU workload; он не доказывает
DDR bandwidth и не является tokens/s. H2D 2.272 GB/s — effective API-path
скорость для 150,528-byte input, включая map/unmap/cache maintenance.

**Проверенный факт.** Foundation `tooling/target_inventory.sh` и
`tooling/profile_command.py` собирают CPU, thermal и NPU devfreq, но не DDR
devfreq/clock/OPP. Поэтому прежние чистые профили не могли обнаружить
отсутствие DMC bind или подтвердить DDR rate.

## Текущая видимая topology

### Target runtime snapshot

| Область | Наблюдаемое сейчас | Что это означает | Доказанность |
|---|---|---|---|
| RAM | `MemTotal=12162208 kB`, `free -h` ≈ 11 GiB, swap 5.8 GiB | capacity, не bandwidth | Проверенный факт |
| CPU A55 | policy0, CPUs 0–5, 416–1794 MHz, governor `ondemand`, current 1794 MHz | CPU clock domain | Проверенный факт |
| CPU A76 | policy6, CPUs 6–7, 416–2002 MHz, governor `ondemand`, current 780 MHz at snapshot | CPU clock domain | Проверенный факт |
| NPU | `/sys/class/devfreq/3600000.npu`: 492/852/1008 MHz, current 1008, governor `performance` | NPU, не DDR | Проверенный факт |
| DDR devfreq | `/sys/class/devfreq`: только `3600000.npu`; DDR entry отсутствует | no visible runtime DDR selector | Проверенный факт |
| DDR platform | `a020000.dmcfreq` exists, driver none; `sunxi-dmcfreq` module exists; `2002000.clk_ddr` → `sunxi-ddrclock` | DMC control path is not bound; clock provider is bound | Проверенный факт |
| DDR clocks | debugfs `pll-ddr=2040000000`, `sdram=2040000000`, `dram0=510000000`, `dram0` parent `pll-ddr` | actual controller clock readback is 510 MHz; not `dram_para00=1200` | Проверенный факт |
| DDR thermal | `ddr_thermal_zone` ≈ 32.984 °C; DT critical 110 °C, no passive/trip cooling shown | temperature sensor/protection only | Проверенный факт |

### DT/boot clock and OPP graph

**Проверенный факт.** Base DT contains:

```text
/dram: compatible = "allwinner,dram"; clock-names = "pll_ddr"
/clk_ddr: compatible = "allwinner,sun60iw2_clock_ddr"; reg = 0x2002000
/opp_table/opp@150000000: opp-hz = 150000000; opp-microvolt = 900000
/dmcfreq@3120000: compatible = "allwinner,sun60iw2-dmc", "syscon"
  operating-points-v2 = /opp_table
  vddcore-supply = /vdd-sys; normalvoltage = boostvoltage = 900000
```

**Проверенный факт.** Активный DT overlay `sun60i-a733-dram-opp.dtbo` содержит:

```text
fragment@0 target /dram:
  dram_para00 = <0x4b0>;          /* 1200, semantic unit vendor-specific */
  dram_para24 = <0x0f0f0f0f>;
fragment@1:
  dram_opp_table_overlay:
    opp@510000000 { opp-hz = 510000000; };
    opp@600000000 { opp-hz = 600000000; };
fragment@2 target /dmcfreq@3120000:
  operating-points-v2 = dram_opp_table_overlay;
```

**Гипотеза.** Для DDR/LPDDR5 значение 600 MHz может соответствовать double
data rate около 1200 MT/s, а текущие 510 MHz — около 1020 MT/s; MT/s остаётся
инженерной интерпретацией, потому что readback доказывает controller clock,
не PHY I/O/data rate. Не смешивать `opp-hz` clock, data rate и `dram_para00`.

**Проверенный факт.** A733 datasheet V0.93, раздел 3.2.2, заявляет поддержку
внешней 32-bit LPDDR4/LPDDR4x/LPDDR5 и LPDDR5 clock до 2400 MHz. Это предел
SoC datasheet, не доказательство, что soldered memory, SI/training, PMIC и
конкретный Orange Pi Zero 3W безопасно работают на такой частоте.

## Кто задаёт DDR rate

| Слой | Evidence | Вывод | Уверенность |
|---|---|---|---|
| BootROM/U-Boot/firmware | boot image содержит DRAM init/training strings (`DRAM Pstate ... frequency`, `DRAM CLK`); kernel стартует уже с рабочей RAM | initial rate/training почти наверняка до Linux | Средняя, лог не сохранён |
| Boot DT/overlay | `boot.cmd` применяет `dram-opp`; overlay задаёт `dram_para00`, OPP и DMC phandle | boot configuration selects available policy/table | Высокая для DT fact, средняя для actual selection |
| Kernel clock | `clk_ddr` bound to `sunxi-ddrclock` | clock provider exists | Высокая |
| Kernel DMC/devfreq | module loaded, platform node exists, но no driver bind и no `/sys/class/devfreq` DDR | currently not observed as active owner/control API | Высокая для snapshot |
| Kernel module gate | disassembly: `dram_para30` bit2 unset → `disable devfreq`/`-ENODEV`; active DT has no `dram_para30` | explains why kernel DMC is not current owner, despite module/OPP strings | Высокая |
| Clock readback | debugfs: `pll-ddr=2040`, `dram0=510`; `dram_para24` four fields=15 → divider 16 | current rate is clock-provider/boot-established 510 MHz | Высокая |
| Voltage | `/vdd-sys` fixed 900,000 µV; DMC normal/boost both 900,000 µV; no DRAM-specific adjustable regulator exposed | no evidence of voltage scaling path | Высокая |

**Проверенный факт / неизвестно.** Read-only `clk_summary` уже устанавливает
текущий controller clock `dram0=510 MHz`; `600 MHz` пока только доступный
overlay OPP и не является текущим readback. Без PHY-specific readback или
U-Boot training log нельзя превращать 510 в доказанные MT/s и нельзя утверждать,
что bootloader применил именно старый N=85 write.

## Что раньше прошло/не прошло и почему

**Проверенный факт.** CPU A55/A76 tests прошли как стабильные throughput
наблюдения (5 samples, no swap/throttle/OOM), но остались `unqualified`:
`llama-bench` не даёт TTFT и deterministic-token quality contract. Их успех —
только reproducible CPU baseline.

**Проверенный факт.** NPU ShuffleNet resident run прошёл 1000/1000 вызовов и
repeat-equal, но остаётся `performance-observed-unqualified`: нет независимого
CPU golden для закрытой fixture и это не LLM tokens/s. DDR rate там также не
измерялся.

**Прошлое наблюдение / неизвестно.** Внешний KB заявляет для raw write
`0xf9005500` lock=1, `memtester 128M 1=pass` и 11 минут стабильности, а также
отдельный operational hang из-за `sudo` pipe/watchdog. Это не independently
verified primary log. Результатов `STREAM`, `mbw`, `stressapptest`, ECC/error
injection, memory checksum после смены OPP или reboot recovery не найдено;
поэтому долговременная целостность и thermal safety старой попытки остаются
неподтверждёнными.

## Риски целостности данных

**Проверенный факт.** На target нет evidence ECC: `/proc/meminfo`, найденные
sysfs nodes и kernel profile не показывают EDAC/ECC correction counters. Для
этого аудита ECC считаем отсутствующим/неизвестным, а не защитой.

**Гипотеза.** Превышение validated DRAM rate или неподходящие PHY timing/
training/voltage может давать silent bit flips, редкие SIGBUS/kernel faults,
filesystem/model corruption, зависание при DMA и failure to boot. Ошибки могут
появиться только под concurrent CPU/NPU/DDR traffic или после прогрева.

**Проверенный факт.** DMC table имеет fixed 900 mV `vdd-sys` и только два
overlay OPP; overlay не содержит доказанной LPDDR5 timing/training sequence.
Добавление произвольного OPP выше 600 без vendor-параметров — не эквивалент
смене CPU governor и имеет высокий риск unrecoverable boot/data state.

## Влияние на Bonsai Q1 tokens/s

**Проверенный факт.** Зафиксированная CPU A55 baseline — 0.726175 decode
tok/s. Q1 audit оценивает минимальный поток полных Q1-весов как
`3,603,087,360 B/token` (без embedding matrix, F32/KV/state/overhead), то есть
при 0.726175 tok/s около **2.616 GB/s** логического weight-read demand.
Это не raw DDR bandwidth: cache reuse, read/write traffic, CPU work и runtime
overhead отдельно не измерены.

**Гипотеза.** Если decode memory-bound и реально сравниваются 510→600 MHz
(+17.647% clock), идеальный потолок —
`0.726175 × 600/510 = 0.854324 tok/s`, +0.128149 tok/s. При memory fraction
70/80/90% Amdahl-оценка даёт примерно 0.811/0.825/0.840 tok/s. Это upper-bound
scenario, не результат и не доказательство, что текущая частота 510 MHz.

| Изменение validated DDR clock | Ideal memory-bound decode | Amdahl при memory fraction 80% |
|---:|---:|---:|
| +10% | 0.798793 tok/s | 0.783130 tok/s (сценарий, не факт) |
| +15% | 0.835101 tok/s | 0.810778 tok/s (сценарий, не факт) |
| 510→600 (+17.647%) | 0.854324 tok/s | 0.825199 tok/s |
| +25% | 0.907719 tok/s | 0.864495 tok/s (сценарий, не факт) |

Все Amdahl-числа — сценарии при предположенной доле memory time 80%; без
измеренной доли их нельзя подменять точными claims. Главный criterion — end-to-end decode median,
а не расчётная `3.603 GB/token`; additionally record sustained bandwidth,
CPU cycles, cache misses и non-memory wall time.

## Безопасный будущий A/B-план (только после approval)

### Phase 0 — freeze и доказательство baseline

1. Сохранить read-only inventory: kernel/DTB/DTBO/boot config hashes, DT
   decompile, `/sys/class/devfreq`, CPU policies, regulators, thermal trips,
   `dmesg`/journal interval, model/runtime hashes. Подготовить serial/physical
   recovery и проверить, что известный-good SD/eMMC image и boot media доступны.
2. Сначала выяснить ownership: vendor-supported способ bind/query DMC, либо
   read-only U-Boot/boot log. Если DMC остаётся unbound, не пытаться писать
   `target_freq` — такой A/B невалиден.
   Следующий безопасный probe: read-only `clk_summary` до/после cold boot,
   decompile active DT `/dram` (включая `dram_para30/24`), `readlink` driver
   bind, OPP list и доступные journal/dmesg строки; отдельно проверить, что
   `dram0`, `sdram`, `pll-ddr`, CPU и NPU не перепутаны. Не загружать/выгружать
   модуль и не добавлять `dram_para30` этим probe.
3. A-control: exact Bonsai Q1 GGUF, A55 0–5, same llama.cpp commit/command,
   one warm-up + 5 measured decode repetitions, deterministic output capture,
   no CPU/NPU governor changes during pair. Record requested/effective affinity.

### Phase 1 — bandwidth-only, no frequency change

4. На охлаждённой плате выполнить `STREAM` (Copy/Scale/Add/Triad) и `mbw` с
   одинаковыми buffer sizes, thread/affinity, repetitions и page policy;
   record median/p10/p90/max, GB/s, RSS, swap delta, CPU/DDR temperatures.
   Это baseline raw-ish memory signal, но STREAM/mbw не заменяют model run.
5. Проверить CPU↔DDR relationship на fixed CPU clocks/governor policy; отдельно
   NPU clock/thermal telemetry, чтобы не приписать NPU/CPU change DDR.

### Phase 2 — supported OPP A/B before any overclock

6. A-control — текущий readback `dram0=510 MHz` с exact hashes выше. B — только
   существующий DT OPP `600000000 Hz` из `sun60i-a733-dram-opp.dtbo`, через
   vendor-supported DMC/devfreq owner после подтверждения bind и readback
   (`dram0=600 MHz`, parent/PLL и voltage). Важная проверка: при нынешнем
   parent 2040 MHz и `dram_para24=0x0f0f0f0f` (divider 4) этот word сам по себе
   даёт только 510; 600 потребует подтверждённого vendor перехода parent к
   2400 MHz либо иного точно восстановленного rate tuple. Наличие `opp-hz=600`
   в DT недостаточно. Если driver остаётся unbound, runtime sysfs/debugfs
   write и прямой `/dev/mem` запрещены: допустим только заранее проверенный
   boot-time configuration path с serial recovery. Не восстанавливать N=85/86/88
   или менять `dram_para24` до нахождения точного источника и доказанного
   call path.
   Если owner одобрит именно восстановление исторического PLL ladder, первый
   кандидат для отдельного A/B — raw field 84→85 (ожидаемо 510→516 по
   readback semantics), затем остановка и проверка; raw86 (ожидаемо 522) —
   только отдельный следующий approval. Это не заменяет supported 510↔600
   OPP-путь и не разрешает прямую запись сейчас.
7. Перед B выполнить безопасный read-only probe: повторить `clk_summary`,
   `of_node`/compatible, OPP list, driver bind, thermal zones и journal/dmesg;
   сохранить hashes, затем после approved transition проверить, что изменился
   только `dram0`/его parent и что CPU A55/A76 и NPU domains не изменились.
8. Для каждого OPP: cold/cooldown baseline → STREAM/mbw → `memtester` or
   `stressapptest` on a bounded, non-swapping allocation → reboot → rerun
   checksum/data-integrity smoke → same llama decode. `memtester`/stressapptest
   must not run while model data is the only copy and must respect a memory
   budget; never infer safety from one short pass.
9. Thermal guard polls all zones, especially `ddr_thermal_zone`; terminate
   workload at conservative 85 °C guard (existing fan policy is 30 °C and DDR
   DT critical is 110 °C). Record fan/cooling state, no ECC counters, OOM,
   kernel/DDR/VIP faults, and exact termination reason. Critical 110 °C is an
   emergency limit, not a target.

### Phase 3 — only if supported A/B passes

10. Any >600 MHz candidate requires vendor-authorized DRAM OPP/timing/training,
   voltage/SI evidence, exact DT/bootloader artifact and an explicit rollback
   gate. Do not extrapolate from A733 datasheet's 2400 MHz SoC maximum.
11. Promote only if both STREAM/mbw and model decode improve, five-repeat
    median gain exceeds measurement noise (suggested ≥5% end-to-end and
    confidence interval excludes zero), deterministic token agreement is 1.0,
    no memory test/checksum/journal fault occurs, swap delta is zero, and all
    thermal/frequency traces are complete. Otherwise mark no-go.

### Rollback

**Проверенный факт.** Current boot path explicitly names `dram-opp` overlay;
this is the exact artifact to remove/restore only under approved change control.

**План rollback.** Keep immutable known-good DTB/DTBO/boot config and image
hashes; restore prior boot config/overlay and reboot only when approved. If the
board fails to boot, use serial/physical boot-media recovery or reflash the
known-good image, then verify model hash, filesystem checks and full read-only
inventory. A runtime data error requires stopping inference, checking model/
sidecar hashes and restoring from known-good storage backup; do not continue
benchmarking after a checksum mismatch.

## Measurable acceptance / rejection criteria

- `DDR rate`: must have direct, repeatable readback from supported kernel/U-Boot
  path; DT `opp-hz` alone is insufficient.
- `STREAM/mbw`: same affinity/threads/allocation; report median and CV; gain
  should track OPP direction without unexplained tails.
- `Bonsai`: exact model hash, same CPU topology/runtime/command; decode median,
  p10/p90, CV, TTFT where applicable, RSS, min available RAM, swap delta.
- `Correctness`: deterministic generated-token agreement 1.0 and model/sidecar
  checksums before/after memory stress and reboot.
- `Integrity`: no ECC is assumed; require zero `memtester`/`stressapptest`
  errors, zero kernel bus/DDR/VIP/OOM faults, and zero thermal guard aborts.
- `Thermals`: record all zones and cooling states; no claim of no throttling
  without cpufreq/devfreq/journal evidence.
- `Rollback`: known-good image/boot artifacts and physical recovery tested
  before first frequency-changing trial.

## Audit status

**Проверенный факт.** This audit created no board-side or repository changes;
the only intended new artifact is this report. No DDR frequency was changed.

**Проверенный факт / неизвестно.** Current controller readback is now proven
as `dram0=510 MHz` with raw PLL register `0xf9005400`/divider `0x80000003`;
the historical raw85 write is not persistent after reboot. Whether the current
board is physically populated with the same LPDDR5 configuration assumed by
the profile, whether 1200 in `dram_para00` denotes MT/s, MHz, or another vendor
parameter, and whether 600 OPP can be reached without a parent/rate-tuple
change remain open.
