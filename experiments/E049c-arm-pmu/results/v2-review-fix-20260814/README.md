# E049c-v2 review fix — finite inputs и раннее резервирование output

Статус: **PASS для двух коротких instrumentation controls; Bonsai не
запускался**. Эта серия проверяет исправления launcher-а после review, а не
скорость модели и не DDR traffic.

## Что именно исправлено

1. Все параметры, разбираемые через `strtod`, принимаются только если
   `isfinite(value)` истинно. Значения `nan`, `NaN`, `inf` и `-inf` для
   `--min-running-ratio` и `--max-temp-c` отвергаются до workload; JSON не
   сериализует не-конечные числа.
2. Output резервируется до pipe, `fork` и `perf_event_open` через
   `O_EXCL|O_NOFOLLOW`; открытый fd остаётся у parent до финальной записи.
   При существующем файле или symlink child не создаётся, PMU не запускается,
   а workload не получает управление. Это доказано adversarial host-тестами
   с отдельным side-effect marker.
3. Значение `4` переименовано в `software_group_size_limit`. Это
   консервативная политика launcher-а, **не измеренная аппаратная ёмкость
   PMU**. Предыдущая target-серия доказывает только, что группы размера
   `3/3/2` работали без multiplexing при
   `time_running/time_enabled=1.0`; произвольная группа из четырёх событий не
   проверялась.

## Точный pin helper-файлов

Перед сборкой на target сверены SHA-256 локальных и переданных исходников:

| Файл | SHA-256 local = target |
|---|---|
| `a733_pmu_exec.c` | `f754845e60b22f118cf82b2f462537e02f5c404e1f1d7b43de657930b8bb6dc0` |
| `a733_pmu_control.c` | `3923b982de69116a94ff20edb5665b4ed426616a85fbe23538b78950748804b2` |
| `a733_pmu_sync_exec.sh` | `71fd0c36b9f1ac9aa9bca8c967ad8812d63e474edd8f7e43eeeb817eeb4d3efa` |

Собранные на AArch64 бинарники имели SHA-256
`a4c1be74a10fc213d0bec5f68b199a8e7b57d28da9952ecb36af3d2efeb7b534`
для launcher-а и
`0c0556897ef8ef068b8af18a667fdfc78221879b174b4036ea196ca019b70072`
для control. Полная запись находится в `raw/helper-sha256.txt`.

Target: `orangepizero3w`, Linux `6.6.98-sun60iw2`, AArch64,
`perf_event_paranoid=2`. Parent запускался root-assisted, но launcher по
умолчанию сбрасывал права child до `orangepi:orangepi`. Пакеты не
устанавливались, OPP/DDR/thermal policy не менялись.

## Короткие target controls

| Control | Результат | Время | Температура max | Точные core counts |
|---|---|---:|---:|---|
| identity через fixed-fd shim | `uid=1000(orangepi)`, `status=ok`, `sample_valid=true`, полный `S/ACK/E` | 5.986250 ms | 37.138 °C | cycles 5,174,845; instructions 2,121,395; stall 1,837,705 |
| CPU 1,000,000 итераций | checksum `1056668018662160035`, `status=ok`, `sample_valid=true`, полный `S/ACK/E` | 22.084872 ms | 35.216 °C | cycles 16,548,820; instructions 15,116,359; stall 6,432,703 |

У всех шести событий `support=supported`, индивидуальный
`sample_valid=true`, а `time_running/time_enabled=1.000000000`. Thermal guard
85 °C был читаем и не сработал. Пустые stderr сохранены как нулевые raw-файлы,
а exit code обоих запусков равен `0`.

Эти counts — события ARM PMU, **не DDR bytes**. Два коротких прогона не
доказывают производительность Bonsai, аппаратную ёмкость PMU или отсутствие
multiplexing у другой комбинации событий.

## Воспроизведение

После сверки SHA-256 исходники на target были собраны без установки ПО:

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror a733_pmu_exec.c -o a733-pmu-exec
cc -std=c11 -O2 -Wall -Wextra -Werror a733_pmu_control.c -o a733-pmu-control
chmod 700 a733_pmu_sync_exec.sh a733-pmu-exec a733-pmu-control
```

Identity-probe:

```bash
printf '<sudo-password>\n' | sudo -S -p '' ./a733-pmu-exec \
  --output identity-core.json --event-group core \
  --min-running-ratio 0.95 --start-on-ready --sync-timeout-ms 5000 \
  --max-temp-c 85 -- ./a733_pmu_sync_exec.sh /usr/bin/id
```

CPU-control отличался только командой workload:

```bash
./a733-pmu-control --mode cpu --iterations 1000000
```

В репозитории нет пароля, model payload или proprietary SDK payload. Старые
raw-артефакты не изменялись.
