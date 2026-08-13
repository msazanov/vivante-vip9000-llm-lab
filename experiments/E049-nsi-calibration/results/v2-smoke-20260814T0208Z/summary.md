# E049 — калибровка sunxi-nsi (A733)

Статус: **complete**; исходный `pmu_timer`: `0` raw; thermal abort: `85.0 °C`.

Классификация `rate`/`volume` — только fit-гипотеза. Она не заменяет документирование аппаратных единиц; сырые значения сохранены в `raw.jsonl`.

| Сигнал (reported aggregate channel) | Гипотеза | Уверенность | R² volume | R² rate | idle ratio | CV read |
|---|---:|---:|---:|---:|---:|---:|
| `pmu_bandwidth` | insufficient | low | n/a | n/a | 0.02088971658761846 | 0.0 |
| `pmu_bandwidth_rd` | insufficient | low | n/a | n/a | 0.01802342151982336 | 0.0 |
| `pmu_bandwidth_wr` | insufficient | low | n/a | n/a | 0.13459616607613337 | 0.0 |
| `pmu_cmd_rd` | insufficient | low | n/a | n/a | 0.017656601716753685 | 0.0 |
| `pmu_cmd_wr` | insufficient | low | n/a | n/a | 0.13504795517308552 | 0.0 |
| `pmu_latency_rd` | insufficient | low | n/a | n/a | 0.5725806451612904 | 0.0 |
| `pmu_latency_wr` | insufficient | low | n/a | n/a | 1.0 | 0.0 |

## Ограничения

- Инструмент записывает только `pmu_timer`; файлы управления портами не обнаруживаются и не записываются.
- `bandwidth_*` в текущем ядре масштабируются data-unit (для A733 в DT IA=16, TA/CPU/RA=64); software source не делит значение на timer.
- Для причинности нужны повторения, холодный/горячий cache и независимый контроль bytes read; это первый bounded calibration gate.
