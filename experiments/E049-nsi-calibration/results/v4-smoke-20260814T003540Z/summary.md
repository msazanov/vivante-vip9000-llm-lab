# E049 — калибровка sunxi-nsi (A733)

Статус: **complete**; исходный `pmu_timer`: `0` raw; thermal abort: `85.0 °C`.

Классификация `rate`/`volume` — только fit-гипотеза. Она не заменяет документирование аппаратных единиц; сырые значения сохранены в `raw.jsonl`.

| Сигнал (reported aggregate channel) | Гипотеза | Уверенность | R² volume | R² active-rate | idle ratio | CV read |
|---|---:|---:|---:|---:|---:|---:|
| `pmu_bandwidth` | insufficient | low | n/a | n/a | 0.07696836018420543 | 0.35295607982235205 |
| `pmu_bandwidth_rd` | insufficient | low | n/a | n/a | 0.06922112606965716 | 0.35340149031760737 |
| `pmu_bandwidth_wr` | insufficient | low | n/a | n/a | 0.24917634818796602 | 0.34315935495058086 |
| `pmu_cmd_rd` | insufficient | low | n/a | n/a | 0.06627552573932631 | 0.35460367418893607 |
| `pmu_cmd_wr` | insufficient | low | n/a | n/a | 0.24868328651685392 | 0.3432672050561798 |
| `pmu_latency_rd` | insufficient | low | n/a | n/a | 0.5743801652892562 | 0.4380165289256198 |
| `pmu_latency_wr` | insufficient | low | n/a | n/a | 0.9574468085106383 | 0.44680851063829785 |

## Ограничения

- Инструмент записывает только `pmu_timer`; файлы управления портами не обнаруживаются и не записываются.
- `bandwidth_*` в текущем ядре масштабируются data-unit (для A733 в DT IA=16, TA/CPU/RA=64); software source не делит значение на timer.
- Для причинности нужны повторения, холодный/горячий cache и независимый контроль bytes read; это первый bounded calibration gate.
