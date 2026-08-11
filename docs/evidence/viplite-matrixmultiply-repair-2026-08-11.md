# VIPLite MatrixMultiply на A733: локализация падения и путь ремонта

Дата: 2026-08-11. Цель: ускорение Bonsai-27B Q1 без изменения численной
математики модели.

## Что доказал E018

OpenVX MatrixMultiply требует `INT16` с fixed point position `8`. Попытка DFP0
корректно отклоняется как `VX_ERROR_INVALID_FORMAT (-14)`. Для бинарных весов
DFP8 даёт точное представление: физические `±256` равны `±1.0`, а физическая
активация `q` равна `q/256`; выходной physical INT16 снова равен точному
`Σ(sign×q)`.

Минимальный граф только из MatrixMultiply дошёл до `vxGenerateNBG`, создал
внутренние transpose/matrix узлы, после чего exporter сообщил сохранённые IO
`0/0` вместо реальных `2/1` и завершился SIGSEGV 139. GDB backtrace:

```text
vxGenerateNBG
  → vxoGraph_Process
  → vxoGraph_EndProcess
  → vxoBinaryGraph_ReSaveInputAndPatchTable
  → SIGSEGV: чтение через NULL rbp
```

Это изолирует дефект от Q1 unpack и размера модели: падает даже отдельный
`M1K32` MatrixMultiply. Однопеременные запуски с
`VIV_VX_ENABLE_GRAPH_TRANSFORM=0` и `VIV_VX_ENABLE_NN_TRANSPOSE=0` дали тот же
exit 139.

## Версии и несовпадение toolchain

Плата сообщает PID/CID `0x1000003b`, VIP `ver1=0x9000`, `ver2=0x9202`, runtime
`2.0.3.2-AW-2024-08-30`, API `0x00020003`, kernel module `1.13.0`.

Официальная документация Radxa для A733 требует `ubuntu-npu:v2.0.10.1` и
предупреждает выбирать SDK по конкретному NPU:
https://docs.radxa.com/en/cubie/a5e/app-dev/npu-dev/cubie-acuity-env

Открытый A733 SDK задаёт target
`VIP9000NANODI_PLUS_PID0X1000003B`:
https://github.com/ZIFENG278/ai-sdk/tree/fc90006d0f6569da2f6726c2d8395877686f5aca

Локальные VivanteIDE images содержат только config без `PLUS`. Это сильная
гипотеза несовместимости native exporter, но не доказанная причина: custom
E003 NBG из того же image реально выполнялся на плате и прошёл golden.

## Что найдено в открытых источниках

- Спецификация OpenVX описывает MatrixMultiply и обязательный INT16 DFP8:
  https://registry.khronos.org/OpenVX/specs/1.2/html/dd/d12/group__group__vision__function__tensor__matrix__multiply.html
- TIM-VX публикует EVIS shader `matrixmul_i16.vx` и initializer uniforms:
  https://github.com/VeriSilicon/TIM-VX
- TIM-VX issue #189 показывает зависимость GEMM от согласованной версии SDK и
  символа `vxBatchGemmNode`: https://github.com/VeriSilicon/TIM-VX/issues/189
- TIM-VX issue #217 показывает ошибки регистрации GEMM при смешанных путях
  headers/libraries: https://github.com/VeriSilicon/TIM-VX/issues/217
- NXP delegate issue демонстрирует чувствительность закрытого `vxoBinaryGraph`
  к порядку и shape IO в cached NBG:
  https://github.com/nxp-imx/tflite-vx-delegate-imx/issues/3
- Исходников `vxoBinaryGraph_ReSaveInputAndPatchTable` в открытом доступе не
  найдено; это закрытая часть OVXLIB/exporter.

Проверяемого публичного архива с newer A733 exporter или `PLUS .config` не
найдено. Неизвестные torrent-бинарники не запускались; точного архива с
provenance, размером и SHA-256 также не найдено.

## Выбранный ремонт E019

E019 регистрирует открытый TIM-VX kernel
`com.vivantecorp.extension.evis.gemm_I16I16toI16` как custom EVIS node и тем
самым не создаёт падающий `org.khronos.openvx.tensor_matrix_multiply`.
Lifecycle `program → custom kernel → generic node → vxGenerateNBG` уже доказан
E003 на нашей A733.

Этот обход не гарантирует скорости native NN core. Он нужен для честного
аппаратного ответа: способен ли опубликованный EVIS GEMM быстрее считать
точные K32/K128 partial dots, чем E014. Решение для Bonsai принимается только
по full-layer end-to-end, включая packing, cache sync и FP32 reduce.

## Границы выводов

Профилирование уже показывает, что общая LPDDR — архитектурное ограничение:
Bonsai читает примерно 3.603 GB Q1 projection stream на один токен. Однако
прямого DDR PMU доказательства насыщения шины ещё нет. Текущий E014 ограничен
не H2D/D2H и не температурой, а большим числом `BitExtract/DP16` внутри EVIS:
измеренный tile `1024×5120` занимает 24.137 ms, тогда как CPU выполняет полный
слой `17408×5120` за 3.850 ms. Поэтому E019 должен одновременно убрать
expanded DDR traffic и радикально сократить стоимость dot product.

## Target-результат E019 после перезагрузки

Старый TIM-VX shader успешно экспортировался и выполнял примерно 29 тысяч
cycles для `M1K32`, но возвращал ноль для всех входов. Сравнение с более новым
TIM-VX выявило изменение координат `xyww → xywz` и явную запись `coord_b.z`.
После этой единственной правки положительные golden стали точными, а
отрицательные насыщались в `32767`. Отключение unsigned-saturation bit у
выходного `VXC_DP2x8` сохранило two's-complement знак.

Финальный вариант `coordfix-signed-output-v2` прошёл на VIP9000:

- четыре `M1K32` adversarial случая, по 100 итераций: bit-exact PASS;
- четыре `M1024K128` случая, по 1024 output и 100 итераций: bit-exact PASS;
- среднее device-время `M1024K128`: `6.088–6.190 ms`;
- среднее wall-время: `6.232–6.457 ms`;
- H2D: `71.5–106.4 µs`, D2H: `9.3–24.2 µs`;
- температура выросла примерно с `35.2 °C` до `37.3 °C`, без признаков
  температурного троттлинга.

Таким образом, падение public MatrixMultiply exporter можно обойти custom EVIS
NBG, и сама аппаратная математика VIP9000 работает корректно. Однако измеренные
`~0.021 GMAC/s` делают этот generic INT16 shader непригодным для ускорения
Bonsai. Он остаётся доказательством работоспособности пути и golden-oracle для
следующего прямого packed-Q1 ядра.
