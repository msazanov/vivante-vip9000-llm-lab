# E020 — несколько custom EVIS-ядер в одном NBG

## Зачем нужен gate

Эксперимент отделяет три разных ограничения VIP9000:

- количество физических NPU devices/cores;
- количество custom EVIS kernels и graph nodes внутри одного NBG;
- возможность загружать новый kernel непосредственно через target VIPLite.

Target-вызов `vip_query_hardware` вернул:

```text
cid=0x1000003b
device_count=1
device[0].core_count=1
```

Это физически один VIP9000 core. Однако host OpenVX extension позволяет
создавать несколько `vx_program`, несколько раз вызывать
`vxAddKernelInProgram` и связывать несколько generic nodes одним graph.
VIPLite на плате принимает уже готовый NBG и не предоставляет публичного API
для загрузки исходника EVIS/OpenCL.

## Двухузловой тест

`two_stage_invert.vx` содержит два разных entrypoint:

1. `e020_invert_stage_a` инвертирует каждый UINT8 байт;
2. `e020_invert_stage_b` повторно инвертирует внутренний virtual tensor.

Итог обязан побайтно совпасть с input. Builder регистрирует оба kernel,
создаёт два graph nodes и связывает их `vxCreateVirtualTensor`. Наружу
экспортируются только один input и один output.

Vendor `nbinfo` для обоих размеров подтвердил:

- `Core Count: 1`;
- `Layer Count: 2`;
- два `SH` operation, по одному на каждый custom layer;
- `Memory Pool Size: 0`;
- промежуточный tensor отсутствует во внешних input/output tables.

Для 16 байт NBG требует `VIP SRAM Size: 0x18200`. Для virtual tensor 32768
байт требование выросло до `0x20100`, примерно на размер intermediate, при
этом external memory pool остался нулевым. Это сильное свидетельство
внутреннего планирования intermediate в VIP SRAM, но не прямое измерение DDR
PMU.

## Target-результаты

Оба размера выполнены на Orange Pi Zero 3W, VIPLite
`2.0.3.2-AW-2024-08-30`, по 100 итераций:

- 16 UINT8: bit-exact PASS, device mean `9.333 µs`, run mean `48.325 µs`,
  `2664.8` cycles;
- 32768 UINT8: bit-exact PASS, device mean `91.394 µs`, run mean
  `144.954 µs`, `84979.0` cycles;
- 32768 UINT8 H2D mean `10.319 µs`, D2H mean `252.845 µs`;
- output повторялся во всех 100 запусках, температура после теста `34.534 °C`.

Большой D2H показывает, почему промежуточный Q1-unpack нельзя возвращать CPU.
Следующий gate заменяет первый SH layer на packed-Q1 unpack, а второй — на
integer dot/native FC; наружу выходит только малый итоговый projection tensor.

Машиночитаемые данные находятся в
`results/2026-08-11-a733-two-custom-kernel.json`.
