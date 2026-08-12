# E038 — DDR secure DFS и граница безопасного эксперимента

Дата: 2026-08-12
Плата: Orange Pi Zero 3W / Allwinner A733, Linux `6.6.98-sun60iw2`.

Статус: **read-only evidence; runtime write prohibited**.

## Установленные факты

Встроенный Linux provider `sunxi-ddrclock` подтверждён на target Image и локальным BSP source `ccu-ddr.c`. Его `sunxi_ddr_clk_set_rate` не пишет PLL напрямую: выбирает firmware `freq_id`, захватывает mutex и вызывает secure monitor через SMC FID `0xc0000096`. После этого devfreq path читает clock rate и ждёт readback; сам `set_rate` не распространяет SMC error. BL31 содержит dispatch для того же FID. SCP содержит DRAM DFS/training strings и код, но без символов/source нельзя восстановить безопасную последовательность PLL/PHY/training/voltage.

Найденная clock implementation — не разрешение вызывать SMC, bind kernel module или писать MMIO. Raw `0x63` уже привёл к аппаратному reset (E025); повторять его и угадывать промежуточные `0x55/0x56` нельзя.

На target runtime DMC devfreq не был bound; `/sys/class/devfreq` показывал NPU, читаемого DMC readout не было. Единственный допустимый будущий аппаратный путь после source-grounded firmware — cold-boot OPP A/B на объявленном BSP `510/600 MHz`, с serial console, watchdog, physical recovery и fallback на 510 MHz. До этого runtime write/SMC/MMIO запрещены.

## Точные входы и hashes

| Артефакт | SHA-256 |
|---|---|
| vmlinux/Image | `dc7a9756d2daa1de763d9efaa26271f774c886bc52c62ec12d78d80b99f06582` |
| System.map | `02460e1fad978d2d8009b2178f7231a010712f030cd4e07bfacc180e6fee14ee` |
| kernel config | `57c9bca62fefa1ef5f88cb492c889129260e0c98ca8c2c1c7a3206816172f2c9` |
| BSP `ccu-ddr.c` | `c286c6ea09c4d1d8077eaacd1d51d749586eb1e61fb51ae44ccf6fa9ef0566ff` |
| `sun55iw3-devfreq.ko` (reference only) | `884c4be170b8c0caf2054cb9e24ab516c45da95cb22ffb01a86ef5086a325e38` |
| E038 clock report | `591e2ce94d9e2e61772687c3fe4e3441b1f8d963832dc3805772669835d71107` |
| E038 search report | `6e2703ebb4efd11527efe843b899affd7174dc0aec672ad4d71826b509246e96` |
| E038 Image disassembly | `3081ecd00f1e1a7edd84efe38b5e32629dd0b99fda34e799e5d4e2fce30a19ef` |
| E036B safe-step report | `041dba5838628aee0d02a2b261d71e449d75b05f2b91765cd876caae5918db40` |

Vendor binaries, kernel module, firmware и private source archives не коммитятся; здесь сохранены только provenance и hashes.
