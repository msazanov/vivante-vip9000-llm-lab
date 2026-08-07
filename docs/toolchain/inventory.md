# Supplied toolchain inventory

**Date:** 2026-08-07  
**Scope:** sanitized host/backup inventory only; vendor binaries, SDK archives, NBG files, weights, credentials, and private documents stay outside Git.

## Question

Which compiler-side and backup runtime materials can be identified without claiming that the host toolchain targets the A733/VIP9000 board?

## Current answer

**Verified in supplied tooling —** the host has AcuityLite Docker images and an embedded VeriSilicon SDK archive with identifiable versions, marker strings, library families, and headers.  
**Unknown/contradictory —** the embedded marker `GCNANOULTRA31_VIP2_PID0X15` is not proof of compatibility with A733 VIP9000.  
**Verified in supplied tooling —** a sanitized backup contains AArch64 `vpm_run`, one NBG, input data, and a runtime-library archive; these are provenance evidence, not a benchmark or a license claim.

## Evidence

- **Verified in supplied tooling —** host Docker tags included `acuitylite:ready`, `acuitylite:latest`, and `ubuntu-npu:v2.0.10.2`. The `acuitylite:ready` image was AcuityLite 6.51.0 and approximately 12.4 GB.
- **Verified in supplied tooling —** the AcuityLite image contained `/usr/local/lib/python3.10/dist-packages/acuitylib/vsi_sdk.tar.gz`.
- **Verified in supplied tooling —** the embedded x86 SDK reported `VERSION CL769531_D769480_A769529_R768876_T768536_O768454` and build marker `5173979`.
- **Verified in supplied tooling —** SDK library names included `libtim-vx`, OpenVX, VSC, CLC, OpenCL, GAL, SPIRV, `ovxlib`, and NNArchPerf; headers included custom operations and matrixmul/fullconnect/rmsnorm declarations.
- **Verified in supplied tooling —** `libtim-vx` strings contained VXC/EVIS source markers.
- **Hypothesis —** VXC/EVIS strings indicate compiler-side support may exist in the supplied SDK; they do not establish target runtime support or usable A733 code generation.
- **Unknown/contradictory —** no supplied-toolchain observation connects the `GCNANOULTRA31_VIP2_PID0X15` marker or x86 SDK to the target's device-tree identity, driver, or VIPLite libraries.
- **Verified in supplied tooling —** `/home/random/orangepi-backup/npu-sdk/vpm_run/vpm_run` was 38,696 bytes with SHA-256 `dbcea3266070d51a4e0b3aac19540814d065f3727a6045ee96e25f9464371407`.
- **Verified in supplied tooling —** `/home/random/orangepi-backup/npu-sdk/vpm_run/network_binary.nb` was 964,224 bytes with SHA-256 `fc0d50bdc863e4dfc74eef6dad9d34655c560e759a2f426979a01c505fa9b95f`.
- **Verified in supplied tooling —** `/home/random/orangepi-backup/npu-sdk/vpm_run/input_0.dat` was 150,528 bytes with SHA-256 `d38c22c2007db314572c9ddfeb513a9f3d11ec99e8deec0f65bc91481502f1b9`.
- **Verified in supplied tooling —** `/home/random/orangepi-backup/npu-sdk/npu-libs.tar.gz` was 106,021 bytes with SHA-256 `75d9fd696d3c17a0daf1f47811b8b33de9734c6bbcfccbfb14c2fd4d6f005cb8`; its listing contained `libVIPhal.so`, `libNBGlinker.so`, and `vip_lite.h`.

## Confidence

**Verified in supplied tooling — high** for tag names, reported versions, sanitized paths, sizes, hashes, and archive member names.  
**Unknown/contradictory — high** for target compatibility; no compatibility conclusion is drawn from a marker, symbol name, or compiler-side string.

## Unresolved

- **Unknown/contradictory —** the exact Docker image digest for each tag was not retained in the supplied observations.
- **Unknown/contradictory —** the target ABI and compiler target selected by the embedded SDK are not established.
- **Unknown/contradictory —** the backup NBG and input pair has no recorded correctness, timing, or thermal result.

## Next experiment

**Hypothesis —** first inventory the SDK and target runtime separately, then compile or convert one minimal operator graph in an isolated, network-disabled, read-only container and compare its metadata with the target's `libNBGlinker`/VIPLite API. Do not execute the resulting graph until compatibility and a quality reference are recorded.

## Read-only reproduction commands

```sh
docker image inspect acuitylite:ready acuitylite:latest ubuntu-npu:v2.0.10.2 --format '{{.RepoTags}} {{.Id}} {{.Size}}'
docker run --rm --network none --read-only --tmpfs /tmp:rw acuitylite:ready sh -lc 'ls -l /usr/local/lib/python3.10/dist-packages/acuitylib/vsi_sdk.tar.gz; sha256sum /usr/local/lib/python3.10/dist-packages/acuitylib/vsi_sdk.tar.gz'
docker run --rm --network none --read-only --tmpfs /tmp:rw acuitylite:ready sh -lc 'tar -tzf /usr/local/lib/python3.10/dist-packages/acuitylib/vsi_sdk.tar.gz | grep -E "(libtim-vx|OpenVX|vip_lite|matrixmul|rmsnorm)"'
for f in /home/random/orangepi-backup/npu-sdk/vpm_run/vpm_run /home/random/orangepi-backup/npu-sdk/vpm_run/network_binary.nb /home/random/orangepi-backup/npu-sdk/vpm_run/input_0.dat /home/random/orangepi-backup/npu-sdk/npu-libs.tar.gz; do stat -c '%s %n' "$f"; sha256sum "$f"; done
tar -tzf /home/random/orangepi-backup/npu-sdk/npu-libs.tar.gz | grep -E '(libVIPhal\.so|libNBGlinker\.so|vip_lite\.h)'
```
