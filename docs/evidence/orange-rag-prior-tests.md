# Prior local observations from `orange-RAG`

**Date:** 2026-08-07  
**Scope:** evidence-only record. These observations are not target-verified results and do not authorize NPU inference.

## Question

What prior artifacts exist in `orange-RAG` or its backups, and what can they support in the clean-sheet research repository?

## Current answer

**Prior local observation —** prior work left model-conversion workspaces, a stopped NPU container, and hashed backup artifacts.  
**Unknown/contradictory —** none of these observations supplies a reproducible performance/correctness record, and the stopped container's OOM state does not prove why it stopped.  
**Prior local observation —** the clean-sheet repository may cite these artifacts as provenance and use their hashes to identify inputs, but must reproduce any claim under the profiling/quality protocol before promotion.

## Evidence

- **Prior local observation —** an existing Mobilenet Acuity workspace contained `mobilenetv2-12.onnx`, converted JSON/data, input metadata, and a calibration input. The artifacts existed, but no associated performance or correctness record was found.
- **Prior local observation —** an old stopped `npu_v2.0.10.2` container had `OOMKilled=true` and mounts for LFM2.5 ONNX files.
- **Unknown/contradictory —** the OOM flag and ONNX mounts do not establish that model size, NPU execution, conversion, or any single subsystem caused the stop.
- **Prior local observation —** the backup AArch64 `vpm_run` was 38,696 bytes with SHA-256 `dbcea3266070d51a4e0b3aac19540814d065f3727a6045ee96e25f9464371407`.
- **Prior local observation —** the backup `network_binary.nb` was 964,224 bytes with SHA-256 `fc0d50bdc863e4dfc74eef6dad9d34655c560e759a2f426979a01c505fa9b95f`.
- **Prior local observation —** the backup `input_0.dat` was 150,528 bytes with SHA-256 `d38c22c2007db314572c9ddfeb513a9f3d11ec99e8deec0f65bc91481502f1b9`.
- **Prior local observation —** `/home/random/orangepi-backup/npu-sdk/npu-libs.tar.gz` was 106,021 bytes with SHA-256 `75d9fd696d3c17a0daf1f47811b8b33de9734c6bbcfccbfb14c2fd4d6f005cb8`; its members included `libVIPhal.so`, `libNBGlinker.so`, and `vip_lite.h`.
- **Prior local observation —** `/home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf` was 3,803,452,480 bytes with SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- **Unknown/contradictory —** the `Bonsai-27B-Q1_0.gguf` filename does not by itself prove the exact upstream variant, tensor contents, quantization semantics, or suitability for a target run.
- **Unknown/contradictory —** no supplied prior record pins the backup artifacts to the current target's device-tree identity, driver build, or kernel ABI.
- **Hypothesis —** the Mobilenet workspace and backup pair can become a small conversion/runtime smoke fixture after a fresh hash-pinned CPU reference and target-side quality check are defined; this is not evidence that the old artifacts execute correctly.

## Confidence

**Prior local observation — high** for the existence, names, sizes, and hashes recorded above.  
**Unknown/contradictory — high** for causality, target compatibility, performance, and correctness because no qualifying run record is present.

## Unresolved

- **Unknown/contradictory —** the exact command line, environment, and memory pressure preceding the old container stop are not recorded here.
- **Unknown/contradictory —** the Mobilenet conversion options and compiler target are not pinned in a reproducible result.
- **Unknown/contradictory —** no deterministic output comparison, token agreement, latency, throughput, or thermal trace is attached to these artifacts.

## Next experiment

**Hypothesis —** use a fresh, read-only inventory of the workspace and a pinned CPU reference to define a minimal smoke fixture; then run the target asset under the repository profiler only after confirming board/runtime identity. Record failures as evidence and keep any NBG or weight files outside Git.

## Read-only reproduction commands

```sh
find /home/random/acuity-workspace -type f \( -name 'mobilenetv2-12.onnx' -o -name '*.json' -o -name '*.data' -o -name '*.dat' \) -print | sort
find /home/random/orangepi-backup/npu-sdk/vpm_run -maxdepth 1 -type f -printf '%s %p\n' | sort
sha256sum /home/random/orangepi-backup/npu-sdk/vpm_run/vpm_run /home/random/orangepi-backup/npu-sdk/vpm_run/network_binary.nb /home/random/orangepi-backup/npu-sdk/vpm_run/input_0.dat /home/random/orangepi-backup/npu-sdk/npu-libs.tar.gz
stat -c '%s %n' /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf; sha256sum /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf
docker ps -a --filter ancestor=ubuntu-npu:v2.0.10.2 --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'
container_id="$(docker ps -aq --filter ancestor=ubuntu-npu:v2.0.10.2 | head -n 1)"; test -z "$container_id" || docker inspect --format '{{.State.OOMKilled}} {{.State.Status}} {{.Config.Image}} {{range .Mounts}}{{.Source}} -> {{.Destination}} {{end}}' "$container_id"
```
