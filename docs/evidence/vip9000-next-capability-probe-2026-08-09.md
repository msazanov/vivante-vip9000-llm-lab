# VIP9000 next capability probe — 2026-08-09

## Purpose and evidence boundary

This document turns already recovered local Vivante/VIPLite assets into a
repeatable next experiment. It does not claim that a host compiler, datatype,
operator, or NBG is compatible with the A733 until the target executes it and a
numerical fixture passes. Proprietary binaries, headers, models, and NBG files
remain external to Git; only identities, API names, and sanitized observations
are recorded here.

## Recovered assets

| Asset | Size | SHA-256 / identity |
|---|---:|---|
| SDK backup `npu-libs.tar.gz` | external | `75d9fd696d3c17a0daf1f47811b8b33de9734c6bbcfccbfb14c2fd4d6f005cb8` |
| `libNBGlinker.so` | 174,560 B | external proprietary library |
| `libVIPhal.so` | 39,248 B | external proprietary library |
| `vip_lite.h` | 44,351 B | external proprietary header |
| `vpm_run` | AArch64 executable | `dbcea3266070d51a4e0b3aac19540814d065f3727a6045ee96e25f9464371407` |
| `network_binary.nb` | external NBG | `fc0d50bdc863e4dfc74eef6dad9d34655c560e759a2f426979a01c505fa9b95f` |
| `input_0.dat` | 150,528 B | `d38c22c2007db314572c9ddfeb513a9f3d11ec99e8deec0f65bc91481502f1b9` |

The NBG string table identifies `ShuffleNetV2_uint8_NCHW` and operations
including `ConvolutionReluPoolingLayer2`, `TensorTranspose`, `TensorCopy`, and
`ConcatLayer`. The input size equals `224 * 224 * 3` bytes and is consistent
with a UINT8 image input. There is no pinned golden output, so a successful
first execution proves runtime compatibility only, not correctness.

The external `vpm_run` binary imports the following VIPLite surface:

```text
vip_init                 vip_destroy
vip_get_version          vip_query_hardware
vip_create_network       vip_query_network
vip_prepare_network      vip_set_network
vip_query_input          vip_query_output
vip_create_buffer        vip_map_buffer
vip_flush_buffer         vip_set_input
vip_set_output           vip_run_network
vip_get_buffer_size      vip_unmap_buffer
vip_destroy_buffer       vip_finish_network
```

Observed diagnostic options include `-s`, `-l`, `-d`, `-t`, `-b`, `-c`,
`--layer_profile_dump`, `--preload`, and `--op_segment`. These are interface
evidence, not permission to treat layer profiling as steady-state throughput.

## GPU and NPU are separate engines

The PowerVR BXM-4-64 GPU uses the DRM/Vulkan stack. The VIP9000-class NPU uses
`/dev/vipcore`, VIPLite, NBG linker/HAL libraries, and its own NPU devfreq
node. Kernel configuration evidence distinguishes them:

```text
CONFIG_AW_NNA_VIP=m
CONFIG_AW_NNA_GALCORE is disabled
CONFIG_AW_GPU_TYPE="bxm"
```

Therefore a VIPLite workload is neither CPU emulation nor a Vulkan GPU
workload. GPU and NPU still contend for shared UMA/DDR bandwidth, so all
offload conclusions must include host-device synchronization and byte counts.

## Target-verified identity and first execution

Read-only inspection on the target confirmed:

- `/dev/vipcore`, module `vipcore` version 1.13.0, and NPU devfreq bound to
  `/sys/bus/platform/drivers/vipcore`;
- NPU frequencies 492/852/1008 MHz with `performance` governor;
- `/dev/dri/renderD128` bound separately to `pvrsrvkm`, confirming that the
  PowerVR GPU and VIPLite NPU use different drivers;
- target `/usr/lib/libNBGlinker.so` SHA-256
  `de93bb4a7d86af67bc4ca597d12228b9b21b062601b18f33ca30393a2af4a842`
  and `/usr/lib/libVIPhal.so` SHA-256
  `648444a26a1aeaec0182c0ba348b7a453134b8ccdf8282a211ce4c0f5e5e51a0`.

The first guarded execution used the exact recovered ShuffleNetV2 UINT8 NBG,
one loop, device 0, core 0, no preload, no NPD/layer dump, 30-second internal
and external timeouts, and 10 ms profiler/guard intervals. It returned 0 and
reported VIPLite driver ABI `0x00020003`, software
`2.0.3.2-AW-2024-08-30`, CID `0x1000003b`, one device, and one runtime-logical
core. The logical core count is an API observation and must not be silently
equated with or used to refute the lower-level NN-engine count in product
descriptions.

| Run | Create µs | Prepare µs | Input read µs | Host run µs | Device µs / cycles | Process elapsed ms (scope) | Correctness | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `vip9000-vpm-shufflenet-uint8-single-001` | 852 | 408 | 8332 | 3212 | 2964 / 2,954,514 | 239.393 (outer profiler) | no golden/output | execution-compatible, unqualified |
| `vip9000-vpm-shufflenet-uint8-output-001` | 859 | 411 | 221 | 3086 | 2863 / 2,829,953 | 178.938 (outer profiler) | two values saved; no golden | output-captured, unqualified |
| `vip9000-vpm-shufflenet-uint8-resident100-001` | — | — | — | — | — | — | not run | failed before launch: profiler interval below 10 ms |
| `vip9000-vpm-shufflenet-uint8-resident100-002` | 839 | 413 | 218 | 2846 median | 2803 / 2,817,775 median | 303.034 (child process) | output bypassed; no golden | performance-observed, unqualified |

The run retained complete profiler lifecycle evidence, 17 telemetry samples,
five thermal-guard samples, matching target/local raw hashes, 40.052 °C peak
CPU temperature, 37.324 °C peak NPU temperature, fan state 4, zero swap delta,
and no kernel fault match. The exact journal interval contained one normal NPU
event setting core 0 to 1008 MHz. The fixture did not save an output and has no
golden, so the timings prove ABI/NBG execution compatibility only. The compact
record is
[`benchmarks/results/vip9000-vpm-shufflenet-uint8-single-001/summary.json`](../../benchmarks/results/vip9000-vpm-shufflenet-uint8-single-001/summary.json).

The output-capture run saved two text values, `-0.2080624550580978` and
`0.2064369618892670`, with SHA-256
`77bea7d9f2ac874cac7d1d27f8c19d321c610d5a460716c6fd3fe09ba33959bb`.
The built-in top-5 printer emitted three sentinel `-1` rows because the output
has only two elements; those rows are not present in the saved tensor.
The compact output-capture record is
[`benchmarks/results/vip9000-vpm-shufflenet-uint8-output-001/summary.json`](../../benchmarks/results/vip9000-vpm-shufflenet-uint8-output-001/summary.json).

The resident test kept the network and buffers alive for 100 loops. Treating
loop 1 as warm-up gives 99 measured loops: median host run was 2,846 us
(351.37 inference/s), median device profile was 2,803 us (356.76 inference/s),
and the median host API/synchronization difference was 42 us, about 1.50% of
device time. The complete child process took 303.034 ms for all 100 loops,
equivalent to 330.00 inference/s including initialization and teardown. The
outer profiled lifecycle was 472.899 ms; it includes profiler startup/teardown
and must not be reported as NPU inference latency.

`vpm_run` printed a final “profile avg” of 2,875 us, which equals its first-loop
value and disagrees with the exact mean of the 100 individual rows
(2,807.57 us). Canonical statistics therefore parse individual rows rather
than trusting that final aggregate. The compact resident record is
[`benchmarks/results/vip9000-vpm-shufflenet-uint8-resident100-002/summary.json`](../../benchmarks/results/vip9000-vpm-shufflenet-uint8-resident100-002/summary.json).

## Ordered target probe

### 1. Read-only identity

Record presence, ownership, driver binding, and frequency nodes without
changing policy:

```text
/dev/vipcore
/sys/class/devfreq/*npu*
/sys/class/devfreq/*gpu*
/dev/dri/renderD*
```

Also retain device-tree compatibility, loaded-module identities, and available
driver-bind journal lines. This distinguishes a present character device from
a working userspace ABI.

### 2. Minimal VIPLite inventory helper

Link only against the recovered target runtime and call, in order:

1. `vip_get_version()`;
2. `vip_init()`;
3. `vip_query_hardware()` for CID, device count, and core count per device;
4. `vip_destroy()`.

Record integer status after every call. A header enum is not a hardware
capability claim.

### 3. NBG metadata without execution

Run this as a self-contained lifecycle: `vip_init`, create the known network
from file, query network name, layer count, input/output counts, memory-pool
size, and core count, destroy the network, then `vip_destroy`. Query each
tensor's dimensions, layout/format, quantization, scale, zero point, and name.
Retain the integer status of every call and use `available`, `unavailable`, or
`not-applicable` for unsupported properties rather than turning missing data
into zero. Destroy every successfully created object on every failure path.

### 4. One guarded execution

Prepare the known network, allocate resident input/output buffers, map and copy
the exact input, flush CPU writes, attach buffers, run once, invalidate/read
the output, and retain output byte size plus SHA-256. This run is wrapped by the
standard profiler and 85 °C guard. Without a golden output it remains
execution evidence only.

Before execution, pin the target `vip_lite.h`, runtime-library, NBG, executable,
and input hashes plus the exact `LD_LIBRARY_PATH`. Record the ordered API call
sequence and every return code. The process wrapper must impose a timeout and
clean up the whole child process group. The baseline must not enable preload,
NPD, layer profiling, or operator segmentation. Those are separate A/B tests.

### 5. Steady-state timing

After one warmup, execute 30–100 repetitions with the network and buffers kept
resident. Compare synchronous `vip_run_network()` with
`vip_trigger_network()` plus `vip_wait_network()` only if both are exposed by
the target header and runtime exports. After each repetition query
`VIP_NETWORK_PROP_PROFILING`; retain `inference_time` in microseconds and
`total_cycle`, along with host monotonic time, only after the exact target ABI
accepts that property.

Initialization, NBG load/link, prepare, first-run warmup, input map/write,
CPU-to-NPU flush, NPU execution, wait/synchronization, output invalidation/read,
and teardown are separate phases. Never publish only the device execution
counter as end-to-end inference speed.

### 6. Profiling overhead isolation

Enable `VIP_NETWORK_PROP_SET_ENABLE_NPD` before prepare for one diagnostic run,
or use `vpm_run --layer_profile_dump`. Do not mix NPD/layer-dump observations
into the uninstrumented throughput median; report their overhead separately.
An enum in a recovered header is not proof that the installed runtime supports
the property.

### 7. Memory and copy matrix

Start with `VIP_BUFFER_MEMORY_TYPE_DEFAULT`. Then evaluate host-handle import
and FD/DMA-BUF import only when a compatible allocator is actually present.
For every mode retain map/write, flush, execution, invalidate/read time, and
bytes moved. Test SRAM preload/shared pools/device/core selection only after
the default resident-buffer reference passes.

## Datatype and compiler guardrail

The recovered header contains FP16, INT8, and INT4 enum values. That proves only
that the API names them. Host AcuityLite 6.51.0 markers such as
`GCNANOULTRA31_VIP2_PID0X15`, VXC, or EVIS are compiler-side evidence and do
not prove compatibility with this target runtime. Each datatype/operator needs
a minimal compiled NBG, successful target status, and numerical comparison to
a pinned CPU golden fixture before it becomes a verified fact.

## Per-test result contract

Every NPU test entry and machine-readable result must retain:

- run/model/NBG/input identities, sizes, hashes, exact command, target ABI and
  runtime-library hashes, driver/kernel identity, and requested/effective CPU
  affinity of the workload, profiler, and guard;
- individual statuses for initialization, network creation/query/prepare,
  buffer creation/map/flush/attach, execution/wait, invalidate/read, teardown,
  and cleanup after failure;
- separate timings for process startup, VIP initialization, NBG load/link,
  prepare, buffer allocation/import, input map/write, flush, device execution,
  synchronization, output invalidate/read, and teardown;
- buffer memory type, input/output tensor metadata, bytes mapped/copied/flushed/
  invalidated, output size/hash, device time/cycles, and all host latency
  samples with min/p10/median/p90/max/CV;
- CPU/RSS/memory/swap/frequency/thermal/fan evidence, guard result, exact kernel
  journal interval, and counts of throttle, OOM, VIP/NPU, and GPU faults;
- golden/reference identity plus an appropriate correctness metric and declared
  threshold. Depending on output type this may include exact match, max/mean
  absolute and relative error, cosine similarity, or top-k agreement.

An output hash without a golden comparison establishes repeatability only.
Missing correctness or application TTFT keeps an otherwise successful run
`unqualified`.
