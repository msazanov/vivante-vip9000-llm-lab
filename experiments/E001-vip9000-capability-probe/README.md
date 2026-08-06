# E001 — VIP9000 SDK and operator capability probe

## Objective

Create the first target-verified capability matrix for the exact Orange Pi A733/VIP9000 board and SDK.

This experiment deliberately precedes any `llama.cpp` modification.

## Questions

1. Which runtime/API families are installed: OpenVX, TIM-VX, VIPLite, Acuity-generated examples, or other vendor wrappers?
2. Which exact NPU product ID and runtime versions are active?
3. Can graphs and buffers remain resident across repeated calls?
4. Which LLM-relevant operations, shapes, layouts and data types compile and execute?
5. What are cold-load, warm-run, transfer and synchronization costs?
6. Can host pointers or DMA-BUF memory be imported without a full copy?
7. Are RMSNorm and RoPE reachable through the exact SDK?
8. Does custom OpenCL execute on a useful accelerator path?

## Do not commit

- SDK archives;
- proprietary headers unless redistribution is explicitly allowed;
- runtime `.so` files;
- firmware;
- NBG binaries unless redistribution is confirmed;
- board credentials, public IPs, SSH keys or serial numbers.

Commit version output, hashes, sanitized manifests, source code written by this project, and benchmark results.

## Phase 1 — target fingerprint

Capture:

```bash
uname -a
cat /etc/os-release
lscpu
cat /proc/cpuinfo
free -h
cat /proc/meminfo
lsmod
find /dev -maxdepth 1 -type c -o -type b
find /sys -maxdepth 6 \( -iname '*vip*' -o -iname '*npu*' -o -iname '*galcore*' \) -print
ldconfig -p | grep -Ei 'vip|vsi|openvx|ovx|gal|npu|nbg'
dmesg | grep -Ei 'vip|npu|galcore|vivante'
```

Also capture, where available:

```text
kernel config
loaded module filenames and modinfo
device-tree compatible strings
NPU product/device ID
firmware and runtime version strings
CPU frequency policies
thermal zones
CMA reservation
IOMMU configuration
NPU clock/power domains
```

Output:

```text
artifacts/target-manifest.yaml
artifacts/system.txt
artifacts/library-hashes.txt
```

## Phase 2 — SDK inventory

Create a private inventory first. Produce a sanitized manifest with:

```yaml
sdk:
  source: unknown
  package_name: unknown
  version: unknown
  sha256: unknown
  license_files_present: []
  components:
    acuity: unknown
    pegasus: unknown
    vivante_ide: unknown
    tim_vx: unknown
    openvx: unknown
    viplite: unknown
    nbg_linker: unknown
    vip_hal: unknown
    kernel_driver: unknown
    firmware: unknown
```

For every component, record:

- exact path;
- version command or embedded version string;
- file hash;
- architecture;
- dynamic dependencies;
- public/private status;
- license status;
- whether redistribution is permitted.

## Phase 3 — minimal persistent runtime

Build the smallest runner allowed by the SDK.

Required lifecycle:

1. initialize runtime once;
2. load/create network once;
3. allocate buffers once;
4. prepare/compile once;
5. repeat input update → run → output read at least 100 times;
6. release resources once.

Separate metrics:

```text
process startup
runtime initialization
network load
network prepare/compile
buffer allocation
first run
warm runs
input copy/map/flush
output invalidate/map/copy
teardown
```

## Phase 4 — operation tests

### Shape families

Use small validation shapes first, then LLM-like shapes.

```text
Sanity:
  [1, 16] × [16, 16]
  [8, 64] × [64, 64]

Decode-like:
  [1, H] × [H, H]
  [1, H] × [H, I]
  [1, I] × [I, H]

Prefill-like:
  [T, H] × [H, H]
  [T, H] × [H, I]
  [T, I] × [I, H]
```

Begin with small reference values such as:

```text
H = 256, 512, 1024
I = 4H or architecture-specific
T = 1, 8, 32, 128, 512
```

Only move to the actual Bonsai dimensions after the harness is stable.

### Type families

```text
FP32 × FP32 → FP32
FP16 × FP16 → FP16
INT8 activation × INT8 weight → supported output
per-channel INT8 weights where supported
INT4 only after a real compile/run test proves support
```

### Operation order

1. Add
2. Multiply
3. MatMul
4. Dense/FullyConnected
5. Reshape
6. Transpose
7. Cast/DataConvert
8. Swish/SiLU
9. Softmax
10. LayerNormalization
11. RMSNorm
12. Gather
13. RoPE
14. complete gated MLP
15. fixed-shape attention prefill

## Phase 5 — memory tests

Test independently:

- ordinary runtime-allocated buffers;
- host pointer-backed tensors;
- handle swapping;
- DMA-BUF import;
- repeated buffer reuse;
- constant weight lifetime;
- variable tensor/state lifetime;
- cache flush/invalidate requirements;
- CPU access while NPU is idle;
- concurrent CPU work during NPU execution.

Never assume shared physical memory equals zero-copy. Measure actual memcpy, cache and synchronization costs.

## Correctness protocol

Generate deterministic input tensors and compare every operation to a CPU implementation.

Record:

```text
max_absolute_error
mean_absolute_error
max_relative_error
cosine_similarity
exact_match_rate for integer output
```

Save failing inputs where licensing and size permit.

## Performance protocol

For each configuration:

1. one cold process measurement;
2. at least ten warm-up iterations;
3. at least fifty measured iterations for small graphs or ten for long graphs;
4. median, p10, p90, minimum and maximum;
5. CPU time and utilization;
6. temperatures and frequencies;
7. input/output bytes;
8. power if measurable.

## Result schema

Suggested JSON Lines record:

```json
{
  "experiment": "E001",
  "timestamp": "ISO-8601",
  "target_id": "orange-pi-a733-01",
  "runtime": {
    "name": "viplite",
    "version": "unknown",
    "device_id": "unknown"
  },
  "operation": "matmul",
  "inputs": [
    {"shape": [128, 1024], "dtype": "fp16"},
    {"shape": [1024, 1024], "dtype": "fp16", "constant": true}
  ],
  "output": {"shape": [128, 1024], "dtype": "fp16"},
  "compile_ok": true,
  "run_ok": true,
  "cold_ms": null,
  "warm_ms_median": null,
  "warm_ms_p90": null,
  "input_bytes": null,
  "output_bytes": null,
  "cpu_time_ms": null,
  "max_abs_error": null,
  "cosine_similarity": null,
  "notes": ""
}
```

## Exit criteria

E001 is complete when:

- the exact runtime and hardware identity are captured;
- a persistent NPU network runs reproducibly;
- the LLM-relevant operation matrix has target-verified results;
- MatMul has a prefill/decode shape sweep;
- data-transfer and launch overheads are measured;
- RMSNorm/RoPE accessibility is answered;
- memory-import capabilities are answered;
- the result is sufficient to select or reject a backend architecture.
