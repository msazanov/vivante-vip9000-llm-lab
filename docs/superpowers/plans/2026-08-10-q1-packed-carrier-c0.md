# Q1 Packed Carrier C0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать воспроизводимый открытый C0-пакет для `Q1_VIP_16x128`: точный pack/unpack, независимый CPU golden, детерминированные fixtures и переносимый OpenCL C 1.2 kernel, готовый к проверке host compiler → NBG → A733.

**Architecture:** Сначала чистые Python-модули стандартной библиотеки фиксируют байтовый layout и математику Q1_0×Q8_0 независимо от vendor SDK. Генератор превращает adversarial fixture в пять хэшированных файлов. OpenCL C0 принимает packed sign bytes и отдельные FP32 scales, чтобы первая compiler/runtime проверка не зависела от `cl_khr_fp16`; реальный FP16-in-payload путь остаётся C1 после успешного C0.

**Tech Stack:** Python 3.10+ standard library, `unittest`, OpenCL C 1.2, Clang syntax validation, существующий repository profiling/thermal contract.

## Global Constraints

- Активный milestone — только binary GGML `Q1_0`, group K=128: `FP16 d` + 16 sign bytes, bit 1 = `+d`, bit 0 = `-d`, LSB-first.
- Physical layout — `Q1_VIP_16x128/v1`: 256 sign bytes, затем 16 little-endian FP16 scales, всего 288 bytes на 2048 weights.
- Q8_0 состоит из четырёх 34-byte subblocks на K=128; у каждого subblock собственный little-endian FP16 scale и 32 signed int8 values.
- Pack/unpack и fixtures должны быть byte-deterministic; ошибки dimensions/length/overwrite fail closed.
- Никакие model weights, vendor headers/libraries, generated NBG, license text, credentials или private SDK payload не попадают в Git.
- C0 kernel использует standard OpenCL C 1.2 byte buffers и отдельные FP32 scale planes; запрещены `half`, fast-math, subgroup, images и vendor extensions.
- C0 не считается NPU success: только target NBG execution с CPU golden, нулевым CPU fallback, device counters и сохранёнными hashes может повысить статус.
- Expanded INT8 weights в DDR запрещены; C0 kernel читает packed sign bytes напрямую.
- Документация и human-facing CLI errors — на русском; имена API и форматов остаются техническими.
- Каждый production behavior реализуется только после наблюдаемого RED test; каждый green cycle заканчивается отдельным commit.

---

### Task 1: Exact `Q1_VIP_16x128` pack/unpack

**Files:**
- Create: `tooling/q1_vip_layout.py`
- Create: `tests/test_q1_vip_layout.py`

**Interfaces:**
- Consumes: canonical row-major Q1_0 bytes: for every output row and K=128 block, `d[2]` followed by `qs[16]`.
- Produces: `canonical_nbytes(ne0: int, ne1: int) -> int`, `packed_nbytes(ne0: int, ne1: int) -> int`, `pack_tensor(source: bytes, ne0: int, ne1: int) -> bytes`, `unpack_tensor(packed: bytes, ne0: int, ne1: int) -> bytes`, and `Q1VipLayoutError(ValueError)`.

- [ ] **Step 1: Write the failing size/one-tile tests**

```python
from tooling.q1_vip_layout import canonical_nbytes, pack_tensor, packed_nbytes

def test_production_shapes_preserve_q1_byte_count(self):
    self.assertEqual(canonical_nbytes(5120, 17408), 12_533_760)
    self.assertEqual(packed_nbytes(5120, 17408), 12_533_760)
    self.assertEqual(canonical_nbytes(17408, 5120), 12_533_760)
    self.assertEqual(packed_nbytes(17408, 5120), 12_533_760)

def test_pack_one_tile_writes_all_signs_before_scales(self):
    blocks = []
    signs = []
    scales = []
    for row in range(16):
        scale = bytes((row, 255 - row))
        sign = bytes((row * 16 + i) & 255 for i in range(16))
        blocks.append(scale + sign)
        signs.append(sign)
        scales.append(scale)
    self.assertEqual(pack_tensor(b"".join(blocks), 128, 16),
                     b"".join(signs + scales))
```

- [ ] **Step 2: Run RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_layout -v`

Expected: import failure because `tooling.q1_vip_layout` does not exist.

- [ ] **Step 3: Implement shape and one-tile pack**

```python
QK = 128
M_TILE = 16
Q1_BLOCK_BYTES = 18
TILE_BYTES = 288
MAX_U64 = (1 << 64) - 1

class Q1VipLayoutError(ValueError):
    pass

def _shape(ne0: int, ne1: int) -> tuple[int, int]:
    if type(ne0) is not int or type(ne1) is not int:
        raise Q1VipLayoutError("ne0 и ne1 должны быть целыми")
    if ne0 <= 0 or ne1 <= 0 or ne0 % QK or ne1 % M_TILE:
        raise Q1VipLayoutError("форма должна иметь ne0 % 128 == 0 и ne1 % 16 == 0")
    if ne0 > MAX_U64 or ne1 > MAX_U64:
        raise Q1VipLayoutError("форма выходит за uint64")
    return ne0 // QK, ne1 // M_TILE
```

Pack loops must use source offset `(row * k_blocks + k_block) * 18` and destination order `m_tile`, `k_block`, 16 sign rows, 16 scales.

- [ ] **Step 4: Run GREEN for the first behavior**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_layout -v`

Expected: both initial tests pass.

- [ ] **Step 5: Add RED tests for round-trip and fail-closed validation**

Add literal tests for K=256/M=32 order, `unpack_tensor(pack_tensor(x)) == x`, zero/non-multiple/bool dimensions, truncated/oversized input, and multiplication beyond `MAX_U64`. Every exception must leave caller-owned source unchanged because functions return new immutable `bytes`.

- [ ] **Step 6: Run RED, implement unpack/validation, then run GREEN**

Run before and after implementation: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_layout -v`

Expected RED: missing `unpack_tensor` or unhandled invalid input. Expected GREEN: all layout tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add tooling/q1_vip_layout.py tests/test_q1_vip_layout.py
git commit -m "feat: add exact Q1 VIP tile layout"
```

---

### Task 2: Independent Q1_0×Q8_0 CPU golden

**Files:**
- Create: `tooling/q1_vip_golden.py`
- Create: `tests/test_q1_vip_golden.py`

**Interfaces:**
- Consumes: one canonical 18-byte Q1 row block or one 288-byte packed tile; one canonical 136-byte Q8_0 K=128 vector.
- Produces: `dot_canonical_block(q1: bytes, q8: bytes) -> float`, `dot_packed_tile(tile: bytes, q8: bytes) -> tuple[float, ...]`, `Q1VipGoldenError(ValueError)`.

- [ ] **Step 1: Write failing literal golden tests**

```python
def q8_block(scale_bits: bytes, values: list[int]) -> bytes:
    return scale_bits + bytes(v & 255 for v in values)

def test_four_q8_scales_are_applied_independently(self):
    q1 = bytes.fromhex("003c") + b"\xff" * 16
    q8 = b"".join([
        q8_block(bytes.fromhex("003c"), [1] * 32),
        q8_block(bytes.fromhex("0040"), [1] * 32),
        q8_block(bytes.fromhex("0038"), [1] * 32),
        q8_block(bytes.fromhex("00bc"), [1] * 32),
    ])
    self.assertEqual(dot_canonical_block(q1, q8), 80.0)
```

Also add hand-derived cases: all plus `128`, all minus `-128`, alternating `0`, Q1 scale 2 gives `160`, and endpoint Q8 blocks `[-128]`, `[127]`, `[0]`, alternating `[127,-128]` give `-48` with all scales 1.

- [ ] **Step 2: Run RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_golden -v`

Expected: import failure because `tooling.q1_vip_golden` does not exist.

- [ ] **Step 3: Implement the scalar reference**

Use only `struct.unpack('<e', raw)[0]` for FP16 and explicit signed conversion `raw if raw < 128 else raw - 256`. For each Q8 subblock, accumulate integer signed dot first, then `float(q1_scale) * float(q8_scale) * integer_dot` into Python float in subblock order 0..3. Reject non-exact byte lengths.

- [ ] **Step 4: Run GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_golden -v`

Expected: all literal golden tests pass.

- [ ] **Step 5: Add RED packed-tile equivalence tests**

Build 16 canonical rows with all-plus, all-minus, `0x55`, `0xaa`, and bit-boundary `01 02 04 08 10 20 40 80` patterns, pack them with Task 1, and compare every `dot_packed_tile` output with its independently computed `dot_canonical_block` output. Add NaN/Inf FP16 scale rejection.

- [ ] **Step 6: Implement packed-tile dot and finite checks, run GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_golden tests.test_q1_vip_layout -v`

Expected: all golden and layout tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add tooling/q1_vip_golden.py tests/test_q1_vip_golden.py
git commit -m "feat: add Q1 Q8 packed tile golden"
```

---

### Task 3: Deterministic public C0 fixture generator

**Files:**
- Create: `tooling/generate_q1_vip_fixture.py`
- Create: `tests/test_generate_q1_vip_fixture.py`
- Modify: `tooling/README.md`

**Interfaces:**
- Consumes: CLI `--output-dir DIR`; no model or vendor input.
- Produces exclusively-created `q1_canonical.bin`, `q1_vip.bin`, `q8.bin`, `expected_f32.bin`, `manifest.json` and exits 0. Manifest schema is `q1-vip-c0-fixture/v1` with byte sizes/SHA-256, shape K=128/M=16, layout/type identities, expected output count, generator repository commit, and sorted keys.

- [ ] **Step 1: Write the failing CLI test**

The test runs the real script twice into two temporary directories, asserts byte-identical file sets, checks exact sizes `288`, `288`, `136`, `64`, verifies every literal SHA-256 from file bytes, and asserts `manifest.json` is UTF-8 canonical JSON with sorted keys, LF, and one trailing newline.

- [ ] **Step 2: Run RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_generate_q1_vip_fixture -v`

Expected: script path does not exist.

- [ ] **Step 3: Implement minimal deterministic generator**

Use fixed row patterns cycling `ff`, `00`, `55`, `aa`, and bit-boundary bytes. Use Q1 scales cycling raw FP16 `1.0`, `2.0`, `0.5`, `-1.0`; Q8 values include `-128`, `127`, zero, and alternating extremes with raw scales `1.0`, `2.0`, `0.5`, `-1.0`. Compute expected outputs only through `dot_packed_tile`, serialize with `struct.pack('<16f', *outputs)`, and use exclusive file creation.

- [ ] **Step 4: Run GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_generate_q1_vip_fixture -v`

Expected: deterministic fixture test passes.

- [ ] **Step 5: Add RED failure-path tests**

Assert missing `--output-dir`, an existing output directory, a symlink output path, and a partially existing target all fail nonzero without overwriting any existing byte.

- [ ] **Step 6: Implement fail-closed CLI and document the exact command**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_generate_q1_vip_fixture -v`

Document: `python3 tooling/generate_q1_vip_fixture.py --output-dir /tmp/q1-vip-c0-fixture` and state that generated fixture files are public synthetic data, while generated NBG remains outside Git.

- [ ] **Step 7: Commit Task 3**

```bash
git add tooling/generate_q1_vip_fixture.py tests/test_generate_q1_vip_fixture.py tooling/README.md
git commit -m "feat: generate deterministic Q1 NBG fixtures"
```

---

### Task 4: Portable OpenCL C0 packed-carrier kernel

**Files:**
- Create: `experiments/E003-q1-packed-carrier/q1_vip_16x128_c0.cl`
- Create: `experiments/E003-q1-packed-carrier/README.md`
- Create: `tests/test_q1_vip_opencl.py`
- Modify: `README.md`

**Interfaces:**
- Consumes five buffers: `packed_signs[256]` UINT8, `q1_scales[16]` FP32, `q8_values[128]` UINT8 carrying signed int8 bit patterns, `q8_scales[4]` FP32, output `float[16]`.
- Produces kernel symbol `q1_vip_16x128_c0`; global size exactly 16; one work-item computes one output row.

- [ ] **Step 1: Write a failing real compiler test**

```python
completed = subprocess.run([
    clang, "-x", "cl", "-cl-std=CL1.2", "-Werror", "-fsyntax-only", kernel
], text=True, capture_output=True)
self.assertEqual(completed.returncode, 0, completed.stderr)
```

The test skips only when `clang` is absent and otherwise fails because the kernel file is absent.

- [ ] **Step 2: Run RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_opencl -v`

Expected: failure naming missing `q1_vip_16x128_c0.cl`.

- [ ] **Step 3: Implement minimal OpenCL C 1.2 kernel**

```c
__kernel void q1_vip_16x128_c0(
    __global const uchar *packed_signs,
    __global const float *q1_scales,
    __global const uchar *q8_values,
    __global const float *q8_scales,
    __global float *output) {
    const size_t row = get_global_id(0);
    if (row >= 16) return;
    float acc = 0.0f;
    for (uint block = 0; block < 4; ++block) {
        int signed_dot = 0;
        for (uint j = 0; j < 32; ++j) {
            const uint k = block * 32 + j;
            const uchar sign_byte = packed_signs[row * 16 + k / 8];
            const int sign = ((sign_byte >> (k & 7)) & 1) ? 1 : -1;
            const uint raw = q8_values[k];
            const int value = raw <= 127 ? (int)raw : (int)raw - 256;
            signed_dot += sign * value;
        }
        acc += q8_scales[block] * (float)signed_dot;
    }
    output[row] = q1_scales[row] * acc;
}
```

- [ ] **Step 4: Run GREEN and reject forbidden dependencies**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_opencl -v`

Expected: Clang syntax succeeds. The test also preprocesses with `clang -E` and requires no extension enable; it must not assert source-text spelling.

- [ ] **Step 5: Write the Russian experiment contract**

Document exact input conversion from the Task 3 fixture, global size 16, no local-size assumption, CPU golden comparison, required compiler/image/NBG/runtime hashes, integer VIPLite statuses, target profiling phases, 50 resident repetitions after warmup, zero CPU fallback, and the fact that C0 FP32 scale planes are a capability probe rather than the production FP16 layout.

- [ ] **Step 6: Run the complete local gate**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`

Expected: previous 120 tests plus all new layout/golden/fixture/OpenCL tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/E003-q1-packed-carrier README.md tests/test_q1_vip_opencl.py
git commit -m "experiments: add portable packed Q1 C0 kernel"
```

---

## Plan self-review

- Spec coverage: this plan implements C1 host fixtures and the narrow portable kernel needed by compiler/runtime Gate C0; sidecar manifest/model conversion, production FP16 payload kernel, multi-tile autotune, native NN/SRAM path and ggml backend remain later plans gated by target C0 evidence.
- Placeholder scan: no `TBD`, `TODO`, unspecified error handling or unnamed test step remains.
- Type consistency: Task 2 consumes Task 1 packed bytes; Task 3 serializes Task 2 outputs; Task 4 consumes the same logical planes but intentionally separates FP32 scales for the first compiler probe.
- Evidence boundary: local Clang success proves only OpenCL C syntax. It cannot be reported as Vivante compilation, NBG compatibility, NPU execution or performance.
