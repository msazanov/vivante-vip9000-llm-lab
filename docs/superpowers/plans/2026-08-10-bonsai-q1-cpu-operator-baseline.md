# Bonsai Q1 CPU Operator Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать воспроизводимый machine-readable baseline для реальных `Q1_0 × Q8_0` FFN-проекций Bonsai-27B на A733, который точно учитывает packed-байты, проверяет результат независимым golden и становится общей точкой сравнения для будущего VIP9000/NBG backend.

**Architecture:** Python читает закреплённый GGUF через `GGUFReader`, строит нормализованный инвентарь и извлекает только выбранный Q1-тензор во внешний fixture-каталог. Отдельный C++ runner исполняет настоящий `GGML_OP_MUL_MAT` через pinned PrismML `ggml-cpu` и требует `CPU_REPACK`, а Python summarizer объединяет его resident samples с telemetry существующего профайлера. Operator schema остаётся отдельной от model-level `tokens/s`; полный Bonsai CPU reference `0.726175 ток/с` остаётся главным критерием для будущей интеграции.

**Tech Stack:** Python 3.11 standard library, pinned PrismML `gguf-py`, C++17, PrismML `ggml` CPU backend, CMake, Linux `taskset`, существующие `profile_command.py` и `thermal_exec_guard.py`, `unittest`.

## Global Constraints

- Закреплённая модель: `Bonsai-27B-Q1_0.gguf`, размер `3_803_452_480`, SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Закреплённый runtime: PrismML llama.cpp commit `38c66ad0241da4f9fcce541cda8edc219086cec5`, AArch64 build `armv8.2-a+dotprod`, `GGML_NATIVE=OFF`.
- Главная метрика проекта остаётся full-model single-stream decode `tokens/s`. Этот этап не меняет model runtime и не имеет права заявлять ускорение Bonsai.
- Сравниваются реальные тензоры `blk.0.ffn_gate.weight` (`K=5120`, `M=17408`) и `blk.0.ffn_down.weight` (`K=17408`, `M=5120`). Каждый содержит `12_533_760` packed Q1 bytes.
- Весовая матрица не расширяется в INT8, FP16 или FP32. `expanded_weight_ddr_bytes` обязан быть явным целым числом и равняться нулю для принятого CPU baseline.
- Setup, чтение GGUF и repack выполняются до measured region. После warmup выполняется не менее 50 resident repetitions.
- `declared_*_bytes` описывают размеры буферов, но не выдаются за аппаратно измеренный DDR traffic. RSS, mmap, swap, температуры и частоты сохраняются отдельно.
- CPU runner завершается ошибкой, если `CPU_REPACK` недоступен или `ggml_backend_supports_op()` отвергает граф; молчаливый переход на другой путь запрещён.
- Fixtures, GGUF, извлечённые веса, q8 activation и raw golden остаются вне Git. В Git попадают только схемы, код, хэши, агрегированные измерения и русская документация.
- Operator results не добавляются в `benchmarks/results/model-runs.jsonl` и не смешиваются с `llama-bench` rows.
- Correctness gate: конечные значения, детерминированный повтор, cosine `>= 0.999999`, max absolute error `<= 1e-4 * max(1, max(abs(reference)))`.
- Performance evidence gate: минимум 50 samples, CV `<= 2%`, swap delta `0`, отсутствие throttle/OOM/fault. Этот gate квалифицирует baseline, но не model speedup.

---

## File Map

| Path | Responsibility |
|---|---|
| `tooling/q1_memory_accounting.py` | Нормализовать GGUF tensor inventory и посчитать decode packed-byte budget. |
| `tests/test_q1_memory_accounting.py` | Проверить классификацию, точные суммы и fail-closed validation. |
| `benchmarks/schema/q1-memory-accounting.example.json` | Зафиксировать versioned accounting contract. |
| `benchmarks/workloads/bonsai-27b-q1.json` | Сохранить воспроизводимый инвентарь закреплённого GGUF без самих весов. |
| `tooling/extract_q1_operator_fixture.py` | Извлечь один реальный Q1 payload и детерминированную F32 activation во внешний каталог. |
| `tests/test_extract_q1_operator_fixture.py` | Проверить offsets, hashes, размеры и запрет overwrite. |
| `tooling/q1_vip_golden.py` | Добавить независимый scalar matrix golden для Q1_0×Q8_0. |
| `tests/test_q1_vip_golden.py` | Проверить matrix golden, invalid lengths и non-finite scales. |
| `experiments/E004-q1-cpu-operator-baseline/q1_cpu_operator_runner.cpp` | Построить реальный Prism `GGML_OP_MUL_MAT`, потребовать CPU_REPACK и записать samples JSONL. |
| `experiments/E004-q1-cpu-operator-baseline/CMakeLists.txt` | Собрать runner против точного Prism source/build tree. |
| `experiments/E004-q1-cpu-operator-baseline/README.md` | Описать цель, термины, сборку и hardware matrix по-русски. |
| `tests/test_q1_cpu_operator_runner.py` | Собрать/запустить self-test runner и проверить реальный JSON/output contract. |
| `tooling/summarize_q1_cpu_operator.py` | Проверить raw JSONL, посчитать статистики/correctness и выпустить operator summary. |
| `tests/test_summarize_q1_cpu_operator.py` | Проверить happy path и все qualification failures. |
| `benchmarks/schema/q1-cpu-operator.example.json` | Зафиксировать общий CPU/NPU-compatible operator result contract. |
| `tooling/run_bonsai_q1_cpu_operator.sh` | Оркестрировать guard, affinity, profiler и summarizer на плате. |
| `tests/test_run_bonsai_q1_cpu_operator.py` | Проверить строгие аргументы, core split и отсутствие встроенного пароля. |
| `docs/evidence/bonsai-q1-cpu-operator-baseline-2026-08-10.md` | Объяснить реальные результаты, ограничения и следующий NBG gate. |

## Task 1: Закрепить статический memory-accounting contract

**Files:**

- Create: `tooling/q1_memory_accounting.py`
- Create: `tests/test_q1_memory_accounting.py`
- Create: `benchmarks/schema/q1-memory-accounting.example.json`

- [ ] **Step 1: Write the failing family-accounting test**

```python
import unittest

from tooling.q1_memory_accounting import Q1Tensor, account_decode_stream


def q1(name: str, index: int, ne0: int, ne1: int) -> Q1Tensor:
    return Q1Tensor(
        name=name,
        index=index,
        ggml_type="Q1_0",
        shape=(ne0, ne1),
        source_offset_bytes=index * 64,
        size_bytes=ne0 * ne1 // 128 * 18,
        payload_sha256=f"{index + 1:064x}",
    )


class MemoryAccountingTests(unittest.TestCase):
    def test_classifies_decode_families_and_excludes_embedding(self) -> None:
        tensors = (
            q1("blk.0.ffn_gate.weight", 0, 5120, 17408),
            q1("blk.0.attn_qkv.weight", 1, 5120, 10240),
            q1("blk.3.attn_q.weight", 2, 5120, 12288),
            q1("output.weight", 3, 5120, 248320),
            q1("token_embd.weight", 4, 5120, 248320),
        )
        result = account_decode_stream(tensors)
        self.assertEqual(result["logical_gemv_count"], 4)
        self.assertEqual(result["excluded_embedding_count"], 1)
        self.assertEqual(
            sum(family["q1_bytes_per_token"] for family in result["families"].values()),
            result["decode_q1_bytes_per_token"],
        )
```

- [ ] **Step 2: Prove the test fails before implementation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_memory_accounting -v
```

Expected: `ModuleNotFoundError: No module named 'tooling.q1_memory_accounting'`.

- [ ] **Step 3: Implement the immutable tensor type and strict classifier**

```python
@dataclass(frozen=True)
class Q1Tensor:
    name: str
    index: int
    ggml_type: str
    shape: tuple[int, ...]
    source_offset_bytes: int
    size_bytes: int
    payload_sha256: str


def classify_decode_tensor(name: str) -> str | None:
    if name == "token_embd.weight":
        return None
    if name == "output.weight":
        return "lm_head"
    if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)\.weight", name):
        return "ffn"
    if re.fullmatch(r"blk\.\d+\.(attn_qkv|attn_gate|ssm_(alpha|beta|out))\.weight", name):
        return "recurrent"
    if re.fullmatch(r"blk\.\d+\.attn_(q|k|v|output)\.weight", name):
        return "full_attention"
    raise Q1MemoryAccountingError(f"unclassified Q1 tensor: {name}")
```

Validate unique names/indices, `ggml_type == "Q1_0"`, two-dimensional shapes, `ne0 % 128 == 0`, `ne1 % 16 == 0`, `size_bytes == ne0*ne1//128*18`, 64-hex payload hashes and non-overlapping payload ranges.

- [ ] **Step 4: Add exact pinned-model assertions**

`validate_bonsai_manifest()` must reject any manifest that does not produce all of:

```python
EXPECTED = {
    "q1_tensor_count": 498,
    "q1_payload_bytes": 3_781_877_760,
    "logical_gemv_count": 497,
    "decode_q1_bytes_per_token": 3_603_087_360,
    "families": {
        "ffn": {"logical_gemv_count": 192, "q1_bytes_per_token": 2_406_481_920},
        "recurrent": {"logical_gemv_count": 240, "q1_bytes_per_token": 781_885_440},
        "full_attention": {"logical_gemv_count": 64, "q1_bytes_per_token": 235_929_600},
        "lm_head": {"logical_gemv_count": 1, "q1_bytes_per_token": 178_790_400},
    },
}
```

- [ ] **Step 5: Add and validate the example schema**

Use schema string `q1-memory-accounting/v1`. Keep `declared_decode_q1_bytes_per_token` separate from `observed_ddr_bytes_per_token`, whose example value is `null`.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_memory_accounting -v
```

Expected: all tests `OK`.

- [ ] **Step 6: Commit Task 1**

```bash
git add tooling/q1_memory_accounting.py tests/test_q1_memory_accounting.py benchmarks/schema/q1-memory-accounting.example.json
git commit -m "feat: define Bonsai Q1 memory accounting"
```

## Task 2: Читать реальные payload offsets и создать закреплённый workload manifest

**Files:**

- Modify: `tooling/q1_memory_accounting.py`
- Modify: `tests/test_q1_memory_accounting.py`
- Create: `benchmarks/workloads/bonsai-27b-q1.json`
- Modify: `tooling/README.md`

- [ ] **Step 1: Write the failing reader-adapter test**

Create fake `ReaderTensor` objects exposing `name`, `tensor_type.name`, `shape`, `n_bytes`, `data_offset`, and byte-like `data`. Assert that `inventory_from_reader()` uses `tensor.data_offset`, not `tensor.field.offset`, and hashes exactly `tensor.n_bytes` payload bytes.

- [ ] **Step 2: Run the focused red test**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_memory_accounting.MemoryAccountingTests.test_reader_uses_payload_offset -v
```

Expected: missing `inventory_from_reader` import or attribute failure.

- [ ] **Step 3: Implement lazy GGUFReader integration**

The CLI must import `gguf` only inside `load_gguf_reader()` so unit tests do not require NumPy/gguf-py:

```python
def load_gguf_reader(model: Path, gguf_python_root: Path):
    sys.path.insert(0, str(gguf_python_root))
    from gguf import GGUFReader
    return GGUFReader(model, "r")
```

CLI:

```text
q1_memory_accounting.py
  --model MODEL.gguf
  --gguf-python-root /path/to/llama-prismml/gguf-py
  --output benchmarks/workloads/bonsai-27b-q1.json
```

Before reading tensors, stream SHA-256 and check the exact file size/hash. Emit all Q1 tensors sorted by `index`, plus the F32/Q1 counts and architecture metadata used by the approved design. Record `source_offset_bytes`, `size_bytes`, `payload_sha256`, `ne0`, `ne1`, `tile_m=16`, `tile_k=128`, `tile_bytes=288`, and deterministic future `sidecar_offset_bytes` aligned to 64 bytes. Do not claim a sidecar payload hash before the sidecar exists.

- [ ] **Step 4: Generate the real manifest**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/q1_memory_accounting.py \
  --model /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf \
  --gguf-python-root /home/random/src/llama-prismml/gguf-py \
  --output benchmarks/workloads/bonsai-27b-q1.json
```

Expected terminal summary:

```text
model_sha256=17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0
q1_tensor_count=498
logical_gemv_count=497
decode_q1_bytes_per_token=3603087360
```

- [ ] **Step 5: Re-read the manifest through the strict validator**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/q1_memory_accounting.py \
  --check benchmarks/workloads/bonsai-27b-q1.json
```

Expected: `PASS q1-memory-accounting/v1`.

- [ ] **Step 6: Commit Task 2**

```bash
git add tooling/q1_memory_accounting.py tests/test_q1_memory_accounting.py benchmarks/workloads/bonsai-27b-q1.json tooling/README.md
git commit -m "data: pin Bonsai Q1 workload manifest"
```

## Task 3: Извлечь external operator fixtures без копирования модели в Git

**Files:**

- Create: `tooling/extract_q1_operator_fixture.py`
- Create: `tests/test_extract_q1_operator_fixture.py`

- [ ] **Step 1: Write failing extraction tests**

Tests must use a small temporary binary and an injected manifest. Cover exact `pread` slice, SHA mismatch, wrong shape/type, truncated file, path already exists and an output path inside the repository.

```python
with self.assertRaises(FixtureError):
    extract_tensor(model, tensor, output_dir, expected_model_sha256="0" * 64)
```

- [ ] **Step 2: Run the red test**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_extract_q1_operator_fixture -v
```

Expected: module import failure.

- [ ] **Step 3: Implement strict extraction and deterministic activation**

Interface:

```python
def extract_tensor(
    model: Path,
    tensor: Mapping[str, object],
    output_dir: Path,
    *,
    expected_model_sha256: str,
    seed: int = 0x733,
) -> dict[str, object]: ...
```

Files written with exclusive creation:

```text
weights.q1_0.bin
activation.f32.bin
fixture.json
```

Generate F32 activation with a code-owned integer PRNG and a defined transform into `[-1, 1]`; do not depend on Python `random` implementation details. Manifest schema is `q1-cpu-operator-fixture/v1` and includes model/tensor/activation hashes and sizes. Reject output directories under the repository root so raw model-derived payload cannot be staged accidentally.

- [ ] **Step 4: Extract both real fixtures on the target or host model store**

```bash
python3 tooling/extract_q1_operator_fixture.py \
  --model /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf \
  --workload benchmarks/workloads/bonsai-27b-q1.json \
  --tensor blk.0.ffn_gate.weight \
  --output-dir /tmp/bonsai-q1-fixtures/blk0-ffn-gate

python3 tooling/extract_q1_operator_fixture.py \
  --model /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf \
  --workload benchmarks/workloads/bonsai-27b-q1.json \
  --tensor blk.0.ffn_down.weight \
  --output-dir /tmp/bonsai-q1-fixtures/blk0-ffn-down
```

Expected sizes:

| Tensor | Q1 bytes | F32 activation bytes |
|---|---:|---:|
| gate `K=5120, M=17408` | 12,533,760 | 20,480 |
| down `K=17408, M=5120` | 12,533,760 | 69,632 |

- [ ] **Step 5: Commit Task 3**

```bash
git add tooling/extract_q1_operator_fixture.py tests/test_extract_q1_operator_fixture.py
git commit -m "feat: extract external Bonsai Q1 fixtures"
```

## Task 4: Расширить независимый golden до полной матрицы

**Files:**

- Modify: `tooling/q1_vip_golden.py`
- Modify: `tests/test_q1_vip_golden.py`

- [ ] **Step 1: Add failing 2-row matrix tests**

Use two adversarial canonical rows with distinct FP16 scales and one Q8 vector. Compare `dot_canonical_matrix()` against two explicit `dot_canonical_block()` accumulation loops. Add failures for short payloads, `ne0 % 128 != 0`, `ne1 <= 0`, non-finite FP16 scales and extra bytes.

- [ ] **Step 2: Run the focused red test**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_golden.Q1VipGoldenTests.test_dot_canonical_matrix -v
```

Expected: missing function failure.

- [ ] **Step 3: Implement a streaming matrix reference**

```python
def dot_canonical_matrix(q1: bytes, q8: bytes, ne0: int, ne1: int) -> tuple[float, ...]:
    blocks_per_row = ne0 // Q1_VALUES
    require_exact_lengths(q1, q8, ne0, ne1)
    output = []
    for row in range(ne1):
        total = 0.0
        for block in range(blocks_per_row):
            q1_off = (row * blocks_per_row + block) * Q1_BLOCK_BYTES
            q8_off = block * Q8_VECTOR_BYTES
            total += dot_canonical_block(
                q1[q1_off:q1_off + Q1_BLOCK_BYTES],
                q8[q8_off:q8_off + Q8_VECTOR_BYTES],
            )
        output.append(total)
    return tuple(output)
```

The production 89M-weight golden may take minutes on the target; it runs once outside the timed region. Never use this Python path as a speed baseline.

- [ ] **Step 4: Add an exclusive-write golden CLI**

```text
q1_vip_golden.py
  --weights weights.q1_0.bin
  --q8 activation.q8_0.bin
  --ne0 K
  --ne1 M
  --output-f32 expected.f32.bin
```

The CLI calls `dot_canonical_matrix()`, writes little-endian F32 values with exclusive creation, and prints the output SHA-256. It refuses a pre-existing output and any non-exact input length.

- [ ] **Step 5: Run the full golden suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_vip_golden -v
```

Expected: all tests `OK`.

- [ ] **Step 6: Commit Task 4**

```bash
git add tooling/q1_vip_golden.py tests/test_q1_vip_golden.py
git commit -m "feat: add full Q1 matrix golden"
```

## Task 5: Зафиксировать общий operator result schema и summarizer

**Files:**

- Create: `tooling/summarize_q1_cpu_operator.py`
- Create: `tests/test_summarize_q1_cpu_operator.py`
- Create: `benchmarks/schema/q1-cpu-operator.example.json`

- [ ] **Step 1: Write failing summary tests**

The happy-path fixture contains one warmup and 50 measured records. Assert exact percentile interpolation, median, min/max, population CV, hashes, byte counters and qualification. Add separate failing cases for 49 samples, duplicate indices, non-finite timing, CV above 2%, wrong tensor hash, missing `CPU_REPACK`, nonzero expanded bytes, output mismatch, swap delta, and thermal guard failure.

- [ ] **Step 2: Run the red tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
```

Expected: module import failure.

- [ ] **Step 3: Implement schema `q1-cpu-operator-baseline/v1`**

Required top-level sections:

```json
{
  "schema_version": "q1-cpu-operator-baseline/v1",
  "run_id": "bonsai-q1-gate-a55-001",
  "model": {},
  "tensor": {},
  "layout": {},
  "workload": {},
  "executor": {},
  "memory": {},
  "execution": {},
  "timing": {},
  "correctness": {},
  "telemetry": {},
  "qualification": {}
}
```

`memory` must include all of:

```json
{
  "declared_canonical_q1_bytes": 12533760,
  "declared_cpu_repack_bytes": 12533760,
  "expanded_weight_ddr_bytes": 0,
  "activation_f32_bytes": 20480,
  "activation_q8_bytes": 5440,
  "output_f32_bytes": 69632,
  "observed_ddr_read_bytes": null,
  "observed_ddr_write_bytes": null,
  "physical_resident_weight_copies": 1
}
```

`layout` keeps logical and physical representations separate:

```json
{
  "logical_type": "GGML_TYPE_Q1_0",
  "source_layout": "GGUF_Q1_0/v1",
  "executor_layout": "CPU_REPACK_Q1_0_4x4"
}
```

The later NPU result will keep the same logical type and tensor identity but set `executor_layout` to `Q1_VIP_16x128/v1`; the schema must never label a CPU_REPACK buffer as a VIP tile. `null` observed DDR counters mean “not measured”, never zero. `execution` includes `logical_gemv_count=1`, `cpu_fallback_count=0`, `strict_mode=true`, `backend="ggml-cpu"`, `kernel="q1_0_4x4_q8_0"`, and `weight_buffer_type="CPU_REPACK"`.

- [ ] **Step 4: Implement correctness before performance qualification**

Read the runner output and independent golden F32 file. Reject non-finite elements. Compute cosine and max absolute error, then evaluate correctness before CV or speed. Store both raw thresholds and boolean decisions.

- [ ] **Step 5: Run tests and validate the example**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_summarize_q1_cpu_operator -v
python3 tooling/summarize_q1_cpu_operator.py --check benchmarks/schema/q1-cpu-operator.example.json
```

Expected: tests `OK`; schema check prints `PASS q1-cpu-operator-baseline/v1`.

- [ ] **Step 6: Commit Task 5**

```bash
git add tooling/summarize_q1_cpu_operator.py tests/test_summarize_q1_cpu_operator.py benchmarks/schema/q1-cpu-operator.example.json
git commit -m "feat: validate Q1 operator evidence"
```

## Task 6: Реализовать настоящий Prism CPU_REPACK runner

**Files:**

- Create: `experiments/E004-q1-cpu-operator-baseline/q1_cpu_operator_runner.cpp`
- Create: `experiments/E004-q1-cpu-operator-baseline/CMakeLists.txt`
- Create: `tests/test_q1_cpu_operator_runner.py`

- [ ] **Step 1: Write a failing executable-contract test before C++**

The test creates a literal `K=128`, `M=16` Q1 payload and F32 activation, then launches the executable named by `Q1_CPU_OPERATOR_RUNNER`. Without that environment variable the ordinary repository suite skips this hardware/runtime integration test with an explicit reason; during this task the variable is mandatory.

```python
completed = subprocess.run(
    [runner, "--self-test", "--fixture-dir", str(fixture),
     "--iterations", "2", "--output-dir", str(output)],
    text=True,
    capture_output=True,
)
self.assertEqual(completed.returncode, 0, completed.stderr)
records = [json.loads(line) for line in (output / "runner.jsonl").read_text().splitlines()]
self.assertEqual([row["record"] for row in records],
                 ["identity", "memory", "sample", "sample", "sample", "result"])
self.assertEqual(records[0]["weight_buffer_type"], "CPU_REPACK")
self.assertEqual(records[1]["expanded_weight_ddr_bytes"], 0)
self.assertEqual(records[1]["cpu_fallback_count"], 0)
```

Read `output.f32.bin` and compare its 16 values with hand-derived expected results for the literal fixture. This is a behavior test; it must not grep C++ source text.

- [ ] **Step 2: Run the red executable test**

```bash
Q1_CPU_OPERATOR_RUNNER=/tmp/q1-cpu-operator-build/q1_cpu_operator_runner \
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_cpu_operator_runner -v
```

Expected: failure because the executable does not exist.

- [ ] **Step 3: Implement strict CLI and tensor allocation**

CLI:

```text
q1_cpu_operator_runner
  --weights weights.q1_0.bin
  --activation activation.f32.bin
  --ne0 K
  --ne1 M
  --threads N
  --warmup 1
  --iterations 50
  --output-jsonl runner.jsonl
  --output-f32 output.f32.bin
  --output-q8 activation.q8_0.bin
```

Create the Q1 weight in a dedicated no-alloc context. Enumerate `ggml_backend_dev_get_extra_bufts` through `ggml_backend_reg_get_proc_address`, select only the buffer whose name is exactly `CPU_REPACK`, allocate with `ggml_backend_alloc_ctx_tensors_from_buft`, call `ggml_backend_tensor_set` once, then release the temporary canonical input bytes before warmup. Allocate activation/output on the ordinary CPU backend, build one `ggml_mul_mat`, and fail unless the backend supports it with the Q1 source actually bound to `CPU_REPACK`.

- [ ] **Step 4: Emit independent q8 fixture and strict events**

Use the public pinned Prism call `ggml_quantize_chunk(GGML_TYPE_Q8_0, ...)` for the saved activation fixture. JSONL records:

```json
{"schema_version":"q1-cpu-operator-run/v1","record":"identity"}
{"schema_version":"q1-cpu-operator-run/v1","record":"memory"}
{"schema_version":"q1-cpu-operator-run/v1","record":"sample","kind":"warmup","iteration":0,"host_us":0.0,"compute_us":0.0}
{"schema_version":"q1-cpu-operator-run/v1","record":"sample","kind":"measured","iteration":0,"host_us":0.0,"compute_us":0.0}
{"schema_version":"q1-cpu-operator-run/v1","record":"result","output_f32_bytes":69632}
```

Measure each resident graph compute with `CLOCK_MONOTONIC_RAW`. Write phase events `setup_begin`, `repack_begin`, `repack_end`, `warmup_begin`, `warmup_end`, `compute_begin`, `compute_end`, `output_read_begin`, `output_read_end`, `teardown` into `VIP9000_PHASE_FILE`. Do not include setup/repack in measured samples. The Python summarizer, not the C++ binary, computes cryptographic hashes of output artifacts; the accepted Prism build has no OpenSSL dependency.

- [ ] **Step 5: Add exact build integration**

`CMakeLists.txt` must require `PRISM_SOURCE` and support two explicit modes:

```text
-DBUILD_PINNED_PRISM_FROM_SOURCE=ON
-DPRISM_BUILD=/absolute/path/to/native/build
```

Host compile checks use `BUILD_PINNED_PRISM_FROM_SOURCE=ON` and `add_subdirectory(${PRISM_SOURCE}/ggml ...)`. A733 TDD/qualification uses the already accepted native `PRISM_BUILD` libraries. Both modes check the source Git SHA at configure time, include Prism public/internal ggml headers, and link `ggml`, `ggml-base`, and `ggml-cpu`. Refuse a different commit unless the caller supplies `-DALLOW_UNPINNED_PRISM=ON`; an unpinned build must be marked unqualified in runner identity.

- [ ] **Step 6: Compile and smoke-test outside the timed hardware run**

```bash
cmake -S experiments/E004-q1-cpu-operator-baseline \
  -B /tmp/q1-cpu-operator-build \
  -DPRISM_SOURCE=/home/random/src/llama-prismml \
  -DBUILD_PINNED_PRISM_FROM_SOURCE=ON
cmake --build /tmp/q1-cpu-operator-build --parallel 2
/tmp/q1-cpu-operator-build/q1_cpu_operator_runner --help
```

Then build natively on the A733 against its accepted Prism build and run the behavior test there:

```bash
Q1_CPU_OPERATOR_RUNNER=/tmp/q1-cpu-operator-build/q1_cpu_operator_runner \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_cpu_operator_runner -v
```

Expected: both configure steps name commit `38c66ad`; host compile succeeds; A733 self-test proves `CPU_REPACK`, numeric output and records. The x86 host is not expected to expose the Arm Q1 CPU_REPACK kernel and never produces a qualified performance result.

- [ ] **Step 7: Run source tests and commit Task 6**

```bash
Q1_CPU_OPERATOR_RUNNER=/tmp/q1-cpu-operator-build/q1_cpu_operator_runner \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_q1_cpu_operator_runner -v
git add experiments/E004-q1-cpu-operator-baseline/q1_cpu_operator_runner.cpp experiments/E004-q1-cpu-operator-baseline/CMakeLists.txt tests/test_q1_cpu_operator_runner.py
git commit -m "feat: benchmark Prism Q1 CPU operators"
```

## Task 7: Оркестрировать безопасный A55/A76 target matrix

**Files:**

- Create: `tooling/run_bonsai_q1_cpu_operator.sh`
- Create: `tests/test_run_bonsai_q1_cpu_operator.py`
- Create: `experiments/E004-q1-cpu-operator-baseline/README.md`
- Modify: `tooling/README.md`

- [ ] **Step 1: Write failing wrapper behavior tests**

Run the wrapper against temporary fake guard/profiler/taskset/runner/summarizer executables passed through test-only environment variables. Assert observable exit codes, the captured argv sequence and created result artifacts. The tests must prove the wrapper:

- uses `set -eu`;
- accepts explicit `--cores` and `--threads`;
- validates only `0-5/6` or `6-7/2` for qualified runs;
- invokes guard, profiler, taskset, runner, golden and summarizer in the required order;
- passes no credential, model payload or network destination to any child;
- refuses an existing result directory.

Do not assert that a shell source line exists. Test the script as an executable with controlled child processes.

- [ ] **Step 2: Run the red test**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_run_bonsai_q1_cpu_operator -v
```

Expected: wrapper missing.

- [ ] **Step 3: Implement the wrapper**

Interface:

```text
run_bonsai_q1_cpu_operator.sh
  --run-id ID
  --fixture-dir DIR
  --runner BIN
  --cores 0-5|6-7
  --threads 6|2
  --results-root benchmarks/results
```

The wrapper starts the thermal guard on a collector core outside the tested set, profiles one persistent runner process, always stops the guard, then runs the summarizer. It records exact commands and binary/model/tensor hashes in metadata. It must not alter the fan policy or governors.

After the runner writes `activation.q8_0.bin`, the wrapper invokes the Task 4 golden CLI once to create `expected.f32.bin`, then passes that file and the runner output to the summarizer. Golden generation remains outside every measured sample.

- [ ] **Step 4: Document terminology in Russian**

README must define:

- `resident`: weights are prepared once and reused;
- `repack`: byte-preserving interleave for the CPU kernel, not dequantization;
- `golden`: independent expected numerical output, not repeated equality;
- `declared bytes`: allocation/accounting fact, not a hardware DDR counter;
- `host-total` versus `compute-only`;
- why operator GEMV/s is not `tokens/s`.

- [ ] **Step 5: Run tests and commit Task 7**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_run_bonsai_q1_cpu_operator -v
git add tooling/run_bonsai_q1_cpu_operator.sh tests/test_run_bonsai_q1_cpu_operator.py experiments/E004-q1-cpu-operator-baseline/README.md tooling/README.md
git commit -m "feat: run profiled Q1 CPU matrix"
```

## Task 8: Выполнить реальные production-shape прогоны на A733

**Files:**

- Create: `benchmarks/results/bonsai-q1-gate-a55-001/summary.json`
- Create: `benchmarks/results/bonsai-q1-down-a55-001/summary.json`
- Create: `benchmarks/results/bonsai-q1-gate-a76-001/summary.json`
- Create: `benchmarks/results/bonsai-q1-down-a76-001/summary.json`

- [ ] **Step 1: Verify target preconditions read-only**

```bash
sha256sum /path/to/Bonsai-27B-Q1_0.gguf
git -C /path/to/llama-prismml rev-parse HEAD
cat /sys/class/devfreq/3600000.npu/cur_freq
cat /sys/class/thermal/thermal_zone*/temp
cat /proc/swaps
```

Expected: exact model/runtime hashes, readable telemetry, no unexpected swap use. Record failure rather than changing system policy inside this task.

- [ ] **Step 2: Build runner natively on the board**

Use the accepted Prism native build tree and the CMake command from Task 6. Record compiler version, flags and runner SHA-256.

- [ ] **Step 3: Run A55×6 matrix**

```bash
tooling/run_bonsai_q1_cpu_operator.sh \
  --run-id bonsai-q1-gate-a55-001 \
  --fixture-dir /tmp/bonsai-q1-fixtures/blk0-ffn-gate \
  --runner /tmp/q1-cpu-operator-build/q1_cpu_operator_runner \
  --cores 0-5 --threads 6

tooling/run_bonsai_q1_cpu_operator.sh \
  --run-id bonsai-q1-down-a55-001 \
  --fixture-dir /tmp/bonsai-q1-fixtures/blk0-ffn-down \
  --runner /tmp/q1-cpu-operator-build/q1_cpu_operator_runner \
  --cores 0-5 --threads 6
```

- [ ] **Step 4: Run A76×2 matrix**

Repeat both fixtures with `--cores 6-7 --threads 2` and run IDs ending in `a76-001`.

- [ ] **Step 5: Check qualification without cherry-picking samples**

```bash
for summary in benchmarks/results/bonsai-q1-*-001/summary.json; do
  python3 tooling/summarize_q1_cpu_operator.py --check "$summary"
done
```

Expected for qualification: correctness passes, 50 samples, CV `<=2%`, zero expanded weight bytes, zero fallback, zero swap delta and no thermal failure. If CV fails, preserve the failed run and create a new run ID; never overwrite it.

- [ ] **Step 6: Commit immutable summaries and bounded raw evidence**

Do not commit extracted weights, activations, raw model or oversized telemetry. Commit summaries and small text/JSON evidence allowed by `docs/profiling/profiling-contract.md`.

```bash
git add benchmarks/results/bonsai-q1-gate-a55-001 benchmarks/results/bonsai-q1-down-a55-001 benchmarks/results/bonsai-q1-gate-a76-001 benchmarks/results/bonsai-q1-down-a76-001
git commit -m "bench: record A733 Q1 operator baselines"
```

## Task 9: Объяснить baseline и зафиксировать следующий NBG gate

**Files:**

- Create: `docs/evidence/bonsai-q1-cpu-operator-baseline-2026-08-10.md`
- Modify: `README.md`
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Write the Russian evidence report from saved JSON only**

The report must contain one comparison table:

| Tensor/shape | Executor | Median ms | p10/p90 ms | CV | Effective packed GB/s | Golden | Qualified |
|---|---|---:|---:|---:|---:|---|---|

Compute effective packed bandwidth as `12_533_760 / median_seconds / 1e9` and label it a lower-bound operator metric. Do not extrapolate it into model `tokens/s` by assuming all 497 GEMVs behave identically.

- [ ] **Step 2: State the exact decision for the NPU tranche**

The next TIM-VX/NBG plan may proceed only if it consumes the same tensor identity, activation fixture, output golden and operator schema. Its acceptance threshold is host-total median at least 10% faster than the qualified A55 result for the same tensor/shape, with zero expanded-weight DDR bytes, nonzero device cycles and no CPU fallback.

- [ ] **Step 3: Run the full local regression suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass; no whitespace errors.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/evidence/bonsai-q1-cpu-operator-baseline-2026-08-10.md README.md benchmarks/README.md
git commit -m "docs: explain Bonsai Q1 operator baseline"
```

## Completion Gate

This plan is complete only when:

1. the real GGUF manifest reproduces the exact 498/497 tensor and byte counts;
2. both production FFN shapes have qualified A55 and A76 operator evidence or retained, explicitly failed runs;
3. every qualified output passes the independent golden;
4. CPU_REPACK is proven and no expanded weight buffer is allocated;
5. setup/repack are outside resident samples and all byte/timing meanings are explicit;
6. no operator number is presented as full-model `tokens/s` improvement;
7. all tests pass, commits are published, and the next NBG compiler plan references this exact contract.

The first full-model speed claim remains gated on deterministic Bonsai generation and measured single-stream decode median above `0.726175 ток/с` with p10 above `0.7263472 ток/с`.
