# Related work and prior art

## 1. TIM-VX

Repository: `VeriSilicon/TIM-VX`

Why it matters:

- public C++ graph API intended for VeriSilicon machine-learning accelerators;
- maps operations onto a VeriSilicon OpenVX implementation;
- provides a plausible integration layer above platform-specific runtime components;
- source license is permissive, while the required platform SDK/runtime must be obtained separately.

Questions to answer on A733:

- Does the Orange Pi/Allwinner SDK include a TIM-VX-compatible OpenVX implementation?
- Which TIM-VX revision matches the runtime ABI?
- Which operations and tensor formats are enabled for the exact VIP9000 product ID?
- Can compiled graphs and constant tensors persist across invocations?
- Can host memory be imported without extra copies?

## 2. `waz664/vip9000-embeddinggemma`

This is the most directly relevant public implementation found during repository bootstrap.

Reported target:

- Radxa Cubie A7S;
- Allwinner A733;
- VIP9000-series device identified as `VIP9000NANODI_PLUS_PID0X1000003B`;
- VIPLite runtime path;
- an EmbeddingGemma transformer graph compiled into a VIPLite NBG;
- CPU handling tokenization, embedding lookup, pooling/projection tail, and normalization around an NPU transformer body.

Important reported findings:

- an additive attention-bias input was required to restore output quality;
- unsupported ONNX operations were rewritten before vendor compilation;
- the persistent runner loads the network binary once to reduce repeated initialization;
- the NPU path reduced CPU time substantially but did not beat a four-thread CPU TFLite latency result in the published single-query benchmark;
- vendor SDK files, model weights, and generated network binaries were deliberately not committed.

Why this matters:

- proves that at least one A733/VIP9000 software stack can execute a transformer-derived graph;
- demonstrates a realistic ONNX → vendor compiler → NBG → persistent runtime path;
- confirms that numerical fidelity, masks, unsupported operators, and cold-start cost are central concerns;
- supplies concrete scripts and runtime conventions to inspect.

Limitations:

- it is an embedding model rather than autoregressive LLM generation;
- it is board- and SDK-specific;
- the reported NPU latency did not outperform the best listed CPU latency;
- it does not provide a general `ggml` NPU backend;
- generated binaries and proprietary runtime components are external.

## 3. `llama.cpp` / `ggml` accelerator backends

Backends such as CPU, CUDA, Vulkan, Metal, SYCL and others illustrate the general integration pattern:

- device and backend registration;
- buffer types and tensor transfers;
- operation capability predicates;
- graph scheduling across backends;
- optional dynamically loaded backend libraries;
- backend-specific graph execution and synchronization.

The Vivante experiment should reuse the public backend contract rather than embed vendor calls throughout model code.

Study targets at a pinned upstream commit:

```text
ggml/include/ggml-backend.h
ggml/src/ggml-backend.cpp
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-cpu/
ggml/src/ggml-vulkan/
ggml/src/ggml-sycl/
tests/test-backend-ops.cpp
```

Exact paths may change upstream and must be recorded with the selected commit SHA.

## 4. Rockchip and other NPU LLM runtimes

Rockchip RKNN-based projects are architecturally relevant even though they do not target Vivante:

- they demonstrate model conversion into fixed accelerator graphs;
- many use model-specific partitioning rather than a fully general `ggml` backend;
- they expose the tension between static graph compilers and autoregressive KV-cache mutation;
- they illustrate why prefill and decode often need different graph shapes or execution strategies.

Use them as design references, not as evidence of VIP9000 capability.

## 5. Bonsai / ternary model work

Ternary and binary models may reduce storage and memory traffic, but they are not automatically suitable for VIP9000.

Questions:

- Does the runtime expose low-bit matrix operations below INT8?
- Can custom kernels be installed?
- Would ternary weights need unpacking to INT8/FP16, removing their main benefit?
- Is CPU execution of ternary kernels more practical than NPU execution on this platform?

This remains a later research branch. The first backend should characterize supported conventional precisions before pursuing custom low-bit execution.

## Research lessons from prior art

1. Keep the NPU runtime persistent.
2. Compile graphs ahead of the hot inference path.
3. Treat attention masks and numerical equivalence as first-class tests.
4. Expect unsupported ONNX/operator patterns.
5. Measure CPU time as well as wall time: freeing CPU capacity may be useful even without lower latency.
6. Do not assume transformer embeddings imply efficient token-by-token generation.
7. Preserve a clean legal boundary around vendor SDK/runtime artifacts.
