# Open research questions

## Legal and distribution

- What exact license/EULA accompanies the Orange Pi/Allwinner A733 NPU SDK?
- May applications using the SDK be open source?
- May headers, sample code, shared libraries, firmware, and generated network binaries be redistributed?
- Are benchmark publication, interface observation, or reverse engineering restricted?
- Which SDK components have independent open-source licenses?

## Target identity

- What is the exact Orange Pi board revision and SKU?
- Is the memory configuration truly 12 GB LPDDR5, and at what effective speed?
- What is the exact VIP device/product ID?
- Which kernel driver, firmware, VIPLite/OpenVX runtime, compiler, and SDK versions are installed?
- Is the NPU exposed through VIPLite, OpenVX, TIM-VX, another API, or several layers?

## Runtime model

- Are graphs compiled on-device or ahead of time?
- What is the cold compile time and graph-load time?
- Can a graph and its constant tensors remain resident?
- Are dynamic tensor shapes supported?
- Can several sequence lengths share a graph?
- Can multiple networks execute concurrently?
- Is execution asynchronous?

## Operations and precision

- Which matrix multiplication shapes are efficient?
- Does the runtime support FP32, FP16, BF16, INT8, UINT8, INT16, INT4, binary, or ternary inputs?
- Are quantization scales per tensor or per channel?
- Which activation, normalization, softmax, reshape, transpose, gather, and masking operations are supported?
- Can custom operators or kernels be installed?
- Which operations silently fall back or are rewritten by conversion tools?

## Memory and data movement

- Is NPU memory separate or shared system RAM?
- Can CPU buffers be imported zero-copy?
- Are dmabuf, ION, CMA, or OpenVX memory-import APIs available?
- What cache-coherency operations are required?
- What are alignment, stride, and layout restrictions?
- Can GGUF weights be repacked once and retained?
- Does concurrent NPU traffic reduce CPU LLM bandwidth materially?

## `ggml` integration

- Is operator-level offload viable, or is subgraph offload mandatory?
- Which backend API revision should be pinned?
- Can the standard scheduler partition the graph adequately?
- How should compiled graph cache keys and invalidation work?
- How should backend failures and unsupported nodes fall back safely?
- Is dynamic loading preferable to a compile-time vendor dependency?

## Transformer partitioning

- Which path wins for prefill?
- Which path wins for decode?
- Can MLP projections be grouped into stable subgraphs?
- Can attention masks and KV-cache updates be expressed efficiently?
- Can full transformer blocks be compiled for fixed context buckets?
- Is it better to keep token-by-token decode on Cortex-A76 and offload only prompt processing?
- Can the NPU instead serve embeddings/RAG while CPU runs generation, producing a better whole-system result?

## Model and quantization strategy

- Which small reference model should be used first?
- Is an FP16/INT8 NPU model competitive with a CPU GGUF Q4 model after all overheads?
- Can ternary Bonsai-like weights map to any native NPU primitive?
- Would unpacking ternary weights erase storage/bandwidth gains?
- Should the first proof use an embedding model, a tiny causal LM, or a synthetic transformer block?

## Measurement

- How will wall time, CPU time, NPU utilization, memory traffic, thermals, and energy be collected?
- Does the driver expose counters or profiling traces?
- What run-to-run variance and thermal steady state are acceptable?
- What correctness metrics are required for operators, logits, and generated text?

## Product decision

The project should answer one of three outcomes honestly:

1. A general `ggml` backend is practical.
2. A model-specific or prefill-only integration is practical.
3. Full LLM offload is not worthwhile, but NPU embeddings/auxiliary models plus CPU generation is the best architecture.

All three are valid research results.
