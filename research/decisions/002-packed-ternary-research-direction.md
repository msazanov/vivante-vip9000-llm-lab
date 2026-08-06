# Decision 002 - Investigate direct packed ternary execution before model-wide INT4 conversion

**Date:** 2026-08-06  
**Status:** accepted research direction, not an implementation commitment

## Context

Ternary Bonsai stores language-model weights using values from `{-1, 0, +1}` with a shared FP16 scale per group. Permanent conversion to INT4 or INT8 would increase the long-lived model footprint and may remove the memory-bandwidth advantage that makes Bonsai attractive.

The A733 VIP9000 target contains:

- eight NN cores optimized for supported compiled graphs;
- one programmable PPU/shader core with EVIS;
- OpenCL/OpenVX compiler and runtime components;
- 512 KiB VIP SRAM;
- no publicly identified native ternary coefficient format.

## Decision

Before adopting a model-wide INT4 representation, the project will test direct packed ternary execution using the programmable PPU/EVIS path.

Two accelerator architectures will be evaluated:

1. direct packed ternary GEMV/GEMM on PPU/EVIS;
2. packed tile unpack on PPU/EVIS followed by native NN GEMM.

INT8 and INT4 conversions remain controlled performance baselines.

## Rationale

- preserves the released Bonsai representation;
- directly tests the memory-bandwidth hypothesis;
- avoids committing to a larger model format before target measurements;
- uses a genuinely programmable feature present in the exact product configuration;
- provides useful knowledge even if the final result favors CPU decode or native NN prefill.

## Consequences

- a full `llama.cpp` backend is deferred until E002 produces target data;
- the project must support both group-64 and group-128 Q2_0 layouts;
- proprietary compiler/runtime files remain external dependencies;
- direct PPU execution must be compared against optimized CPU NEON kernels;
- a custom kernel is not considered an eight-NN-core implementation unless profiling proves it;
- the likely final design is heterogeneous CPU + PPU + NN execution.

## Revisit conditions

Reconsider this decision when:

- the exact SDK cannot compile or execute custom kernels;
- PPU direct packed execution is consistently slower than CPU;
- producer/consumer transfers make tile-unpack-to-NN uncompetitive;
- native INT4/INT8 provides a decisive end-to-end advantage at an acceptable memory cost;
- a vendor-supported native packed-low-bit path is discovered.