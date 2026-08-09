# Prism CPU reference runtime — 2026-08-09

## Pinned source and target

- Repository: PrismML llama.cpp fork
- Commit: `38c66ad0241da4f9fcce541cda8edc219086cec5`
- Tag observed in the clean checkout: `prism-b9594-38c66ad`
- Target ABI: Debian glibc 2.36, Linux 6.6.98, AArch64
- Target CPU topology: CPUs 0–5 in policy 0 and CPUs 6–7 in policy 6

A host cross-build was rejected before testing because its dynamic symbols
required glibc as new as 2.43. The accepted binaries were built natively from a
clean shallow Git checkout whose `HEAD` matched the full pinned commit. This
avoids both the ABI mismatch and an unverifiable `commit unknown` build string.

## Native build

The reference build uses Release, GCC 12.2.0,
`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`, `GGML_NATIVE=OFF`, and disables ccache,
CUDA, Vulkan, BLAS, CURL, and OpenSSL. Its build number is pinned to `9594`.
Configuration detected NEON, ARM FMA, DOTPROD, OpenMP, and REPACK; SVE and the
configured FP16-vector probe were not enabled. An FP16-enabled build remains a
separate future A/B candidate rather than an untracked change to this reference.

Run `prism-cpu-native-build-38c66-002` completed in 707.353 seconds with child
and profiler exit `0` and 6,319 profiler samples. All generated executable and
shared-library dependencies resolved on the target; the highest observed GLIBC
symbol requirement was `GLIBC_2.34`.

| Artifact | SHA-256 |
|---|---|
| `llama-cli` | `47a2e9cc8616145580a49688b21f1040dcfa8f1501925424781314290dd065d1` |
| `llama-bench` | `27f77e562950158b0bf76bb9026d38d27bbb0826485b07dfc3cbb3bd406cefbc` |
| `llama-completion` | `891eb1b5d34797774762e3b5198b58322c92ee59203412ec5e7667c4cc2acb18` |
| `libggml-base.so` | `c6da0b2918fcb6f63c096a62295348b754e6bea6fe2ade7052886869e09a7066` |
| `libggml-cpu.so` | `cadea3fb3a68d9cb1b81ea813208a101bcf8315c8f4c346c6ebc9589e1f3c167` |
| `libggml.so` | `b88b4a824d85e3a33b62f6aeaa7a56b68ba9448d36a6ac00aa54fbc7974a63d7` |
| `libllama.so` | `e9f02d41870a980d6a6249b429dfc68e50906663a81813828955ed763c5d36cc` |
| `libllama-common.so` | `cf58da74917500f139bc5d2332dba5a99d07356f8de616d9dfdde446256880b2` |
| `libllama-completion-impl.so` | `52f8c65e9cf69341e95e84c76cd7fb4b0bddedda48541947cb1fd68d086ed13d` |

`llama-cli` and `llama-completion --version` both report build `9594`, commit
`38c66ad`, GCC 12.2.0, and Linux AArch64. `llama-bench` has no working
`--version` option in this revision, so its binary hash and the shared build
tree provide its executable identity.

## First model execution findings

Run `bonsai27b-q1-cpu-a76-smoke-001` exposed an interface mismatch before a
valid result was claimed. Although the common help lists conversation flags,
`llama-cli` reports that `--no-conversation` is unsupported and directs callers
to `llama-completion`. Combined with `--simple-io` and EOF, it entered a fast
prompt-printing loop. The thermal guard stopped the exact run with status 143.
Its 2,034,908,227-byte repetitive stdout was losslessly compressed to
9,862,898 bytes to reclaim target storage; the raw SHA-256 is
`1709d82e6388f15faae0b08010d9193e235dade1fdaca3aa861b5051da384c39`
and the gzip SHA-256 is
`14a77b5daa3a05d6efa05f63c8e2fcbca7b05c18b24c90f9d714b5717c8e8787`.

Run `bonsai27b-q1-cpu-a76-smoke-002` therefore used `llama-completion` without
`--simple-io`. It completed with exit `0` in 55.243 seconds on CPUs 6–7, peak
direct-child RSS 7,361,012 KiB, and no swap use. It measured 19 prompt tokens at
1.30 token/s and seven decode evaluations at 0.63 token/s. The eight-token cap
ended while the model was still inside its `<think>` section, so this result
proves model/runtime execution only and is not quality evidence.
