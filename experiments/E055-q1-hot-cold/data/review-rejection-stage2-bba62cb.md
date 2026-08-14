# Stage 1 second review rejection — commit bba62cb

Status: **REJECTED EVIDENCE; PRESERVED FOR AUDIT**.

The revision at commit `bba62cbb7abd8737281355f5f784e30904037cc6` is not
accepted as a target-ready Stage 1 gate. The independent review found:

1. Its manifest recorded the parent commit rather than using a verifiable
   staged-tree plus post-commit ref binding.
2. Its analyzer validated only counts of pair IDs and divided independent
   medians instead of computing actual per-pair penalties.
3. Its validator accepted unbound or all-zero hashes, negative PMU counts,
   integer/Boolean running ratios, malformed checksums, and unverified hot
   conditioning.
4. Working-set rounding could overflow; a non-aligned thrash size left trailing
   requested bytes untouched; the nominal 12.5 MiB constant was wrong.
5. The upstream `repack.cpp` hash and the documented 4% promotion threshold
   were not enforced by code.

This evidence remains in Git history and in the manifest. It contains no target
run and must not be used for a bottleneck or performance claim.
