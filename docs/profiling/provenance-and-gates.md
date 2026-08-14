# Provenance and promotion gates

## Evidence hierarchy

Use the strongest available class and state it explicitly:

1. **Verified on target**: reproduced on the exact Orange Pi target.
2. **Verified upstream**: confirmed in an official SoC, vendor, or upstream
   source.
3. **Verified in supplied tooling**: observed in a host SDK or local toolchain;
   target compatibility is not implied.
4. **Prior local observation**: retained legacy output not reproduced under the
   current contract.
5. **Hypothesis**: a testable design expectation.
6. **Unknown**: missing, contradictory, or unsafe to infer.

The experiment registry stores the authoritative branch, full commit SHA, and
evidence path for each row. A path on another branch is a pointer, not a copy.
The [branch inventory](../experiments/branch-inventory.md) records the remote
heads used for this snapshot.

## Required gates

### Identity gate

Pin the repository, runtime, compiler, SDK, driver, kernel, model, tokenizer,
prompt suite, configuration, and target identity. A shortened SHA or an
unhashed model path is not sufficient for a qualified result.

### Correctness gate

Use an independent CPU golden for operator and tensor experiments. Use exact
deterministic token agreement for generation. A repeated device output that
matches only the first device output is a repeatability result, not a golden.

### Performance gate

Report end-to-end wall time with separate prompt, TTFT, and decode phases,
warmups, repetitions, and dispersion. Include setup/compile/load/copy costs
when the claim includes them. A three-token PMU marker, one short screen, or a
device-only interval cannot become a full-model optimization claim.

### Safety gate

Record thermal guard state, all relevant thermal zones, cpufreq cooling state,
governors, observed frequencies, swap, available memory, and reset or kernel
events. Restore temporary settings. Frequency and voltage work must remain
inside secure firmware/BSP support and the approved safety boundary.

### Memory gate

The 32-bit LPDDR5-4800 / 19.2 GB/s value is a theoretical SoC ceiling. A
controller readback of 510 MHz is not a data-rate or sustained-bandwidth
measurement. PMU events are not DDR bytes. Any bandwidth claim needs a declared,
source-grounded measurement method.

### Promotion gate

Use these statuses in the machine registry:

| Status | Meaning |
|---|---|
| `accepted` | Evidence passed the experiment's declared gate for its claim class. |
| `rejected` | A declared quality or performance gate failed; retain the evidence. |
| `failed` | Setup, command, board, or measurement execution failed. |
| `diagnostic` | Useful bounded observation with no optimization or full-model claim. |
| `unqualified` | A run completed but lacks a required quality, end-to-end, or safety field. |
| `planned` | A question or design exists without a completed result. |

Only a row with `claim_class=full_model`, complete quality evidence, and
`is_current_best=true` may define the current best. This repository marks E035
as the only current best. E044 and E049d-v2 remain explicitly bounded.

## Public artifact and privacy policy

Public weights, binaries, NBGs, custom kernels/source, SDK or kernel patches,
and NPU tools may be published when redistributable. Each public artifact must
be listed in [`public-artifact-manifest.json`](../experiments/public-artifact-manifest.json)
with SHA-256, byte size, origin, source commit, build/runtime/toolchain
provenance, and destination (`git`, `release-assets`, or `external`).
Personal/sensitive data is prohibited, including tokens, passwords, logins,
private keys, identifiers, and credentials. Existing payloads and raw evidence
are not automatically tracked or copied; use Git LFS or release assets for a
large public payload only after the manifest and privacy gate pass.

## Review checklist

Before committing a result or changing the registry:

- validate JSON against the registry schema;
- run `python3 tooling/check_repository_index.py --repo-root .`;
- run the focused and relevant existing tests;
- inspect the diff for language, broken links, duplicate payloads, and changed
  legacy evidence;
- record the exact test command and commit in the handoff.
