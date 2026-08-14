# Provenance failure: `117cc6bd` versus `12262831`

Status: **RESOLVED ROLE CONFUSION; TARGET RERUN REMAINED BLOCKED DURING AUDIT**.

The reported prefix `117cc6bd` is real, but it is not an executable hash. It
is the SHA-256 of the first sample's 917-byte JSON stdout:

```text
117cc6bdab782ba15b5ebbfb3742bacb6dc0fa9dd5217371b0beb68023a3d24d
raw/cpu0-pair1-hot/harness.stdout.raw
```

The executable is the distinct 75,160-byte ELF whose full SHA-256 is:

```text
12262831a5527f98426863721ea9fb0c1bb6045bd468124c00675328e8d35f80
```

## Direct recomputation

The binary hash was recomputed independently from all four surviving copies:

| Copy | Size | Device/inode | SHA-256 |
|---|---:|---|---|
| target build C `/tmp/e055-harness-native-7dcdab61-20260814t161000z-c/e055` | 75,160 | 34/2043 | `12262831...35f80` |
| target build D `/tmp/e055-harness-native-7dcdab61-20260814t161000z-d/e055` | 75,160 | 34/2048 | `12262831...35f80` |
| first local scp copy `/tmp/e055-harness-native-7dcdab61.bin` | 75,160 | 52/249204 | `12262831...35f80` |
| result artifact `artifacts/e055-native-aarch64` | 75,160 | 52/249477 | `12262831...35f80` |

Both target files are mode 0755, regular, single-link AArch64 PIE executables.
Both are dynamically linked through `/lib/ld-linux-aarch64.so.1`, require only
GLIBC 2.17/2.34, and have identical version-info. They intentionally contain
no GNU linker build ID because the exact link used `-Wl,--build-id=none`; their
only displayed ELF note is the Linux ABI 3.7 tag. An absent build ID is
therefore expected, not missing evidence.

## Source and object chain

Direct target recomputation in both fresh build directories produced:

| Role | C SHA-256 | D SHA-256 |
|---|---|---|
| C++ source | `7dcdab617cfee84468c25e70a9e130107de4b3a1c4361f0949353446be89c281` | identical |
| stock assembly | `3338df3dfedf683247ef0da3975db371667add1c2552f77219718a444687143f` | identical |
| C++ object, 49,560 bytes | `8523c3953b8b56324dd430cd68beee5947540ec5b30de0ecc4dab1e76966d109` | identical |
| assembly object, 3,728 bytes | `00b5bb114f300b04b69a596964f499ba76b9b6e66d8d83c2342747f3fb590c53` | identical |
| linked executable | `12262831a5527f98426863721ea9fb0c1bb6045bd468124c00675328e8d35f80` | identical |

The target compiler was independently rehashed as
`22084e38d464db4299003c899eae3f1cb563ed6a75b3179e3e2a2d593d640528`,
size 1,319,624, at `/usr/bin/aarch64-linux-gnu-g++-12`; its identity is Debian
G++ 12.2.0. Exact compile argv is recorded in
`build-evidence/native-builds.json`.

The initial uncommitted build-evidence draft contained incorrectly transcribed
C++ and assembly object hashes. Those claims were corrected only after the
fresh target recomputation above. This discrepancy is preserved here rather
than hidden.

## Runner/raw/manifest trace

The direct phase deliberately did not use the rejected helper runner protocol,
so it has no helper `runner.json` asserting an executable hash. Every E049c raw
file records the exact target build-C executable path in `command`. The local
artifact copy is hash-bound separately in `raw-manifest.json` as
`artifacts/e055-native-aarch64=12262831...35f80`. The same manifest binds the
first harness result as
`raw/cpu0-pair1-hot/harness.stdout.raw=117cc6bd...3d24d`. The two values are
therefore distinct declared file roles, not competing identities for one file.

The target build directories contain source, objects, and executable but no
separate compiler stdout/stderr log. The compile argv record is consequently a
recorded command/provenance assertion supported by byte-identical sources,
objects, and outputs, not an immutable raw build transcript. This limitation
is explicit.

## Identity conclusion

The single supported hypothesis is **role confusion in the abbreviated hash
report**: `117cc6bd` was read from the manifest's first harness-stdout entry
and mistakenly labeled as the harness artifact. No 75,160-byte binary with
that hash was found locally or in either target build. No replacement of the
native executable occurred. The only qualified executable identity for a
future unchanged rerun is the full `12262831...35f80` hash.
