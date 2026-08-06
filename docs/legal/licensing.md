# Licensing and redistribution research

> This document is an engineering inventory, not legal advice. Final conclusions require review of the exact SDK and board-image licenses supplied with the target device.

## Current working conclusion

An open-source integration appears feasible **only if proprietary vendor assets remain external dependencies** unless their licenses explicitly permit redistribution.

The likely safe architecture is:

- repository code under a permissive open-source license;
- optional build/runtime detection for user-installed Vivante/Allwinner/Orange Pi components;
- no committed SDK archives, proprietary shared libraries, firmware, kernel modules, confidential manuals, model weights, or generated binaries without explicit permission;
- documented version hashes and acquisition steps instead of copied assets.

## Component matrix

| Component | Known status | Repository policy | Verification needed |
|---|---|---|---|
| Project-original source | Project choice | May be committed under selected license | Select license after dependency review |
| TIM-VX source | Public repository with permissive MIT-style license text | May be referenced or used subject to its license and notices | Confirm exact pinned revision and third-party notices |
| VeriSilicon OpenVX SDK | Required by TIM-VX; platform package supplied through SoC/vendor channels | Do not redistribute by default | Obtain exact SDK EULA/license |
| VIPLite runtime and tools | Vendor/platform specific | Do not redistribute by default | Inspect package license and binary redistribution clauses |
| Acuity conversion tools | Vendor tooling | Do not redistribute by default | Inspect license, output-binary restrictions, confidentiality terms |
| Kernel driver / firmware | Board/BSP/vendor assets | Do not copy from images by default | Determine GPL obligations for kernel code and separate firmware terms |
| Generated NBG/network binary | Derived using vendor tooling and model assets | Keep private until rights are clear | Check SDK output terms and model license |
| GGUF/model weights | Model-specific license | Do not commit unless redistribution is explicitly allowed | Record exact model license |
| `llama.cpp` / `ggml` code | Upstream license applies | Preserve notices and comply with upstream terms | Pin source revision and review before redistribution |
| Khronos OpenVX headers/spec | Khronos terms | Use official packages and preserve notices | Verify exact header/license package used |

## TIM-VX

The public `VeriSilicon/TIM-VX` repository contains permissive license text allowing use, modification, distribution, sublicensing, and sale, with preservation of the copyright and permission notice. It also lists bundled third-party components.

Important limitation: an open TIM-VX frontend does **not** imply that the device runtime, compiler, drivers, firmware, or board SDK are open or redistributable. TIM-VX documentation states that it requires the VeriSilicon OpenVX SDK and directs users of platforms not covered by public packages to obtain the SDK from the relevant SoC vendor.

## Questions for the exact SDK package

Before committing integration code that links or loads vendor components, collect and answer:

1. Is the SDK publicly downloadable, account-gated, NDA-gated, or supplied only in a board image?
2. Does the license permit development of open-source applications?
3. May headers be redistributed?
4. May sample code be copied or modified?
5. May runtime shared libraries be redistributed with an application?
6. May generated NBG/network binaries be distributed?
7. Are reverse engineering, benchmarking, or publication of benchmark results restricted?
8. Are interface descriptions or logs confidential?
9. Does static linking impose additional restrictions?
10. Are there patent notices or export-control conditions?
11. Does the SDK incorporate GPL/LGPL components requiring source or relinking provisions?
12. Are model-derived compiled artifacts governed by both the SDK license and model license?

## Repository handling rules

Create local-only directories in `.gitignore` for:

```text
vendor/
sdk/
models/
artifacts/private/
*.nb
*.nbg
*.so
*.a
*.ko
*.bin
```

Exceptions should be explicit and documented per file.

For proprietary packages, record only:

- package name and version;
- official acquisition location;
- SHA-256 checksum;
- required directory layout;
- expected library names and ABI;
- commands that operate on a user-supplied installation;
- a short original summary of relevant license obligations, without reproducing confidential text.

## Linking strategy

Prefer runtime dynamic loading (`dlopen`/`dlsym`) or an optional build target over mandatory static linking. This can keep the general project buildable without proprietary packages and makes the dependency boundary visible. It does not itself resolve license restrictions; the exact license remains controlling.

## Publication policy

Until the license review is complete:

- publish original source code and measurements only;
- do not upload vendor binaries or documentation;
- redact serial numbers, access tokens, SDK credentials, and private download URLs;
- describe reverse-engineering work only after confirming it is lawful in the applicable jurisdiction and contract;
- distinguish API observation and black-box measurement from decompilation or circumvention.

## Evidence to add

Place sanitized findings under:

```text
docs/legal/sdk-license-summary.md
references/sdk-inventory.example.yaml
```

The private original license files should remain outside Git unless redistribution is clearly permitted.
