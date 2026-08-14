# Public artifacts and release destinations

Public weights, binaries, NBGs, custom kernels/source, SDK or kernel patches,
and NPU tools may be published when the license and source permit
redistribution. Personal/sensitive data is prohibited, including tokens,
passwords, logins, private keys, identifiers, and credentials.

Every intentional public artifact is recorded in
[`public-artifact-manifest.json`](public-artifact-manifest.json). The manifest
requires the artifact path, SHA-256, byte size, origin, source commit,
build/runtime/toolchain provenance, destination, approved class, publication
status, scientific-use flag, and verification state. The checker compares the
digest and size for artifacts whose destination is Git.

Large files use one of two reviewed destinations:

1. Git LFS patterns in [`.gitattributes`](../../.gitattributes), after the
   manifest and privacy gate pass. Adding an attribute does not retroactively
   start tracking existing payloads.
2. A release asset attached to a versioned GitHub release, with the same
   manifest entry and immutable source commit. Every release or external entry
   also needs an immutable HTTPS source URL or release-asset locator and a
   checksum provenance/attestation. Planned entries must be explicitly
   `verification_state=unverified` and `scientific_use=false`; they cannot
   support a scientific claim. Release assets keep large public payloads out
   of the canonical Markdown layer.

Raw historical evidence and existing payloads remain on their authoritative
branches unless a separately reviewed public-artifact entry authorizes a copy.
