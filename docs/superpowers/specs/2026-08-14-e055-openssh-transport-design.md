# E055 Disabled OpenSSH Transport Design

## Scope

This design adds one explicitly constructed OpenSSH transport to the accepted
`E055TargetExecutor` interface. The executor remains disabled when no transport
is injected. This change performs no board connection, identity smoke,
measurement, or model workload.

The transport exists only for the already bounded E055 O3 microgate. It does
not generalize arbitrary remote commands, arbitrary paths, arbitrary
environments, or interactive sessions.

## Trust boundary

The SSH host key is cryptographically checked against a caller-provided
external `known_hosts` file and an independently supplied canonical SHA-256
fingerprint. The helper also reports a digest derived from fixed board identity
inputs, and the caller requires that digest to equal an expected value.

These checks do not attest the remote userspace, kernel, Python interpreter,
helper output, or physical board. OpenSSH, the endpoint, the remote interpreter,
and the helper execution remain operationally trusted. The sealed records make
the observed configuration and bytes immutable; they do not turn observations
from that trusted endpoint into device attestation.

Credentials remain outside the repository and raw bundle. The first
implementation accepts non-interactive OpenSSH authentication through an
external agent or identity configuration. It does not accept a password,
token, private key, username, or credential-bearing environment value as
evidence, and it never includes command stderr or a credential-bearing argv in
sealed errors. Future explicitly injected ephemeral askpass or sshpass support
requires a separate review.

## Components

### Fixed remote helper

`tooling/e055_remote_helper.py` is a self-contained Python standard-library
program. Its exact local bytes, SHA-256, and size are publication artifacts.
SFTP places those bytes at the content-addressed path
`/tmp/e055-q1-hot-cold/helpers/<sha256>/helper.py`. SFTP batch text is generated
only after every inserted path matches the fixed content-addressed grammar; no
phase ID, argv element, environment value, credential, endpoint string, or
other untrusted value enters batch command text.

After bootstrap, every remote SSH command is the same fixed template:
`exec /usr/bin/python3 -- <validated-content-addressed-helper-path>`. All
operation inputs travel as one bounded canonical JSON object on stdin. The
helper rejects unknown keys, wrong schemas, noncanonical types, excessive
sizes, path traversal, unexpected roots, invalid hashes, invalid modes,
unexpected argv shapes, and unknown environment variables.

The helper supports exactly five stateful operations:

1. `exclusive_deploy` acquires the canonical lock with an owner request ID,
   creates the content-addressed phase tree exclusively, writes the two exact
   executables with `O_EXCL`, and returns fresh stat and SHA-256 observations.
2. `fresh_readback` opens those two files without following symlinks and returns
   a second independently generated observation bound to a distinct request ID
   and nonce.
3. `capture` creates one canonical run output directory exclusively and starts
   a fixed gate in a new session. The gate synchronously persists the exact
   owner/deployment/PID/PGID/session/start-time identity before `execve` of the
   E049c argv. The helper limits streams and time, observes the pinned child
   affinity, and returns all available raw streams and exit information.
4. `restore` removes only the deployment and lock whose inode and owner record
   match this transport instance. It terminates any surviving owned process
   group before removal.
5. `finalize_helper` is a separate, final request issued only after the caller
   has parsed and durably retained the complete restore response. It removes
   only the still-exact content-addressed helper inode and its empty directory;
   it rechecks device/inode immediately before unlink and the adapter compares
   the returned identity with the deploy/readback runtime identity.

Signals, timeout, stdin disconnect, and SSH channel failure trigger bounded
TERM/KILL cleanup of the helper-owned process group. Ambiguous ownership,
replacement, collision, partial output, or inability to prove cleanup fails
closed. Primary and restore failures remain distinct.

The helper does not self-remove during deploy, readback, capture, or restore.
The transport first receives, validates, and retains the complete restore
response; it then performs the separate `finalize_helper` exchange. This avoids
claiming that flushing a remote stdout stream proves local durability. The
local repository retains and seals the exact helper source bytes and digest.

### OpenSSH client transport

`tooling/e055_openssh_transport.py` constructs every local subprocess with an
argv tuple and `shell=False`. It ignores user SSH configuration with
`-F /dev/null` and explicitly sets the reviewed options: strict host checking,
one external known-hosts file, no global known-hosts file, no forwarding, no
TTY, no local command, bounded connect/alive timeouts, and non-interactive
authentication. Endpoint, port, helper path, fingerprints, identity digest,
binary paths, and external known-hosts file are validated before any subprocess
starts.

Operational mode permits exactly `/usr/bin/ssh` and `/usr/bin/sftp`. Both files'
path, mode, size, and SHA-256 are independently pinned and sealed. Fake client
paths require an explicit test-only constructor and are rejected by operational
transport-evidence validation. Helper staging uses a separate exclusive
`mkdir` step; rollback deletes only a staging directory whose successful
creation proved ownership.

The transport verifies the expected fingerprint against the external
known-hosts material before helper upload, records the local OpenSSH executable
realpath/hash/size/mode and sanitized version, and records the exact non-secret
client option allowlist and its canonical digest. It sanitizes all raised
errors; stdout and stderr from OpenSSH are bounded and are never copied into an
exception or evidence without exact schema validation.

### Runtime provenance

The transport evidence schema advances to version 2. Both deploy and readback
responses independently bind:

- helper remote path, SHA-256, size, mode, device, inode, and protocol version;
- `/usr/bin/python3` realpath, version, SHA-256, size, mode, device, and inode;
- board identity digest and pinned host-key fingerprint;
- operation sequence, request ID, and request nonce.

Static implementation evidence binds the exact local helper source, local
OpenSSH executable identity and version, and reviewed option allowlist. The
executor accepts this evidence only through the canonical transport protocol;
the same document is sealed into the phase manifest and every successful
runner record.

## Failure behavior

No constructor or default executor path contacts a target. Deployment begins
only from an explicit `execute()` call on an explicitly injected transport.
Unknown hosts, fingerprint mismatch, board digest mismatch, lock collision,
deployment collision, partial helper transfer, stale or forged observations,
response truncation, timeout, disconnect, remote nonzero exit, and signal exit
all fail before qualification. Exact available run bytes are retained through
the existing reservation-only local scaffold. Restore always runs after an
attempt that entered the transport lifecycle; a restore error never replaces a
primary error.

## Verification strategy

Helper tests call the real protocol handler against an isolated temporary root
and real files/processes. Transport tests use executable fake `ssh`, `sftp`, and
`ssh-keygen` boundaries to exercise the actual subprocess argv/stdin parser
without network access. Literal adversarial fixtures cover shell metacharacters,
newlines, traversal, wrong fingerprints and board digests, collisions, partial
transfer, forged readback, disconnect, remote nonzero/signal, cleanup, secret
redaction, and the disabled default. Existing E049d/E053/E055 regression,
publication, privacy, and English gates remain required.

Official behavior references are the current OpenBSD
[`ssh(1)`](https://man.openbsd.org/ssh),
[`ssh_config(5)`](https://man.openbsd.org/ssh_config), and
[`sftp(1)`](https://man.openbsd.org/sftp) manuals.
