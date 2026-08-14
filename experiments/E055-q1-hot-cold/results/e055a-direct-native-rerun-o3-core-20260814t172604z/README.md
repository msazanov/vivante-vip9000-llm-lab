# E055 PTY-contaminated direct phase

Status: **INVALID — stopped after sample 1/20**.

This fresh phase used the exact native PMU and harness binaries, but the SSH command
allocated a remote PTY. The root-assisted sample itself returned E049c status `ok`,
passed S/A/E, ratio 1, H/M/T timing, golden, checksum, observed CPU0, thermal, exit,
and cleanup checks. It is still invalid because the wrapper stderr role contains one
byte, exactly `0x0a`:

```text
size: 1 byte
SHA-256: 01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b
```

The live validator correctly rejected `wrapper.stderr.raw must be empty`, and no
sample 2 ran. The evidence is preserved rather than deleting the newline or relaxing
the qualifier.

The first remote-no-TTY identity attempt is also preserved under `controls/`. It
started no PMU child because the local executor supplied EOF on SSH stdin before a
live password write. Its 61-byte stderr states that sudo received no password. The
subsequent accepted method used a local echo-disabled PTY while retaining remote
`ssh -T`; its passing controls and fresh 20-sample result are published separately in
`../e055a-direct-native-rerun2-o3-core-20260814t173636z/`.

No model, NPU, or sample retry belongs to this invalid phase.
