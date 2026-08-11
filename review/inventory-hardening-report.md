# Inventory hardening review

## RED evidence

Added regression coverage for Docker failure propagation and cleanup, malicious
image rejection, container hardening flags, license redaction requests,
mandatory target-command failure cleanup, empty Docker binary validation, and
unchanged overwrite behavior.

Before the shell changes, the focused command:

```text
python3 -m unittest tests.test_inventory_scripts -v
```

failed 5 of 6 tests. The failures showed status `0` instead of Docker status
`37`, missing hardening flags, acceptance of `--privileged`, raw
`licence.txt` output, and swallowed mandatory `uname` failure.

## GREEN evidence

After implementation:

```text
bash -n tooling/inspect_acuitylite.sh tooling/target_inventory.sh
python3 -m unittest discover -s tests
................................
----------------------------------------------------------------------
Ran 32 tests in 9.606s

OK
```

The Acuity inspector now validates image and Docker-binary inputs, uses the
requested container restrictions, hashes and conservatively redacts the
license marker, and commits output only after successful Docker completion.
The target inventory now distinguishes required commands from optional probes,
writes both artifacts in a sibling temporary directory, and renames the
directory only after all required writes and hashes succeed.

## Concerns

The license behavior is covered at the script-source level plus a Docker stub
fixture, because the stub cannot execute the container's inner shell or mount
the SDK archive. A real container fixture should be added when a test image
with representative SDK paths is available.
