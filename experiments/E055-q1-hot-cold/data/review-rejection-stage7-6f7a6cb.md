# Rejected revision `6f7a6cb`

Independent review rejected revision `6f7a6cb12013a69038d2bbac2f02b1305262e467`
before any target phase. It is not measurement evidence and does not qualify a
board or model result.

The review found two load-bearing defects: killing the per-request helper could
leave the capture process group alive without durable ownership metadata, and
the SFTP executable was neither fixed to `/usr/bin/sftp` in operational mode nor
bound by independent path/mode/size/hash pins. It also found three bounded
cleanup-order defects: helper finalization did not recheck device/inode before
unlink or cross-check the returned identity, the initial bootstrap operation
was outside owned-staging rollback, and local known-hosts/SSH/SFTP/helper
provenance was not validated before remote preparation.

The RED suite reproduced all findings locally:

- helper replacement after hash was unlinked rather than rejected;
- helper `SIGKILL` left no `owned-process.json` record while the capture PGID
  survived;
- operational configuration accepted injected SSH/SFTP executables;
- bootstrap had no separate ownership-proving staging step and its first
  failure did not enter cleanup state;
- finalization accepted a different helper device/inode;
- SFTP provenance was absent; and
- a forged local pin reached `prepare` and `readback` before rejection.

No board connection, target workload, model inference, or network operation was
performed while reproducing or correcting these defects.
