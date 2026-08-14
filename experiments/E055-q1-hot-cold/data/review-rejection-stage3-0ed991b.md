# Stage 1 third review rejection — commit 0ed991b

Status: **REJECTED EVIDENCE; PRESERVED FOR AUDIT**.

The revision at commit `0ed991b1b17c46f46b09ea9ed6731eb585cf60f5` is not
accepted for target execution. The second independent rereview found that:

1. PMU event names were checked, but their exact E049c raw config values were
   not qualified.
2. The caller could still inject an arbitrary expected provenance mapping
   instead of using immutable publication artifacts.
3. CPU affinity/migration and thermal numeric fields did not reject every
   Boolean or floating-point forgery.
4. Cache conditioning lacked complete byte/line/checksum coverage metadata;
   the exact 18-case golden and output checksum were not bound to a specific
   harness result and build.
5. The promotion path accepted a partial subset instead of requiring the
   complete documented seven-size, two-CPU, three-mode, two-cache-state and
   three-PMU-group matrix.

This rejected revision contains no target or model workload. It must not be
used for a bottleneck, performance, or optimization-promotion claim.
