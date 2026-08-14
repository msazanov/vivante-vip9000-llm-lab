# Stage 1 fourth review rejection — commit 7679828

Status: **REJECTED EVIDENCE; PRESERVED FOR AUDIT**.

The revision at commit `7679828c8d0891a8d42374279e096b09db87e0f5` is not
accepted for target execution. The third independent rereview found that its
analyzer still accepted caller-created sample mappings. A caller could alter a
measurement, recompute the unkeyed canonical `harness_result_sha256`, and
produce an internally consistent object without preserving immutable raw E055
or E049c output.

The replacement gate must accept only raw artifacts sealed in the containing
Git commit and tree. It must recompute every file hash and size, parse exact
E055/E049c/runner streams, reject index/worktree/path/role substitution, and
derive all statistical rows internally. A complete mapping without a committed
raw bundle is not measurement evidence.

This rejected revision contains no target or model workload. It must not be
used for a bottleneck, performance, or optimization-promotion claim.
