# Rejected E055 transport-provenance revision

Status: **REJECTED**. This file preserves the independent review result for
commits `88bc2b7` and `eca4b03`; it is not qualification evidence.

The adversarial RED checkpoint reproduced five failures in the committed
runtime-evidence design:

- `StrictHostKeyChecking=accept-new` was accepted;
- `ConnectTimeout=0` was accepted;
- two matching `operationally_trusted` endpoint claims were accepted without
  independent pins;
- recomputed helper and OpenSSH hash claims were accepted when the remote
  runtime claim was changed to match;
- the outer raw and runner schemas remained v1 while containing transport v2.

`ProxyCommand` was already rejected because it was outside the old option-key
allowlist. The other cases required production changes. The replacement uses
exact fail-closed OpenSSH option semantics, caller-supplied immutable endpoint
and implementation pins, direct hashing of local helper/OpenSSH bytes, a clean
HEAD/index/worktree binding for the helper Git blob, and an explicit outer-v2
boundary that rejects v1 target-phase records.

No target, board, or model workload was executed while producing this review
evidence or its fixes.
