# Stage 1 review rejection — commit 51d1c1c

Status: **REJECTED EVIDENCE; PRESERVED FOR AUDIT**.

The first Stage 1 publication at commit `51d1c1c` was rejected because:

1. It compared total hot and cold windows without normalizing by traversal
   count. A 250-call hot window and one-call cold window were therefore
   incomparable.
2. The cold CLI allowed explicit iteration/budget settings that could turn the
   measured region hot.
3. `packed_stream` used a serial hash per byte, and `unpack_scale` performed
   repeated per-block reductions/stores. Those artifacts could dominate the
   intended controls.
4. The sample validator accepted missing synchronization, empty PMU events,
   insufficient provenance, and other unqualified inputs.
5. The experiment documentation was not consistently English.

The original commit remains in Git history. None of its performance semantics
are accepted as target evidence. The subsequent revision adds adversarial
regressions for each defect and still performs no board workload.
