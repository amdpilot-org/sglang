# Independent review of PR 2597

Candidate: https://github.com/amdpilot-org/sglang/pull/2597 at exact commit `eb5294ff4b3d4885053941fcfe8aa5f077897052`

Upstream issue: https://github.com/sgl-project/sglang/issues/31766

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2601

## Conclusion

Recommendation: **accept**. The candidate fully resolves the original cache-driven descriptor exhaustion in the exercised host-side ZMQ contract. It is a source fix with regression coverage, not test-only hardening.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real PyZMQ probe under soft `RLIMIT_NOFILE=128` failed at 38 distinct endpoints with `ZMQError: Too many open files`, leaving 38 PUSH sockets and 38 monitors cached and 127 descriptors open. On the exact candidate, the same probe completed 100 connections while retaining 21 sockets/monitors and 76 descriptors.

The earlier active-sender counterexample is also corrected. An independent probe filled a four-entry cache, acquired every endpoint send lock, and attempted a fifth endpoint. The fifth connection remained blocked with the cache at four and 25 descriptors; after one lock was released it completed, the thread exited, and the cache remained at four.

## Source and build validation

The candidate changes only Python files (`conn.py`, `environ.py`, tests, and reports). No C/C++/HIP/native source changes, generated native artifacts, or extension changes are present, so a native rebuild is not applicable. With the candidate checked out, `inspect.getsourcefile` resolved both reviewed modules to `/job/repo/python/sglang/...`, confirming the prepared interpreter loaded checkout source rather than an installed copy.

## Tests

- Base real-PyZMQ low-FD probe: reproduced `ZMQError: Too many open files`; 38 completed/cached endpoints and 127 open descriptors.
- Candidate real-PyZMQ low-FD probe: completed 100 endpoints; cache and monitor maps bounded at 21; 76 open descriptors.
- Candidate regression file: `9 passed`.
- Independent active-sender backpressure probe: blocked while all four entries were busy, completed after release, cache stayed at four, no exception.
- Environment unit tests: `13 passed, 2 subtests passed`.

Raw logs, candidate diff, issue snapshots, and standalone probes are retained outside the checkout in `/job/evidence-j-12e0674a3f17/`, `/job/upstream-31766.json`, and `/job/candidate-2597.json` so revision switches did not overwrite the evidence.

## Limitations and residual risks

The prepared host is AMD/ROCm 7.2 rather than the reporter's 16x NVIDIA H800 CUDA deployment, and the production multi-node/full-model workload and weights were unavailable. No GPU execution was needed for this CPU/PyZMQ descriptor-lifecycle defect, so no GPU or end-to-end serving claim is made.

An operator can explicitly set `SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE` above the process descriptor budget, and unrelated code can consume descriptors after the default is calculated. Those are process-wide/configuration exhaustion cases, not counterexamples to the reported unbounded default endpoint cache. The candidate's one-second `EMFILE` retry only covers transient libzmq retirement; persistent unrelated process-wide exhaustion still raises honestly.
