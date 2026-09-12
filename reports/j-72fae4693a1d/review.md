# Independent review of PR 3467

Recommendation: **accept**. Candidate commit `eb9138b0b6a1f795ce4b7090d4242dbf444e9f63` fully resolves the original issue on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/35891

Candidate provenance issue: https://github.com/amdpilot-org/sglang/issues/3440

Review deliverable issue: https://github.com/amdpilot-org/sglang/issues/3470

Candidate: https://github.com/amdpilot-org/sglang/pull/3467

## Finding

The prepared checkout exactly matched the required recorded base. That base already filters completed futures when `_dispatch_group` begins and before each non-batchable per-request encode. It still had a reproducible race after initial filtering: a future completed while its image/audio group waited for `encode_dispatch_lock` remained in the precomputed `requests` list and reached both TP broadcast and `batch_encode`.

The candidate adds the missing liveness check under `encode_dispatch_lock`, returns if the group is empty, and rebuilds `requests` from the surviving group before any broadcast or encoder execution. There are no native changes.

## Executed evidence

- Failing before: 2 focused cases passed and the post-lock case failed; `batch_encode` saw `['completed', 'live']`.
- Passing candidate: the same three regressions passed.
- Independent adversarial cases passed for mixed cancelled/completed/live work with decoded two-socket payloads, an entirely abandoned batch, and cancellation during a prior per-request encode.
- Full scheduler module: 17 passed.
- Adjacent encoder server and health modules: 75 passed, 4 subtests passed.
- The interpreter imported the candidate runtime from `/job/repo/python/sglang/srt/disaggregation/encoder/runtime.py`.

Raw commands and output are retained under `raw/`; structured claims are in `result.json`.

## Limitations

This review used CPU test doubles at the production Python broadcast and encoder boundaries. It did not launch real tensor-parallel workers, an HTTP server, model weights, or GPU execution. That is sufficient for the original scheduler race, but it does not validate distributed collective behavior or model semantics. No native source changed, so a native rebuild was not applicable.
