# Independent review of PR 2088

Candidate: https://github.com/amdpilot-org/sglang/pull/2088 at exact commit `fe3b1d60e9c3f1c5e535606e0765f1efa272064f`

Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2121

## Verdict

Recommendation: **request changes**. The candidate is a meaningful partial fix, not a complete proof of the original issue's KV-provenance boundary.

It fixes the recorded-base idle completed-prefix failure (the candidate regression changes a four-token post-pause match from 4 to 0), resets the complete radix and allocator namespace, and handles the ordinary live disaggregated-PREFILL `chunked_req` fixture by aborting sender ownership before retract/requeue. Its focused suite passed: 36 tests and 2 subtests.

Two independent adversarial cases are not handled:

1. The sender retirement is guarded by `chunked_req not in retract_reqs`. If a live chunk is also represented in `running_batch`, it is retracted and the global cache/pools are reset without calling `clear_pending_chunk_send`, `sender.abort`, or releasing its metadata slot. The adversarial fixture observed zero sender-cleanup calls. Normal scheduling generally separates the active chunk from `running_batch`, so this is an invariant/race boundary rather than a reproduced normal-loop state; a real multi-node run was unavailable.
2. `disagg_kv_sender.abort()` is not protected. A transport-side exception escapes `pause_generation` before `retract_all`, `tree_cache.reset`, and allocator clearing. The adversarial fixture raised `RuntimeError("transport unavailable")`, so retract did not complete its cache-release boundary. Existing `_retire_aborted_prefill_result` treats sender notification as best effort and continues local retirement, which highlights the inconsistency.

The first is especially hazardous because cache/pool clearing proceeds while sender ownership was not retired. The second means a sender failure prevents the original contract from being established at all. Candidate prose and its happy-path mock do not cover either case.

## Evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate's two issue-specific regressions both failed: the idle prefix remained matchable 4/4 and live PREFILL teardown was absent.
- At exact candidate commit, imports resolved to `/job/repo/python/sglang/...`.
- Candidate focused tests passed: `36 passed, 2 subtests passed`.
- Independent adversarial run: `2 failed, 46 passed, 4 subtests passed`; the failures are the two cases above.
- No native source changed and the prepared environment declares no native artifact, so native rebuild was not applicable.
- One AMD Instinct MI350X, `gfx950`, was visible under Torch `2.11.0+rocm7.2` / HIP `7.2.26015`; a real GPU tensor calculation returned 14.0. The issue-specific scheduler/cache tests are CPU deterministic fixtures.

Raw commands/output and the exact adversarial test are retained under `reports/j-7356641494e2/evidence/`.

## Limitations

No Qwen weights were available, so the reported old/new-weight logprob and generated-text divergence were not reproduced. No multi-node disaggregated deployment was available, so real NIXL/Mooncake/MORI in-flight transfer cancellation and the synthetic running-batch overlap boundary remain unverified on transport hardware. The deterministic tests establish control-flow and cache-state behavior, not distributed transport completion.
