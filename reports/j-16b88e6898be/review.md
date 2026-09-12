# Independent review of PR 1794

Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1830

Candidate: https://github.com/amdpilot-org/sglang/pull/1794 at `7fe1b5433ded03af468fe10969159eae92792666`

Recommendation: **request changes**. The change is a useful partial fix, but it does not fully establish the retract pause as an old-weight KV invalidation boundary.

## Findings

1. **Blocking: an idle retract pause leaves completed-request KV matchable.** The candidate puts final eviction inside `retract_all()`, while `Scheduler.pause_generation()` calls `retract_all()` only when `retract_reqs` is nonempty. I inserted a real four-token `RadixCache` prefix representing cache left by a completed request, invoked scheduler retract pause with no active requests, and rematched all four tokens afterward. A weight update performed during such an idle pause can therefore still reuse old-weight KV.

2. **The active-request path is genuinely improved.** On the recorded base, the candidate regression produced 3 failures and 2 passes, including a real radix prefix remaining matchable. At the exact candidate commit, its regression plus related scheduler and lock tests produced 37 passes and 2 passing subtests. This supports a partial fix rather than test-only hardening.

3. **Hierarchical invalidation is not demonstrated.** `HiRadixCache.evict()` may demote or stage entries to host under write-back policy, whereas `reset()` explicitly clears the cache controller and host pool. The candidate invokes generic `evict()` and has no hierarchical-cache regression proving old KV cannot be loaded back. This remains an unverified and source-supported counterexample risk.

4. **Disaggregated prefill remains outside the guarantee.** Existing scheduler code intentionally retains a live mid-chunk request in disaggregated-prefill mode and documents that a weight-update pause leaves stale-weight prefix KV. The candidate does not alter that path.

## Environment and scope

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; no image/base discrepancy was observed. Candidate imports resolved from `/job/repo/python/sglang`, including `/job/repo/python/sglang/srt/managers/schedule_batch.py`. No native files changed, so native rebuilding was inapplicable.

One AMD Instinct MI355X/gfx950 was visible through Torch 2.11.0+rocm7.2 and executed a simple device tensor operation. The issue-specific tests are scheduler/cache ownership tests and ran on CPU. The reported Qwen weights were unavailable, so this review does not claim a model weight-swap, semantic-output, multi-GPU, or multi-node reproduction.

Raw commands, output, the candidate diff, and the independent adversarial test are retained under `reports/j-16b88e6898be/evidence/`.
