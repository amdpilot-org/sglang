# Independent review of candidate e32bd205

Recommendation: **request changes**. The candidate is a useful, verified partial fix, but it does not fully resolve the original full-lifecycle observability request.

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2737 at exact commit `e32bd205e2a9ec107903f9b03da56802e4868200`

Upstream issue: https://github.com/sgl-project/sglang/issues/35808

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2694

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2769

## Findings

The candidate genuinely adds three bounded-label Prometheus families for UnifiedRadixCache prefetch scheduling outcomes and L3 query request/token classification. It also changes the L3 query and terminal prefetch messages to a request-correlated `[HICACHE] rid=...` format. The recorded-base failure was reproduced, the candidate's focused CPU tests passed, and its real GPU file-storage round-trip passed on the assigned MI350X.

That implementation is narrower than the original contract. The issue requests visibility across L1 GPU, L2 host, and L3 storage, including cross-layer transfer success/failure and request-level tracing across the hierarchy. The candidate's new classification is generated only from `UnifiedRadixCache._account_prefetch_outcome`, which classifies L3 presence as hit/partial/miss. Its structured events cover only `storage_query` and the terminal `storage_prefetch`. It does not add unified L1/L2 lifecycle events or cross-layer transfer success/failure totals. Legacy HiRadixCache does not feed the new snapshot at all.

The existing `storage_prefetch_unfulfilled_tokens_total{reason=...}` can distinguish some post-query storage-transfer failures at token granularity, but it predates the candidate and is not a complete substitute for the proposed full cross-layer request lifecycle. Consequently, the candidate should be described as an L3/UnifiedRadixCache observability increment, not a complete fix for the open feature request.

## Environment and evidence

The prepared checkout exactly matched the required base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image-checkout difference. Tests used `/tmp/amdpilot-repo-j-809204b0a14a/venv/bin/python`, Torch 2.11.0+rocm7.2, HIP 7.2.26015, and one AMD Instinct MI350X. Source imports resolved under `/job/repo/python/sglang`. No native source changed, so rebuilding native code was not applicable.

Raw logs, JUnit, import paths, and the full candidate diff were preserved outside the revision-switching checkout at `/job/review-evidence-j-809204b0a14a/`.

Unavailable and therefore unverified: Mooncake, HF3FS, NIXL, UMBP, multi-node, TP/PP distributed, other model/storage paths, NVIDIA/CUDA, and full HTTP/model semantic behavior. The GPU test qualifies only the direct file-backed HiCache path exercised by the candidate regression.
