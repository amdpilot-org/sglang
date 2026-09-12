# Independent review of amdpilot-org/sglang PR 542

Upstream issue: https://github.com/sgl-project/sglang/issues/38452

Mirror issue: https://github.com/amdpilot-org/sglang/issues/629

Candidate: https://github.com/amdpilot-org/sglang/pull/542 at `4728bc00c490439134d4ddcc2782ad291effb5d9`

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Reject. The candidate is test-only hardening for an already-working fallback and does not fix or reproduce the original backup-only-stub failure. It changes no production or native source.

## Findings

The original issue requires a tree node that survives L1 and L2 eviction as `evicted && backuped`, so matching counts its tokens while no host indices remain. That state is not representable in this prepared base:

- `UnifiedTreeNode.backuped` is derived from the Full component's non-null `host_value`.
- `_evict_host_leaf` frees all component layers and removes the leaf from its parent.
- The prepared-base `test_hicache_host_leaf_eviction` passed on GPU and confirmed that behavior dynamically.

The candidate's new test acknowledges this different behavior and then tests something else. After host eviction it asserts `partial.last_host_node == partial.best_match_node`, manually slices the absent suffix, and directly calls `cache.prefetch_from_storage`. Its complete-eviction case likewise asserts a root anchor and directly prefetches the whole sequence. Those are live-L2-anchor and fresh-tree/root-anchor paths, not the matched backup-only-stub path described by the issue.

The direct call also bypasses `scheduler._prefetch_kvcache`, where the issue reports `matched_len` consuming the stub span and producing an empty storage query. Setting `prefetch_threshold=1` further avoids the original default-threshold contract. The restored 23/24 KV values and unavailable-key check are valid evidence that the file backend round trip works, but the issue already states that the backend works after `flush_cache`; they do not prove stub-anchored recall.

## Execution evidence

On the exact candidate, the three page-size variants plus the existing L3 round trip and host-leaf eviction test all passed (5 tests, 3.344 seconds). Imports resolved to `/job/repo/python/sglang/...`; the production diff count under `python/` was zero. Torch reported one AMD Instinct MI355X, gfx950, using torch 2.11.0+rocm7.2, and the tests loaded the prepared AITER native library from `/tmp/amdpilot-repo-j-125933c91a43/cache/aiter/module_aiter_core.so`.

No native source changed, so a native rebuild was not applicable. This environment differs from the reported NVIDIA L40S/CUDA 13 setup. No server/model-scale reproduction was attempted because the required state is absent from this revision's tree semantics.

Raw review evidence was preserved outside the checkout in `/job/review_evidence/` while revisions were switched.
