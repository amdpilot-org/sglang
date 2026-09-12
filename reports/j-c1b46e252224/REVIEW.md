# Independent review of PR 3125

Candidate: https://github.com/amdpilot-org/sglang/pull/3125 at `22a6bcf1d48f7331c4090cc0e6da49f320e9de92`

Upstream issue: https://github.com/sgl-project/sglang/issues/35808

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3137

Recommendation: **request changes**. The candidate is a meaningful partial fix, but it does not fully resolve the original full-lifecycle observability request.

## Verified fixes

The candidate regression failed three-for-three on the untouched recorded base and passed three-for-three at the exact candidate commit on an AMD Instinct MI355X. This verifies the narrow UnifiedRadixCache correction for:

- host-capacity rejection metrics and request-correlated logs;
- successful target+draft GPU-to-host-to-GPU retraction metrics and logs, with exact restored K/V comparisons;
- injected retraction submission exceptions producing a bounded `transfer_error` metric and rid log before re-raise.

The candidate's collector suite and focused UnifiedRadixCache load-back suite also passed.

## Remaining original-contract failures

Independent adversarial checks exercised the other supported HiCache implementation, `HiRadixCache`:

- `loading_check` re-raises asynchronous L2-to-L1 completion exceptions with no transfer-failure metric;
- `writing_check` re-raises asynchronous L1-to-L2 completion exceptions with no transfer-failure metric;
- successful L2-to-L1 completion emits an aggregate metric but no request-correlated terminal `[HICACHE]` record, because `ongoing_load_back` retains only the node rather than the rid;
- `init_load_back` still logs L1 as `result=complete`, not hit/miss with hit tokens, and logs L2-to-L1 only as `scheduled` rather than a request-correlated terminal outcome.

These are concrete lifecycle, failure-classification, and rid-tracing gaps from the original issue, not unavailable-backend speculation. Therefore `fully_resolves_original` is false.

## Environment and scope

Imports resolved to the checked-out candidate under `/job/repo/python`. The prepared AITER module loaded from the private runtime cache. No native source changed, so no native rebuild was applicable. Testing used torch 2.11.0+rocm7.2, HIP 7.2.26015, and one gfx950 AMD Instinct MI355X.

Mooncake, HF3FS, NIXL, UMBP, multi-node, TP/PP distributed paths, model-specific storage backends, and NVIDIA/CUDA were unavailable. The GPU test validates cache transport and exact synthetic K/V restoration, not HTTP/model semantic accuracy.

Raw logs and standalone adversarial scripts are retained outside the checkout at `/job/review-evidence-j-c1b46e252224` so they survived revision switching.
