# Independent review of PR 1153

Candidate: https://github.com/amdpilot-org/sglang/pull/1153 at exact commit `20e6fa7d408c689ffc3753969396a745d5c462b2`

Upstream issue: https://github.com/sgl-project/sglang/issues/37712

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1191

## Verdict

Recommendation: **request changes**. The candidate is a substantive partial fix, not test-only hardening: it activates a conservative logits budget and changes the chunked plan path from batch-concatenated K to request-local K. This fixes both counterexamples recorded in the review of PR 985, including the fused PAGED global-table/global-row-index mismatch.

It does not fully enforce its computed allocation budget. `_topk_ragged_kpool_grouped` uses `max(1, logits_budget_bytes // (group.k_rows * 4))`, so when one request-local fp32 logits row is wider than the budget it still calls `fp8_mqa_logits` for that entire row. The independent GPU adversarial case measured a 40-byte output allocation with a 4-byte budget. This is also acknowledged in the candidate prose, but means the original OOM contract is only partially resolved rather than fully bounded.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the candidate's two focused files produced 7 failures and 1 pass. Failures include the unbounded returned budget, no plan-path chunking, a 240-byte concatenated logits allocation against a 24-byte budget, and a 240-byte allocation against a 48-byte PAGED case budget. See `base-regression.log`.
- Exact candidate `20e6fa7d408c689ffc3753969396a745d5c462b2`: the submitted focused regression produced 8 passes. See `candidate-regression.log`.
- Independent boundary plus the pre-existing adjacent logits-budget tests produced 4 passes. The boundary test deliberately asserts and observes the remaining request-local overshoot (40 allocated bytes versus a 4-byte budget). See `adversarial-and-adjacent.log`.
- Imports resolved to `/job/repo/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py` and `/job/repo/python/sglang/srt/layers/attention/dsa/kpool_plan.py`; the new grouped method was present at the candidate revision. See `import-paths.log`.

## Environment and limits

Execution used one AMD Instinct MI350X (`gfx950`) with PyTorch 2.11.0+rocm7.2 and HIP 7.2.26015. Real FP8 and metadata tensors were allocated on the GPU, but DeepGEMM and top-k were mocked by the focused orchestration tests. The candidate changes Python only, so no native rebuild was applicable.

No GLM-5.3-Flash or RadixArk weights, four-B300 CUDA TP=4/EP=4 system, NVIDIA DeepGEMM execution, hierarchical-cache traffic, or DFLASH serving workload was available. Consequently, the reported production workload and end-to-end semantic behavior remain unverified. The evidence supports the two reviewed orchestration corrections, but not a claim that every original-issue allocation is bounded.
