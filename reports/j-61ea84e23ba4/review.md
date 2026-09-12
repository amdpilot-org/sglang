# Independent review of PR 938

Upstream issue: https://github.com/sgl-project/sglang/issues/37892

Mirror issue: https://github.com/amdpilot-org/sglang/issues/969

Candidate: https://github.com/amdpilot-org/sglang/pull/938 at `f21c189be8eed6f307b07179764f756af2ef94a2`

Recommendation: **accept**, with `fully_resolves_original=false`.

The recorded base contains the reported routing contradiction. DSV4 prefill allocates `c4_sparse_raw_indices`, but `run_topk_transform` requires `raw_indices is None` before selecting v2, so the reported paged-prefill state cannot reach v2. The candidate is based directly on the recorded base and narrowly removes that contradiction. When raw indices are required, it asks v2 for logical indices and independently translates them through the page table; otherwise it retains the existing fused v2 transform.

The candidate's three focused regressions passed. Two existing tests exercised the real v2 JIT kernel on the assigned GPU. An independent adversarial script ran the real kernel at sequence length 32768 and matched `torch.topk` as an unordered selection; its separate translation checks matched an independent `torch.gather` reference for a non-power-of-two page size, repeated pages, invalid padding, and noncontiguous outputs.

This is strong evidence for the narrow dispatch fix, not proof of the complete original serving failure. The environment had one AMD Instinct MI350X (gfx950-class), not four NVIDIA GB300 GPUs, and had no DeepSeek-V4-Pro weights. Consequently the original CUDA illegal address was not reproduced before the candidate and could not be shown absent afterward. FP4 model execution, DP attention, hierarchical cache, speculative decoding, multi-GPU behavior, and GB300 performance remain unverified.

No native source changed, and the prepared environment declared no native rebuild target. Import inspection confirmed `sglang`, the changed indexer, and top-k Python wrapper loaded from `/job/repo`; `sgl_kernel` loaded from the prepared installed package. A nearby existing ROCm test still fails its `dsa_drop_wide_page_table` expectation for the sgl-kernel subcase, while its flashinfer subcase passes; it does not exercise the candidate helper and is recorded as a limitation rather than evidence against the patch.

Raw evidence was retained outside the checkout at `/job/review-evidence-j-61ea84e23ba4/` while revisions were switched.
