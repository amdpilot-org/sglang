# Independent review of amdpilot-org/sglang PR 736

Upstream issue: https://github.com/sgl-project/sglang/issues/39173

Mirror issue: https://github.com/amdpilot-org/sglang/issues/767

Candidate: https://github.com/amdpilot-org/sglang/pull/736 at `2aea7cad31ab4ef5547ee7a23c6425220e41a65f`

Recommendation: **request changes**. The candidate is a useful partial fix, but it does not fully resolve the original Engram compact-ragged contract.

## Findings

1. On the prepared base, the actual layout helper reproduced the reported 42-token geometry: `build_capture_verify_lens(42, 8, 6)` returned `[6, 6, 5, 5, 5, 5, 5, 5]`. The candidate changes the capture slot count to `ceil(num_tokens / captured_req_width)`, making 42 tokens seven uniform width-six rows. This directly fixes the initial reported assertion geometry for the profiled 42-token tier.

2. On the prepared base, a real MI355X tensor reproduction of `DSV4RawVerifyMetadata.copy_` raised the same overlapping-memory `RuntimeError` as the issue's later first-forward traceback. At the candidate commit, verify and decode metadata copies produced correct values for identical views and forward/backward overlapping views. This validates the isolated alias-copy correction, though not the complete DeepSeek-V4.1 serving path.

3. The candidate does not solve the Engram contract for genuinely ragged capture tiers. Its own intended 43-token behavior is eight rows `[6, 6, 6, 5, 5, 5, 5, 5]`. The issue's `EngramHasher.forward()` target-verify branch requires one equal block per request. Therefore 43, 41, 47, and other non-multiple tiers remain counterexamples unless Engram hashing is made layout-aware and validated against per-request references. Silently forcing verify-all would not meet the requested semantics, and the candidate correctly does not do that, but it also does not provide the missing Engram-side support.

4. The prepared base and candidate have no `python/sglang/srt/layers/engram.py`, `EngramHasher`, or DeepSeek-V4.1 Engram integration. Consequently, actual hash context and hash tensor outputs could not be compared with an independent per-request reference. This is an architecture/source limitation, not proof of correctness.

## Validation summary

- Candidate regressions: 14 tests passed, with 5 subtests passed.
- Independent tier checks covered 1, 5, 6, 7, 11, 12, 13, 23, 24, 25, 41, 42, 43, 47, and 48 tokens at width 6 / max batch 8.
- Real `RaggedVerifyLayout` device outputs (`qo_indptr_device`, `extend_start_loc`) matched independent prefix-sum references for uniform and ragged request lengths.
- Real MI355X verify/decode metadata copies matched cloned references for identical and overlapping storage views.
- Imports resolved to `/job/repo/python/sglang/...`; no installed sglang wheel shadowed the checkout.
- No native source changed, so no native rebuild applied.

## Unverified deployment scope

The assigned machine has one AMD Instinct MI355X with torch 2.11.0+rocm7.2 / HIP 7.2.26015. It cannot reproduce the report's four-node NVIDIA GB10 TP4/EP4 DeepSeek-V4.1-Flash CUDA deployment. CUDA graph capture, real Engram weights/hashes, TP4/EP4 behavior, and end-to-end first-request serving therefore remain unverified.
