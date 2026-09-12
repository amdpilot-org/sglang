# Independent review of amdpilot-org/sglang PR 2832

Candidate: `3f2279024311a27aadc133ac24d0d53c8e37989a`

Upstream issue: https://github.com/sgl-project/sglang/issues/31261

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2777

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2871

## Verdict

Recommendation: **accept as test-only hardening**. This is not a full original-issue fix, so `fully_resolves_original` is false.

The candidate changes two test files and adds its own report; it changes no runtime or native implementation. The recorded base already contains the earlier batch-invariance implementation associated with merged upstream PR 27869. I extracted the candidate's new kernel test outside the checkout and ran it against the exact base before checking out the candidate. It passed there unchanged (`2 passed, 11 subtests passed`). Therefore the candidate does not establish a failing-before/passing-after repair and must not be credited as the implementation that fixes the open feature request.

At the exact candidate, the focused test also passed. Independent adversarial GPU coverage extended the candidate's one BF16 sample to BF16 and FP16, six batch sizes, and first/middle/last shared-row positions. All 36 comparisons were bitwise equal and close to the FP32 PyTorch reference. Deterministic launch geometry remained four rows per block across 14 boundary sizes. For comparison, forcing the historical adaptive decision selected one row per block for smaller batches and four for larger batches, confirming that the policy-level condition under test is real.

## Source and environment evidence

- Prepared base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference from the image-prepared checkout.
- Candidate was temporarily checked out detached at the exact requested commit and the checkout was returned to `amdpilot/j-1dccb1d575aa` before this report was committed.
- SGLang imported from `/job/repo/python/sglang`; gated normalization imported from `/job/repo/python/sglang/kernels/ops/attention/fla/layernorm_gated.py`.
- Interpreter: `/tmp/amdpilot-repo-j-1dccb1d575aa/venv/bin/python`.
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`.
- GPU: one AMD Instinct MI355X, `gfx950:sramecc+:xnack-`.
- Hugging Face configuration confirms `Qwen/Qwen3.5-27B` declares `Qwen3_5ForConditionalGeneration`, 48 linear value heads, and value/key head dimension 128, matching the candidate's kernel shape rationale.
- No C++, FlyDSL, or other native source changed. No native rebuild was applicable.
- Raw logs, fetched metadata, the exact extracted test, and the adversarial probe are preserved at `/job/review-evidence-j-1dccb1d575aa/` outside the checkout.

## What remains unverified

The original report asks for deterministic generation from the actual Qwen3.5-27B conditional-generation model with TP=2. This node has one GPU and no Qwen3.5-27B weights. Consequently the collected serving regression was not run, and the review does not qualify TP=2 collectives, real scheduler batching, repeated generation/logprob equality, the 262144-token context path, or the parser and served-name options in the report. The local AMD result also cannot establish NVIDIA H100 behavior. The deterministic tiny Llama fixture was not used because it would validate transport and engine execution only, not the requested architecture or distributed contract.

The candidate's use of `rms_norm_ref` is a useful algorithmically separate FP32 PyTorch comparison, but the helper is imported from the same implementation module. Its description as an "independent" reference should be read with that limitation.
