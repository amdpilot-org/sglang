# Independent review of amdpilot-org/sglang PR 2481

Candidate commit: `02a233ac466039ac4804f719a37382309f3afa37`

Upstream issue: https://github.com/sgl-project/sglang/issues/29562

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2434

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2516

## Verdict

Recommendation: **accept as a narrow partial fix**.

The candidate fixes the original, explicitly reported `3072` versus `6144`
weight-load mismatch. It does not fully resolve the original issue's practical
goal of serving GLM-5.2-NVFP4 on eight RTX PRO 6000 (SM120) GPUs. The issue
discussion documents a subsequent independent SM120 DSA failure in released
FlashInfer/DeepGEMM dependencies after the loader mismatch is bypassed.

## Evidence

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
published ModelOpt exclusion pattern reports the shared expert as excluded and
the routed experts as included, but
`GlmMoeDsaForCausalLM.shared_experts_fusion_disable_reason` returns `None`.
The assertion that mixed-precision experts must disable fusion therefore fails.
This reproduces the faulty loader decision that remaps an unpacked shared
expert into packed FP4 routed-expert storage.

At the exact candidate commit, the same check returns the new mixed-precision
disable reason. The candidate regression passed (`4 passed`, with 32 unrelated
cases deselected), and the existing GLM-5 NextN ModelOpt suite passed (`3
passed, 2 subtests passed`).

Independent adversarial checks also passed for:

- the reported wildcard exclusion;
- an exclusion affecting only one sparse layer;
- an exclusion confined to a pre-MoE dense layer (must not disable fusion);
- both shared and routed experts excluded together (no precision mismatch);
- uniform FP4 experts; and
- `--enforce-shared-experts-fusion` not overriding the unsafe mismatch.

`py_compile` and `git diff --check` passed. The imported SGLang and GLM module
paths were under `/job/repo/python`, confirming that the checkout source, not a
separate installed SGLang copy, was tested.

No native files changed, so no native rebuild was required or performed.

## Environment and remaining scope

The prepared host has one AMD Instinct MI355X (`gfx950`) with PyTorch
`2.11.0+rocm7.2`. It does not have eight NVIDIA RTX PRO 6000 SM120 GPUs or the
GLM-5.2-NVFP4 weights. Consequently this review cannot independently validate
TP=8 model loading, NVIDIA NVFP4 kernels, DSA execution, CUDA graph capture, or
an HTTP generation request.

The remaining counterexample to a full original-issue resolution is the actual
SM120 serving path after weight loading: released dependency paths have been
reported to fail with `TllmGenFmhaRunner ... Unsupported architecture` or the
DeepGEMM paged-MQA metadata equivalent. The candidate changes only the fusion
gate and cannot address that architecture support gap. A full-resolution claim
therefore remains false even though the loader correction itself is verified.

Raw command output and JUnit XML were preserved outside revision switches in
`/job/review-evidence-j-ba33103216ea/`.
