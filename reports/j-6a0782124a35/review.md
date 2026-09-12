# Independent review of PR 639

Candidate: https://github.com/amdpilot-org/sglang/pull/639 at `2d784005479317580b5e07cff6c219a243ddc056`

Upstream issue: https://github.com/sgl-project/sglang/issues/38513

Mirror issue: https://github.com/amdpilot-org/sglang/issues/657

## Recommendation

Request changes. This is a useful partial fix, but it does not fully resolve the original issue.

The candidate correctly changes the CUDA capability API to return false for controlled SM 7.5 and 8.0 probes and true for SM 8.9 and 9.0. It also replaces the channelwise `_apply_fallback_scaled_mm` call to `torch._scaled_mm` with explicit FP32 conversion and `torch.mm`. That fallback passed the candidate regression and matched an independent NumPy reference exactly on the assigned AMD Instinct MI350X for both per-row and scalar activation scales.

The original failure class remains for per-tensor scales. `apply_fp8_linear` has a separate `per_tensor_weights and per_tensor_activations` branch that still calls `torch._scaled_mm` unconditionally. With `cutlass_fp8_supported=False`, scalar input and weight scales, and `_scaled_mm` forced to raise the unsupported-SM error, the exact candidate propagated that error. This is not merely an unrelated smoke: it is an FP8 linear dispatch path covered by the original contract.

The corrected `supports_fp8()` result also does not create an early refusal. A source audit found no production call sites for `supports_fp8()` under `python/`; only platform method definitions exist. Consequently, SM75/SM80 execution is not protected by that capability result in this checkout. The candidate also did not add the requested quantization documentation warning.

## Environment and architecture limits

The prepared interpreter imported SGLang from `/job/repo/python/sglang`, including `/job/repo/python/sglang/srt/platforms/cuda.py` and `/job/repo/python/sglang/srt/layers/quantization/fp8_utils.py`. Torch was `2.11.0+rocm7.2` with HIP `7.2.26015` on one AMD Instinct MI350X.

No NVIDIA GPU was assigned. The SM 7.5, 8.0, 8.9, and 9.0 results are controlled capability probes through the actual candidate platform API, not physical NVIDIA qualification. The MI350X executed the software fallback and a valid-layout native `torch._scaled_mm`; the latter returned finite float16 output. This verifies preservation on the assigned ROCm device only.

No native source changed in the candidate, `repository-environment.json` recorded `native: null`, and no native rebuild was applicable. There is no compiler/ISA claim to validate from this Python-only diff.

Raw evidence was preserved outside the revision-switching checkout at `/job/review-evidence-j-6a0782124a35`.
