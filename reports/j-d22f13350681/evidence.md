# Independent review of candidate d184e1cb9c82e4d4e062000539d8fe0c92e39e1a

Upstream issue: https://github.com/sgl-project/sglang/issues/31861

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2195

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2298

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2263

## Finding

Recommendation: **request changes**. The candidate is a verified partial fix, not a full fix for the original issue.

The Python change makes `DeepSeekV2MoEGate.forward` return FP32 logits on non-CUDA devices for both BF16 and FP32 router weights. Its focused tests pass, and an independent random-input comparison exactly matched `F.linear(hidden.float(), weight.float())` for both weight dtypes.

The original issue also requires the AMX TopK boundary to prefer or enforce FP32 logits. The candidate does not change `biased_grouped_topk_cpu` or its native implementation. An independent call-boundary test demonstrated that the Python wrapper still forwards a BF16 `gating_output` unchanged, while `python/sglang/kernels/aot/csrc/cpu/topk.cpp` still dispatches BF16 as an accepted input type. The adversarial logits `1.001` and `1.002` both round to `1.0` in BF16, so accepting BF16 at this boundary preserves the information-loss failure for callers outside the corrected gate.

The candidate also does not add the optional specialized BF16-activation x FP32-weight AMX GEMM. The existing FP32 router-weight fallback is semantically correct but unaccelerated.

## Reproduction quality

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate regression did fail, but not with the candidate report's claimed `torch.bfloat16 != torch.float32` assertion. On this prepared environment it reached `get_exec()` and failed because the runtime `exec` namespace had not been published. Thus the checked-in test is passing-after coverage, but its claimed failing-before evidence is not portable to the prepared base and does not directly force the AMX branch.

Source inspection independently confirms the original AMX mechanism: `weight_packed_linear` allocates output with `mat1.options()`, so BF16 activations produce BF16 output. No native source was changed by the candidate, so no native rebuild was required or performed.

## Environment limitations

The host CPU is an AMD EPYC 9965 and has no Intel AMX. The installed `sgl_kernel` does not expose `weight_packed_linear` on this host, so actual AMX execution and the packed-weight loading lifecycle remain unverified. Repository Python source was confirmed loaded from `/job/repo`; the native Python package was loaded from the prepared environment's installed egg.

The assigned GPU is one AMD Instinct MI350X (`gfx950`). As a numerical reference only, `linear_bf16_fp32` returned FP32 and matched CPU FP32 matmul of identical BF16 operands with maximum absolute error `2.384185791015625e-07`. This is not an AMX, full-model, serving, semantic-accuracy, or distributed-workload reproduction. No model weights were available.

Raw outputs are retained under `reports/j-d22f13350681/evidence/`.
