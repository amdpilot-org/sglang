# Investigation evidence

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2263
Independent review PR: https://github.com/amdpilot-org/sglang/pull/2364

The candidate's gate correction is retained. Independent reproduction showed that
`biased_grouped_topk_cpu` still accepted a BF16 `gating_output` and dispatched it
unchanged. The focused boundary regression failed on the candidate with
`ValueError not raised`; see `evidence/failing-before-candidate-topk.txt`.

The consolidated correction rejects non-FP32 router logits at the Python CPU
TopK boundary. This is an enforcement check rather than a cast: values such as
1.001 and 1.002 both round to 1.0 in BF16, so an upcast at TopK cannot recover
their ordering. The candidate regression was also repaired to force the old
AMX branch by mocking `use_intel_amx_backend=True`, avoiding the unpublished
runtime-context failure identified by review.

Focused and adjacent tests pass (14 passed); see
`evidence/passing-after-focused.txt`. On the assigned gfx950, the independent
`linear_bf16_fp32` reference returned FP32 and matched CPU FP32 matmul with max
absolute error 2.384185791015625e-07; see `evidence/gpu-reference.txt`.

The host is AMD EPYC 9965 and has no Intel AMX. No native code changed, no native
rebuild was applicable, and no claim of actual AMX execution is made. The
optional specialized BF16-activation x FP32-weight AMX kernel remains absent;
FP32 router weights use the semantically correct FP32 fallback. No affected
model weights were available, so no full-model or serving claim is made.
