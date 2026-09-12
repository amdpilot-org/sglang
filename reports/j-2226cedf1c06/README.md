# ROCm router GEMM precision correction

Upstream issue: https://github.com/sgl-project/sglang/issues/34857

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1678

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1527 at
`034ffd7a3995b4cd941f38ee5398250be938b599`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1643

## Result

The independent review's remaining counterexample reproduced on one assigned
MI350X/gfx950. With the candidate implementation, the production
`(M,K,N)=(8,7168,256)`, fallback `(65,512,64)`, and skinny `(1,513,63)` outputs
were fp32 tensors whose every value was exactly representable in bf16. The new
precision regression failed on all three subtests.

The consolidated correction preserves fp32 router output and the prepared
base's existing GLM-5.2 correction-bias fixes, but uses ROCm
`torch.mm(..., out_dtype=torch.float32)` rather than aiter's cast-after-bf16
fallback. Afterward all three precision subtests pass. Every measured output
value is no longer bf16-representable, and maximum error against an independent
fp32-input `F.linear` reference is at most `7.16e-7`, versus at least `7.48e-3`
for the same reference rounded through bf16.

## Evidence

- `raw/candidate_precision_regression.txt`: exact candidate behavior, three
  failing precision subtests.
- `raw/corrected_precision_regression.txt`: corrected behavior, one test and
  three subtests passing.
- `raw/corrected_gpu_numerics.txt`: device, architecture, imported source path,
  and independent numerical comparisons.
- `raw/existing_correction_bias_tests.txt`: the prepared base's nine GLM-5.2
  correction-bias tests still pass.

## Limitations

No GLM-5.2 model weights were available, so no full-model semantic or accuracy
run was performed. Validation used one GPU only; no multi-GPU or multi-node
claim is made. The correction bypasses aiter's tuned GEMM dispatcher for this
small router GEMM because its untuned fallback does not honor fp32 computation;
performance against a tuned production entry was not evaluated. No native
source changed, so no native rebuild was applicable.
