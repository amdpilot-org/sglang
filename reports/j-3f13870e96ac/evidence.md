# Correction generation 2 evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/31861

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2525

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2420

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2492

The review counterexample was reproduced at the exact candidate commit
`8929cfe77e5eb9ecc407ed9338da6b515d5376e4`. With BF16 activation and BF16
weight, `ernie4.MoEGate` returned BF16 logits (`[[1.0, 1.0]]`). The candidate's
FP32 CPU TopK guard then raised `ValueError` before native dispatch. See
`evidence/failing-before-candidate-ernie.txt`. The candidate's own five tests
passed unchanged; see `evidence/candidate-existing-tests.txt`.

The consolidated correction preserves the candidate's DeepSeek FP32 fallback
and CPU TopK guard, and makes the Ernie4 CPU gate compute with FP32 activation
and weight. The regression covers both BF16 and FP32 router weights and verifies
that FP32 logits reach the native TopK boundary. Seven focused tests and eleven
focused-plus-adjacent tests pass; see the corresponding raw logs.

The optional specialized BF16-activation x FP32-weight AMX kernel is still not
implemented. Source inspection confirms FP32 is excluded by
`amx_utils.dtype_is_supported`, so FP32 router weights take the semantically
correct FP32 `F.linear` path without specialized AMX acceleration. The host is
an AMD EPYC 9575F and cannot execute Intel AMX; no AMX performance or ISA claim
is made and no native rebuild was applicable.

The assigned AMD Instinct MI355X executed the existing GPU
`linear_bf16_fp32` path. It returned FP32 and matched an independent CPU FP32
`F.linear` reference with maximum absolute error `7.62939453125e-06`; see
`evidence/gpu-linear-reference.txt`. This validates only the GPU numerical
reference, not Intel AMX or an Ernie full-model run. No affected model weights
were available.
