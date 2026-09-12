# Independent review of PR 3226

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3226 at exact commit `6eee21cad4fbdf2e6593eee89a616d756a2c8e5f`.

Upstream issue: https://github.com/sgl-project/sglang/issues/3642

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3233

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the upstream completion-truncation behavior, so the original zero-reserved-token 400 was not reproducible there: a 6-token prompt plus 8 requested completion tokens at context 10 was accepted after clamping completion to 4. The base did reproduce the reserved-token defect inherited by the earlier candidate: prompt 12, reserved 3, completion 5 was returned as prompt 10 plus reserved 3, exceeding context 10.

The exact reviewed candidate passes its regression suite and independent adversarial checks. It truncates feasible prompt/completion budgets while retaining reserved tokens in accounting, and rejects the impossible case where reserved tokens alone exceed context. An exhaustive grid covering contexts 1-12 and prompt, completion, and reserved counts 0-15 found no accepted request above its context budget and no rejection when the reserved budget was feasible.

The recommendation is **accept**. This is a full fix for the original request-length contract, including the concrete remaining counterexample from PR 3180. The explicit error for `reserved > context` is appropriate because no prompt or completion truncation can make that server configuration fit.

No native source changed, so no native rebuild was applicable. Imports were confirmed from `/job/repo/python/sglang`. One AMD Instinct MI355X was visible with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`, but no compatible EAGLE draft model/weights were available. Therefore nonzero-reserved behavior was validated directly at `TokenizerManager`, not end-to-end on GPU; the qualified tiny Llama fixture cannot exercise that architecture-specific path.
