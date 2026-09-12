# Independent review of amdpilot-org/sglang PR 2167

Candidate reviewed at exact commit `40a488defc20fc508eb9f5d4e099ae869d2cbd9f`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/33187

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2205

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2167

## Recommendation

Request changes. The candidate fixes the previously reported abort-only FP32
case and routes full-NaN prefill/decode cleanup through the comprehensive abort
helper. However, an independent MI350X test found that FP16 full-NaN logits are
converted to `-inf` by `nan_to_num_(nan=-1e30)`. Softmax remains all-NaN, so a
sampling backend can fail before the request-scoped abort is processed. The
candidate regression uses FP32 only and does not expose this counterexample.

Speculative decoding remains explicitly unsupported (assertion), and a live
hierarchical-cache write-through configuration was not available. The cleanup
unit test verifies mocked hooks and `release_kv_cache(..., is_insert=False)`,
but does not constitute an end-to-end hierarchical-cache publication test.

No native sources changed, so no native rebuild was applicable. Imports were
confirmed from `/job/repo/python/sglang`; Torch was loaded from the prepared
environment. Raw command output is preserved outside revision switching at
`/job/review-evidence-j-5beae295a223/`.
