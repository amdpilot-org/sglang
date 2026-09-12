# Independent review of PR 1970

Candidate reviewed at exact commit `41e7ba79d5a83b5e3ddb4f43b733ed0820a04351`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a partial containment fix,
not a fully verified implementation of the original contract.

The base failure mechanism was reproduced on the assigned AMD Instinct MI350X
(`gfx950`) with ROCm 7.2 and PyTorch 2.11. With
`SGLANG_SANITIZE_NAN_LOGITS=1`, a fully-NaN `[1, 163840]` logits row became a
constant `-1e30` row and softmax was exactly uniform at tensor precision
(`sum=1.0`, `max_abs_error=0.0`). Imports resolved to the prepared checkout at
`/job/repo/python/sglang`.

At the candidate commit, its five new unit tests passed, as did 19 neighboring
batch-result, Mamba-boundary, and disaggregation tests (plus four subtests). On
the GPU, enabling both `SGLANG_ABORT_ON_NAN_LOGITS=1` and
`SGLANG_SANITIZE_NAN_LOGITS=1` produced the expected mask `[true, false]` for a
full-NaN and partial-NaN mixed batch; both rows were finite after preprocessing,
and the partial row retained its healthy argmax.

The independent counterexample is the advertised abort flag enabled by itself:

```text
SGLANG_ABORT_ON_NAN_LOGITS=1
SGLANG_SANITIZE_NAN_LOGITS=0
mask [True]
nan_after_preprocess True
finite_after_preprocess False
argmax [0]
```

`detect_full_nan_rows()` records the request, but `sanitize_nan_logits()` still
returns without sanitizing. Sampling kernels therefore receive the original
NaNs and may assert, return an invalid token, or otherwise fail before the host
result processor can turn the mask into a request-scoped 503. The new variable
is documented as an independent opt-in and no candidate test covers this
configuration. Either abort mode must imply safe downstream sanitization, or the
dependency on the sanitize flag must be explicit and enforced/tested.

There is also incomplete lifecycle evidence. Decode abort calls the established
`_handle_sampling_mask_abort()` cleanup, while prefill abort directly calls
`release_kv_cache()`. The latter bypasses multimodal feature release,
disaggregated offload finalization, HiSparse notification, and backend-specific
`prepare_for_kv_cache_release()` used by the established abort helper. The
reported event occurs at first-token sampling after prefill. No candidate test
drives a full prefill abort through these cleanup variants, and hierarchical
cache write-through behavior is not tested.

No native source changed, so no native rebuild was applicable. The available
machine is AMD gfx950, not the report's Blackwell SM10x TP=8 deployment. Kimi-K3
weights, the production transient NaN source, multi-node behavior, and a full
HTTP serving reproduction were unavailable; the GPU test validates only the
logits mechanism and preprocessing boundary.
