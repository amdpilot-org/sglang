# Independent review of PR 1382

Upstream issue: https://github.com/sgl-project/sglang/issues/35673

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1417

Candidate: https://github.com/amdpilot-org/sglang/pull/1382 at exact commit
`695bab76f0928ad190ac33134d5fd6f15af62638`.

The prepared checkout exactly matched the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The candidate's regression was
preserved outside the checkout and run against that base: both cases failed
because `Gemma4VisionAttention.forward` did not pass `max_seqlen`. On the exact
candidate, its focused suite passed 9 tests. The imported SGLang source was
`/job/repo/python/sglang`; Torch was loaded from the prepared environment at
`/opt/venv/lib/python3.12/site-packages/torch`. No C++/FlyDSL/native source was
changed, so a native rebuild was neither required nor performed.

The source change correctly passes the known dense host value `seq_len` to
`VisionTritonAttention`. This avoids `resolve_precomputed_max_seqlen` deriving
the value from a GPU cumulative-length tensor with `.item()`. Independent
forwarding cases covered a masked `batch=2, seq_len=1` input and a masked
`batch=4, seq_len=9` input. A real gfx950 run of the implicated Triton backend
for `batch=2, seq_len=1` matched PyTorch SDPA exactly.

This is a narrow synchronization mitigation, not proof of the original issue's
expected result. A stack blocked at `.item()` proves that the host waited for
earlier GPU work; it does not by itself prove that this synchronization caused
the GPU work or distributed startup to hang. Passing a host scalar removes that
wait site but may merely defer observation of a preceding kernel/collective
stall. The candidate did not run Gemma-4-31B-it, TP=8, hybrid SWA allocation,
multimodal preprocessing, or HTTP readiness.

An independent direct-backend numerical harness also disagreed with PyTorch
SDPA for batched shapes `(3, 5)` and `(4, 9)` when using the Gemma call's
`softmax_scale=1.0` (maximum absolute differences 1.60046 and 2.66284). This
appears to be outside the one-line forwarding change and is not presented as a
newly introduced regression, but it means the candidate's retained numerical
claim was not independently reproduced by this harness. The harness and raw
outputs are retained under `raw/`.

Recommendation: `request_changes`. The source-level optimization is reasonable
and its narrow contract is verified, but the PR should not be accepted as a fix
for the original TP=8 readiness hang without evidence that readiness completes
or stronger causal evidence excluding an earlier GPU/collective stall.
