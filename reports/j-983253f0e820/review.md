# Independent review of candidate PR 2296

Candidate commit: `650754532755ba75de1195f7d5231899ae9cabb5`

Upstream issue: https://github.com/sgl-project/sglang/issues/31475

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2234

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2331

## Finding

Request changes. The candidate corrects the arithmetic for the reported
`moe_a2a_backend=none`, EP + DP-attention reduce-scatterv path, but applies the
same `1 / moe_ep_size` scaling to every reason that
`should_skip_post_experts_all_reduce()` can return true.

That predicate also returns true for A2A backends such as FlashInfer and PPLX.
Their combine operation has already reduced routed expert output back to the
source rank; the separately evaluated TP1 shared expert does not participate in
a later cross-rank SUM. The existing full-strength addition is therefore
required. At EP=8 the candidate changes the correct `R + S` result to
`R + S/8`.

The candidate regression mocks the composite skip predicate as a Boolean and
models every `True` result as a later eight-rank SUM, so it cannot detect this
distinction. The correction needs to be gated by the skip reasons that really
defer to a downstream SUM, or the shared output must be placed/scaled according
to the actual communication path.

## Evidence

- Prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` on the assigned AMD
  Instinct MI355X (`gfx950`) reproduced the issue arithmetic: maximum error
  against `R + S` was `15.595806956291199`, while error against `R + 8*S` was
  `1.4603137969970703e-06`.
- Exact candidate source imported from
  `/job/repo/python/sglang/srt/models/deepseek_v2.py`; Torch was
  `2.11.0+rocm7.2`, HIP `7.2.26015`.
- Candidate contract arithmetic for the reported downstream-SUM path matched
  `R + S` with maximum error `6.92903995513916e-07`.
- Candidate regression: 5 passed.
- Related runtime-context and communicator tests: 99 passed, 140 subtests
  passed.
- Independent A2A boundary: candidate returned `[10.25, -3.0]` for routed
  `[10, -4]`, shared `[2, 8]`, EP=8; the no-later-SUM contract requires
  `[12, 4]`. The assertion failed with maximum error `7.0`.

Raw logs, scripts, issue metadata, and the candidate diff are retained outside
the revision-switching checkout at
`/job/review-evidence-j-983253f0e820/`.

## Limitations

Only one gfx950 GPU was assigned. The review validates GPU arithmetic but not
an eight-rank RCCL reduce-scatterv or A2A transport run. DeepSeek-V2-Lite-Chat
and GLM-5.2 weights were unavailable, so no full-model generation or semantic
accuracy claim is made. No native source changed, so no native rebuild was
applicable.
