# Independent review of candidate PR 780

- Upstream issue: https://github.com/sgl-project/sglang/issues/38707
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/821
- Candidate: https://github.com/amdpilot-org/sglang/pull/780
- Exact candidate commit: `228043c10a9a95e33a94076e3f1e74abdae151df`
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

`unverified`. The patch is a plausible partial correction for SCATTERED
attention-TP PP proxy views in decode graph capture/replay and dummy warmup,
and its focused tests pass. It does not demonstrate that the original open
issue is fixed. The original failure is an 8-GPU gfx942, TP=4/PP=2,
GLM-5.3 DSA multi-chunk serving failure whose threshold changes across decode
graph backends. This environment has one gfx950 GPU and no GLM-5.3 weights.

The candidate's failing-before regression is an import error for its newly
introduced helper. Its passing test directly tests only helper arithmetic; it
does not execute the modified graph capture, replay, or dummy-run call sites,
nor any PP communication. Thus it is useful unit hardening, but not a
failing-before/passing-after reproduction of the reported defect.

## Source review

The candidate changes Python only. It introduces `get_pp_proxy_num_tokens`
and uses it in three places: base dummy warmup, decode graph capture input,
and decode graph replay output. For a SCATTERED boundary with 32 global rows
and attention TP size 4, those paths now select 8 rank-local rows. The change
is consistent with the proposed change in upstream PR 31012 for those paths.

An adjacent path remains: `PrefillCudaGraphRunner._capture_pp_proxy_tensors`
still slices PP proxy buffers with the global `num_tokens` and does not consult
`require_attn_tp_gather`. An independent constructed check returned 32 rows
for a 32-row input where the candidate's stated SCATTERED TP=4 invariant would
be 8. This does not prove that path caused issue 38707, but it prevents a broad
claim that all graph PP-boundary proxy sizing is corrected.

## Verification and limitations

The imported `sglang` and all three runner modules came from `/job/repo/python`,
while Torch came from the prepared ROCm environment. The assigned device was
an AMD Instinct MI355X, `gfx950:sramecc+:xnack-`; only one GPU was visible.
The candidate's 57 focused/neighboring tests passed. Independent boundary
checks covered scattered and unsharded views, TP=1, zero rows, non-divisible
rows, and invalid TP size. A real GPU tensor was used in the adjacent prefill
capture check.

No native source changed, `repository-environment.json` declares no prepared
native component, and no native rebuild was applicable. The review did not
use the tiny Llama fixture because it cannot qualify GLM DSA semantics,
multi-GPU PP, gfx942 behavior, or the reported length/backend threshold.

The following remain unverified: the original 20k/43k/100k request matrix;
`full`, `tc_piecewise`, `breakable`, and `disabled` backend behavior; overlap
scheduler first-request failure; PP=2 versus PP=4; and absence of the async HIP
fault on gfx942 with the reported 753B FP8 model.
