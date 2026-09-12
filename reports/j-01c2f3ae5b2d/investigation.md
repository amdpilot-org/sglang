# TP1 shared expert deferred-reduction correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31475

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2418

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2296

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2380

The exact candidate commit `650754532755ba75de1195f7d5231899ae9cabb5`
was applied unchanged before correction. Its five focused tests passed, but the
review counterexample reproduced on the assigned gfx950: for routed output
`[10, -4]`, shared output `[2, 8]`, and EP=8, an A2A skip with no later SUM
returned `[10.25, -3]` instead of `[12, 4]` (maximum absolute error 7.0).

The candidate used the composite `should_skip_post_experts_all_reduce()` result
to decide whether to scale the replicated shared output. That predicate also
covers A2A combines and an FP4 all-gather path that already absorb the routed
collective and do not cause a later SUM of the separately computed shared
output. The correction introduces the narrower
`should_defer_post_experts_all_reduce()` predicate for skip reasons that really
do defer to a later cross-rank SUM: fused MLP all-reduce, MLP reduce-scatter,
DWDP, and DP reduce-scatterv. The original broad predicate remains responsible
for deciding whether to execute the immediate post-experts all-reduce.

After correction, the focused suite passes six tests, including the A2A
no-later-SUM boundary. The related runtime/communicator suite passes 99 tests
and 140 subtests. A gfx950 arithmetic check produced exact expected results for
both boundaries: `R + S` for A2A with no later SUM and `R + S/8` per rank for
the DP reduce-scatterv contribution.

Raw command output is retained in `reports/j-01c2f3ae5b2d/raw/`.

## Limitations

Only one AMD Instinct MI350X (`gfx950`) was assigned, so no real eight-rank
RCCL reduce-scatterv or A2A transport was executed. DeepSeek-V2-Lite-Chat and
GLM-5.2 weights were unavailable, so no full-model serving or semantic-accuracy
claim is made. The source change is Python-only and required no native rebuild.
