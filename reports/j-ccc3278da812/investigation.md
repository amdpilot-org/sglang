# Independent review of PR 2460

Upstream issue: https://github.com/sgl-project/sglang/issues/31475

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2495

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2460

Exact candidate commit: `6c8bf4d4fddcbe5ce6fabf1b3248e9d9d5014e0d`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original issue at the source/component
level and also fixes the concrete A2A counterexample left by the earlier
candidate. It separates skips caused by a genuinely deferred cross-rank SUM
from A2A/FP4 paths that already absorb the routed collective. Consequently a
replicated TP1 shared output is divided by `moe_ep_size` only when a later SUM
will count every rank's copy.

On the recorded base, the actual `forward_normal` source adds the full replicated
shared output after skipping the immediate all-reduce. A deterministic gfx950
calculation using that exact ordering reproduced `R + 8*S` and did not match
`R + S` (maximum absolute error 15.595807194709778).

At the exact candidate commit, all six focused regression tests passed. An
independent GPU test exercised five boundaries with separate random data:
deferred SUM, an absorbed A2A/FP4-style skip with no later SUM, immediate
all-reduce, nonreplicated shared expert, and TP1. Every comparison had zero
maximum absolute error. The related runtime/communicator suite passed 99 tests
and 140 subtests.

The interpreter imported `sglang`, `deepseek_v2.py`, and `moe/utils.py` from
`/job/repo/python`, not an installed SGLang wheel. The change is Python-only;
there is no changed native code to rebuild, and `compileall` succeeded.

## Limitations

Only one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) was assigned. Therefore
the review did not execute a real eight-rank RCCL reduce-scatterv or A2A
transport. DeepSeek-V2-Lite-Chat and GLM-5.2 weights were unavailable, so no
full-model serving or semantic-accuracy claim is made. These limitations do not
invalidate the directly tested source-level arithmetic and predicate boundary,
but they remain architecture/workload coverage gaps.

