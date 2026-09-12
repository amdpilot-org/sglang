# Independent review of candidate PR 1682

Upstream issue: https://github.com/sgl-project/sglang/issues/34239

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1717

Candidate: https://github.com/amdpilot-org/sglang/pull/1682 at exact commit
`6d5477605b06c845b25f9260c59599dc33a0aedd`.

## Finding

Recommendation: **accept as a partial, source-level fix**. The candidate fixes
a real context-boundary inconsistency: the scheduler's generation limit left
only one free position, although EAGLE/NEXTN target verification can require
four output slots with the reported `steps=3`, `topk=1`, and
`draft_tokens=4`. On the recorded base, the reported boundary example permits
52 new tokens, giving `262091 + 52 + 4 = 262147`. The candidate permits 49,
giving exactly 262144.

This does **not** fully resolve the original issue as presently evidenced. The
production failure was an NVIDIA H800 TP8 CUDA illegal memory access involving
Qwen3.5-397B, FA3, decode CUDA Graphs, overlap scheduling, hybrid Mamba state,
and a 262144-token request. This job has one AMD MI355X (gfx950), ROCm 7.2, and
no model weights. Neither the original CUDA fault nor its precise causal chain
could be reproduced. The candidate may prevent one plausible overrun for
requests whose effective generation limit reaches this scheduler path, but a
separate CUDA Graph/static-buffer defect or another long-context path remains
a counterexample to the claim of a complete fix.

## Evidence

The prepared checkout was exactly the required base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Its existing focused suite passed,
but an independent probe exposed the contract violation above. I then checked
out the exact candidate commit in detached mode. Its regression suite passed
9 tests and 542 parameterized subtests. Independent cases covered the reported
boundary, one-token and zero-token remaining space, explicit clipping, a small
explicit request, non-speculative behavior, page-budget interaction, and 3,895
exhaustive small-context combinations across reserves 0, 1, 2, 4, and 8.

The interpreter imported `scheduler.py` from `/job/repo/python`, confirming the
tested source checkout rather than an installed SGLang copy. Torch was
`2.11.0+rocm7.2`; the visible device was AMD Instinct MI355X, gfx950. No native
source changed, so no native rebuild was required. GPU execution would not
exercise this Python-only clipping decision and was not used as surrogate proof
for the unavailable CUDA/H800 failure.

Raw command results are retained under `raw/`. After review, the checkout was
returned to `amdpilot/j-7f6e91fd8856` at the recorded base before this report
was committed; the candidate itself was not modified or duplicated.
