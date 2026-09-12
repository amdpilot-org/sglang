# Consolidated investigation of PR 1382 and PR 1469

Upstream issue: https://github.com/sgl-project/sglang/issues/35673

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1506

Candidate: https://github.com/amdpilot-org/sglang/pull/1382 at exact commit
`695bab76f0928ad190ac33134d5fd6f15af62638`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1469 at exact
commit `cc76bf7c4ca3e5b4b456335d5b4d0a2a0df7c2cf`.

The candidate's source correction is retained. Gemma 4 vision inputs are dense,
and its forward path already has the host integer `seq_len`. Passing that value
as `max_seqlen` avoids recomputing it from a GPU cumulative-length tensor with
`.item()` in each vision block. The regression failed twice on the recorded base
because the backend call lacked `max_seqlen`, and the focused suite passes after
the change.

The review's numerical counterexamples were reproduced exactly, then traced to
the review harness rather than the candidate. `VisionTritonAttention` consumes
`softmax_scale` in `forward`; its intentionally generic constructor ignores that
keyword. The review harness supplied `1.0` only to the constructor, omitted it
from `forward`, and compared the resulting default scale `1/sqrt(32)` against an
SDPA reference using scale `1.0`. This reproduces errors 1.60046387 for `(3,5)`
and 2.66284180 for `(4,9)`. Supplying `softmax_scale=1.0` to `forward`, matching
the actual Gemma call, yields maximum absolute error 0.00195312 for both shapes
on the assigned gfx950 GPU. The contract regression now also asserts that Gemma
passes the scale to the backend.

This evidence does not establish that the original distributed hang is fixed.
A host wait at `.item()` can expose earlier stalled GPU work, so removing that
wait site is a justified synchronization reduction but not proof that kernels or
TP collectives complete. Only one GPU was assigned and Gemma-4-31B-it weights
were unavailable. TP=8, hybrid SWA, the full vision/model path, and HTTP `/health`
readiness therefore remain unverified. No speculative source change was made for
those unavailable conditions.
