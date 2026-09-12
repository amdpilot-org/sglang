# EAGLE greedy TP synchronization correction

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2376 at
`cad9443d635cf595832da6a7e4d6f82ce81adcf4`

Independent review: https://github.com/amdpilot-org/sglang/pull/2443

The candidate's three tests passed unchanged on the assigned MI350X/gfx950.
The prepared source already synchronized finalized greedy decisions on ROCm,
but guarded that synchronization with `_is_hip`.

An independent regression selected the same greedy branch with `_is_hip=False`,
two logical TP ranks, and opposed near-tied GPU logits. Before correction it
failed because logical rank 0 returned three accepted tokens and logical rank 1
returned two. No broadcast calls occurred. This reproduces the review's
concrete remaining source-level counterexample.

The correction makes the existing finalized-decision broadcast
backend-independent for the greedy branch. It preserves rank-0 authority,
DP-attention group selection, and the TP=1 no-collective boundary. The focused
suite passes all four cases after correction.

Only one GPU was assigned. The tests use real gfx950 tensors and argmax but
replay broadcast semantics across two logical ranks. They do not claim a real
two-process TP group, a downstream collective deadlock reproduction, AITER
all-reduce fusion coverage, full serving, or model semantic validation.
