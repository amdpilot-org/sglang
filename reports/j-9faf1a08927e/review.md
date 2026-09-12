# Correction generation 2: candidate rejected

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/876

Candidate parent: https://github.com/amdpilot-org/sglang/pull/762 at
`26f1a21ab632cea2bb9001ce6bd7277dcd464958`.

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/844.

## Reproduction

I checked out the exact candidate before changing the delivery branch. Its focused
suite passed 20 tests, but the n-1 regression passes by asserting zero reusable
indices. An independent actual-cache probe on the assigned AMD Instinct MI350X
reproduced all concrete review claims:

- inserting `[10,20,30]` with its only complete Mamba state at depth 3, then
  matching `[10,20]`, split the edge, left the depth-2 node without a state, and
  returned 0 indices;
- an exact owned depth-2 checkpoint returned 2 indices equal to an independent
  GPU `torch.arange` reference;
- an arbitrary owned depth-37 checkpoint returned 37 indices equal to the GPU
  reference; and
- divergence at depth 31 before the first owned depth-64 checkpoint returned 0.

## Why no runtime source correction is justified

Moving the depth-3 state to the split depth-2 node would relabel recurrent and
convolution state from the wrong token depth. Removing the `best_value_len`
truncation would retain attention KV through depth 2, but it would also omit those
tokens from the extend batch. The recurrent layers would therefore have no path
to replay `[10,20]` from the root state. `mamba_branching_seqlen` only selects a
checkpoint inside tokens that are already being executed; it does not provide
per-layer prefix lengths or replay recurrent layers while attention layers reuse
deeper KV.

The cache already accepts a complete checkpoint at arbitrary exact depths, as the
depth-37 control proves. What remains missing is either checkpoint production at
the required exact depth (including every recurrent and convolution layer), or a
model execution path with separate attention-KV reuse and recurrent replay. Both
require hybrid-Mamba model execution and cold/warm numerical validation. No such
weights were prepared, and the qualified tiny Llama fixture cannot validate this
architecture. Consequently a runtime source change would be speculative.

This delivery preserves the candidate's valid ownership tests and records the
remaining limitation rather than claiming the reported zero-hit is fixed.
