# Independent review of candidate 152efc1

Upstream issue: https://github.com/sgl-project/sglang/issues/33360

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1998

Candidate: https://github.com/amdpilot-org/sglang/pull/1958 at
`152efc1aaa4da7cf62be370847166446667d21e2`

## Recommendation

Request changes. The candidate is test-only hardening around a production fix
already present in the recorded base, but its token-ID AST assertion accepts an
incorrect source operand. Its report also records the wrong mirror issue and a
GPU model that does not match the device exposed during this review. The
original eight-H800, TP8/DP4, Marlin, DeepSeek-V4 accuracy case remains
unverified in this environment.

## Source finding and failing-before evidence

The image-prepared checkout exactly matched the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. That base already uses
`dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)` in the
synchronous post-attention MoE path and clones the local token-ID views before
replicate gathers in the main and NextN models. The captured upstream PR #31700
patch shows these are the issue-specific changes from the historical behavior.

Against a temporary source fixture restoring the historical partial hidden
state gather and removing both token-ID clones, the candidate regression
produced `2 failed, 1 passed`. This reproduces the source-level defect contract,
not the original model's corrupt generated text.

## Candidate and adversarial results

At the exact candidate commit, its regression produced `3 passed`. SGLang was
imported from `/job/repo/python/sglang`, while Torch came from the prepared
interpreter. The candidate changes only Python tests and reports, so no native
library rebuild was applicable.

An independent corrupted fixture replaced both correct local arguments
`input_ids[:, None].clone()` with `input_ids_global.clone()`. This reads the
newly allocated, uninitialized destination rather than local token IDs, but all
three candidate tests still passed. The assertion checks only that the second
argument is some method named `clone`; it does not check the receiver. The
hidden-state test likewise does not assert that `dp_gather_partial` is absent,
so an extra erroneous partial gather can coexist with the required replicate
call and still satisfy the test.

The candidate report points to mirror issue `1904`, whereas this reviewed task
is mirror issue `1998`. Its retained GPU log says MI350X, while the prepared
interpreter reported `AMD Instinct MI355X` with
`gfx950:sramecc+:xnack-` during this review.

## GPU mechanism check and limitations

A real tensor calculation ran on the assigned MI355X/gfx950. Summing replicated
values scaled them exactly by widths 2, 4, and 8; selecting one replica
preserved values. In-place zeroing of `input_ids[:, None]` changed the caller,
while zeroing a clone did not. This independently supports the proposed
arithmetic and aliasing mechanism, but it is not execution of SGLang's
distributed collective or NVIDIA Marlin.

Only one AMD gfx950 GPU was assigned. DeepSeek-V4-Flash-0731 weights, eight
NVIDIA H800 GPUs, CUDA, and the Marlin backend were unavailable. Consequently
the original TP8/DP4 serving failure, generated-text accuracy, and full fix
remain unverified locally. The candidate should be described as test-only
hardening of an already-present fix, not a locally verified full resolution of
the original deployment.

Raw outputs and the captured related-fix patch are under `evidence/`.
