# Independent review of PR 2034

Upstream issue: https://github.com/sgl-project/sglang/issues/32781

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1988

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2070

Candidate: https://github.com/amdpilot-org/sglang/pull/2034 at
`7de89a2736938ae5d0ef060aa226275c93b7efc1`

## Recommendation

Accept the candidate as **test-only hardening**. It is based directly on the
recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, adds no source or
native implementation changes, and its focused regression passes. The test
exercises the corrected non-padded loader path that the reported CUDA Marlin
configuration uses.

Do not describe this candidate itself as a full fix for the original issue.
The recorded base already contains the source correction: `_load_w13` and
`_load_w2` derive checkpoint offsets from the unpadded loaded tensor instead
of the padded destination shard. The candidate adds regression coverage for
that existing behavior.

## Evidence

The prepared checkout was clean at the exact recorded base. The candidate's
sole parent is that base, and its diff contains only one Python test file and
the originating task's report files.

Using the old destination-derived offsets independently reproduced all four
reported failures:

- w13 rank 13: start 3328 into length 3072;
- w13 rank 15: start 3840 into length 3072;
- w2 scale rank 13: start 104 into length 96;
- w2 scale rank 14: start 112 into length 96.

At the exact candidate commit, its regression suite passed 8 tests. An
independent oracle then exercised the actual `_load_w13` and `_load_w2`
methods for all 16 ranks, including separate `w1`/`w3` placement, fused
`w13`, w2 MXFP4-scale geometry, and a presharded boundary. It passed on CPU
and on the assigned AMD Instinct MI355X (`gfx950`), with source slices checked
against independently constructed tensor references and padding checked for
zeros.

The active interpreter imported SGLang and the loader directly from the
candidate checkout under `/job/repo/python`, rather than from an unrelated
installed package. No native files changed, so no native rebuild was needed
or performed.

Raw review evidence was preserved outside the checkout at
`/job/review-evidence-j-0f4317e5d661/` before switching back to the prepared
review branch.

## Remaining limits

The assigned device is gfx950, not an NVIDIA H100, and only one GPU was
available. DeepSeek-V4-Pro weights were unavailable. Consequently this review
does not validate CUDA Marlin kernel execution, the full model's weight-loading
integration, semantic accuracy, a real TP16 process group, or the reported
two-node deployment. Those are verification gaps rather than method-level
counterexamples found by this review.
