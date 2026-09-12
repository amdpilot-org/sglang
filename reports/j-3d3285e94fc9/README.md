# Correction review for issue 32470

This correction independently reproduced both counterexamples from
https://github.com/amdpilot-org/sglang/pull/2235 against candidate
https://github.com/amdpilot-org/sglang/pull/2129 at exact commit
`61a5235257fc808054bc3fe0061f763b49c5293f`.

The candidate's static regression rejected the historically correct initial
upstream fix, which retained scratch initialization and placed a block barrier
before the per-warp reduction. The corrected invariant accepts either removal
of the redundant initialization or initialization ordered by a
`__syncthreads()`, and it explicitly rejects the same initialization without
the barrier. The candidate's useful real-GPU planner parity and bounds tests
are preserved.

The exact candidate also exits 2 from `git diff --check` due to trailing
whitespace in its retained logs, despite recording exit code 0. Those artifacts
are cleaned in this consolidated patch.

The full corrected suite passed on the assigned gfx950, including the reported
`[4] * 72 + [3] * 24` compact shape, warp-boundary variants, uniform controls,
and the new safe/racy source boundaries. The original CUDA H20 TP=2 full-model
capture remains unverified because this environment has one AMD GPU and no
DeepSeek-V4-Flash-DSpark weights.
