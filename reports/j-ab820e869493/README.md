# Independent review of PR 2319

Reviewed exact candidate commit `732c5174c612e2ca662eebec2075e55f2bf15b8a`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original
compact ragged-prefill planner race.

The candidate is useful test-only hardening: its focused suite passes on a real
gfx950 JIT build, it rejects the historical unsynchronized scratch initializer,
and it accepts the upstream init-plus-unconditional-barrier form. Prepared main
already has the production fix (the redundant initializer was removed), so the
candidate itself makes no production change.

Request changes because the new source helper does not fully encode the claimed
block-synchronization invariant. It accepts a barrier placed inside
`if (tx < kNumWarps)`, where only a subset of the block reaches
`__syncthreads()`. The adversarial reproducer exits zero and prints
`CANDIDATE_HELPER_ACCEPTED_DIVERGENT_BARRIER`; see
`evidence/adversarial-divergent-barrier.log`.

The candidate suite passed with a newly compiled JIT library when the actual
cache variable, `SGLANG_JIT_CACHE_DIR`, was used. The candidate's recorded
command instead uses the unrecognized `SGLANG_JIT_KERNEL_CACHE_DIR`; its retained
artifact is under the shared `/job/.cache` rather than its claimed private cache.

The historical racy source was reconstructed temporarily on the recorded-base
implementation. The static regression failed as expected, but 100 executions of
the reported `[4] * 72 + [3] * 24` planner shape did not manifest corruption on
the available AMD MI355X/gfx950. The original NVIDIA H20 CUDA graph capture,
DeepSeek-V4-Flash-DSpark weights, TP=2, and full serving path were unavailable.
