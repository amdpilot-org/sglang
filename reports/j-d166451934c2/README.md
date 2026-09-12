# Issue 32470 investigation

The prepared base already contains the source correction merged upstream in
https://github.com/sgl-project/sglang/pull/32467.  The five-line warp-0 scratch
initialization was removed from `plan_compress_prefill_kernel0`, leaving each
warp as the sole writer of its own min/max slot before the existing block-wide
barrier.

This change adds missing regression coverage.  The runtime cases execute the
actual JIT planner on the assigned gfx950 and compare complete valid plan rows
against the separate CPU implementation.  They cover the issue's compact
`[4] * 72 + [3] * 24` extend shape, transitions on either side of warp
boundaries, and uniform fast-path controls.  They also check that emitted write
`ragged_id` values are unique and within the compact input.

`failing-before.log` was captured after temporarily restoring exactly the five
lines deleted by upstream PR #32467.  The deterministic invariant test failed;
the runtime stress cases happened to pass on gfx950, so this is not claimed as
a reproduction of the reported nondeterministic H20 illegal memory access.
`passing-after.log` is from the restored prepared source and records all tests
passing.

The original TP=2 CUDA graph capture could not be run with one AMD GPU and no
DeepSeek-V4-Flash-DSpark weights.  No full-model, CUDA/H20, or distributed claim
is made.
