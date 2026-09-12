# Correction review for side-effect-aware step reuse

- Upstream issue: https://github.com/sgl-project/sglang/issues/35332
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2884
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/2763
- Independent review PR: https://github.com/amdpilot-org/sglang/pull/2850
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Exact candidate reviewed: `f9d58a04741052affdf8b9d5d90d7538b40eb0e4`

## Confirmed counterexample and correction

The exact candidate returned `StepReuseState.last_real_prediction` directly from
`reused_prediction()`. An in-place scheduler/consumer mutation therefore changed
the controller's canonical cached value. The independent reproducer in
`failing-before.log` failed on CPU and AMD Instinct MI355X: after multiplying the
first reuse of `3.25` by two, the next reuse was `6.5` and shared storage.

The correction returns a clone for each skipped step. The regression test mutates
one reused tensor and verifies that both the cached last real prediction and the
next reuse remain `3.25`, with independent storage. `passing-after.log` records
this passing on CPU and MI355X.

## Remaining limitations

The other review findings are real validation or integration limits, but do not
justify speculative source changes without the unavailable prerequisites:

- No concrete model adapter is enabled, so architecture-specific observation
  semantics, thresholds, output-quality gates, and persistent session/KV writes
  remain unvalidated on a stateful diffusion model.
- Only one GPU was assigned. The synthetic control/follower synchronization test
  passes, but real multi-rank collective-control-flow safety remains unverified.
- Model-specific `_run_denoising_step` overrides do not automatically inherit the
  shared-stage hooks and require explicit per-architecture integration before
  opting into reuse.
- No native rebuild applies because this contribution changes Python only and the
  prepared environment records `native: null`.

The tiny Llama fixture is not applicable evidence for diffusion architecture
semantics, output quality, persistent world-model state, or distributed execution.
