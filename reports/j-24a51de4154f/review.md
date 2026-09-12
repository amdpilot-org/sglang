# Independent review of PR 2917

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2917 at exact commit `7ba8f72eedf5db55b7b554cb2197cdbc454015ab`

Upstream issue: https://github.com/sgl-project/sglang/issues/35332

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2884

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2949

## Verdict

Recommendation: **accept as a partial contribution**. The candidate does not fully resolve the original issue.

The exact candidate adds an opt-in, disabled-by-default framework controller and fixes the concrete correction-generation counterexample: downstream in-place mutation of a reused tensor no longer changes `StepReuseState.last_real_prediction` or a later reuse. Both the stored observation and every returned reuse have independent storage. This passed on CPU and one AMD Instinct MI350X.

The candidate is not a complete implementation of the original feature request. No concrete model adapter is enabled, several model-specific denoising overrides bypass the shared integration, and real multi-rank agreement was not exercised. The candidate's own PR prose and retained reports disclose these limits, so the partial implementation is not presented as full end-to-end proof.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: importing the proposed controller fails with `ModuleNotFoundError`; the framework feature is absent.
- Exact candidate source imports resolved to `/job/repo/python/sglang/...`, not an installed copy.
- Candidate focused tests: 17 passed, covering controller behavior, mutation isolation, scheduler advancement, terminal verification, scope reset, trace-only behavior, synthetic token synchronization, and CFG-gating compatibility.
- Independent mutation check: after mutating the first reuse from 3.25 to 6.5, the canonical state and next reuse remained 3.25 with three distinct storage pointers on CPU and GPU.
- Compatibility tests: 68 passed plus 30 subtests.
- Source inspection found bypassing `_run_denoising_step` implementations in model-specific Ideogram, LTX-2, JoyEcho, and progressive Ideogram paths.

## Classification

This is a **partial fix**: it provides phase-one framework/controller behavior and corrects the independently reported tensor-aliasing defect. It is neither merely test-only hardening nor an unverified claim. It is not a full original-issue fix because model semantics, quality gates, persistent state equality, model-specific integration, and actual distributed execution remain outstanding.

## Environment limits

The assigned device was one AMD Instinct MI350X with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. One GPU cannot validate real multi-rank collective control flow. No qualified stateful diffusion weights/fixture were available. The tiny Llama fixture is irrelevant to diffusion observation semantics, quality, or persistent world-model state. All candidate implementation changes are Python; `repository-environment.json` records `native: null`, so no native rebuild applied.
