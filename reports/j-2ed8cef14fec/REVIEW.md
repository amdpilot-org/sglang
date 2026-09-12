# Independent review of PR 3098

Candidate: https://github.com/amdpilot-org/sglang/pull/3098 at `fd613337e313f299b9c3bd94dbbf9211e2fb356c`

Upstream issue: https://github.com/sgl-project/sglang/issues/35333

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3119

## Recommendation

Request changes. The candidate is a meaningful partial fix, but it does not fully implement the original stateful full-trajectory promotion contract.

## Findings

1. **The named stateful production path never resolves a promoted plan.** `CausalDMDDenoisingStage`, used by LingBot-World, overrides `forward()` and does not call either `_resolve_compile_plan()` or `_maybe_enable_cache_dit_and_torch_compile()`. When manifests are configured, `DenoisingStage.__init__()` deliberately defers compilation. Consequently the stateful causal request path remains eager even for a matching validated manifest. The independent AST regression in `raw/candidate-causal-counterexample.log` fails on the exact candidate with `stateful causal production request path never resolves manifests`.

2. **There is no production/offline validation flow that creates promotion evidence.** Outside tests, `TrajectoryCapture` and `evaluate_trajectory_gate` have no callers, and no model adapter registers named checkpoints or public/task metrics. Manifest loading is connected to ordinary denoising request handling, but manifest production is not connected to a complete warmup plus measured rollout. Thus the candidate cannot itself qualify history regimes, cache on/off, CFG serial/parallel, reset isolation, mutation/alias metadata, or semantic output metrics required by the issue.

3. **The previous concrete safety defects are fixed.** The candidate constructs `CompilePlanResolver` from server configuration, derives a request signature in the ordinary denoising path, passes the resolution to `_maybe_torch_compile`, disables previously compiled regional wrappers for uncovered requests, and reports an unsupported tensor metric as a failed gate rather than raising `KeyError`. Its focused suite passed 22 tests.

4. **GPU evidence is narrow.** The bundled deterministic Linear/tanh script compiled and ran on one AMD Instinct MI355X with Torch `2.11.0+rocm7.2`; four-step maximum absolute error was `1.1920928955078125e-07`, reset/short-history passed, and an intentional step-1 perturbation was rejected. This is useful compiler/numerical evidence but is not evidence for a diffusion or stateful world-model architecture. Torch emitted non-fatal Dynamo metrics JSON-serialization errors.

## Base reproduction

At recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, `compile_trajectory.py` does not exist and the causal request path has no trajectory-plan integration. This reproduces the original missing feature at the prepared base; see `raw/base-original-failure.log`.

## Architecture and environment limitations

No production diffusion/LingBot model weights or promoted manifests were available. Therefore no real stateful trajectory, cache mutation/alias behavior, CFG mode, history-length regime, reset isolation, task metric, or distributed topology was qualified. The available single GPU was an AMD Instinct MI355X. No C++/native source changed in the candidate, `repository-environment.json` reports no native component, and no native rebuild was applicable.

## Classification

This is a **partial fix**: runtime enforcement exists for ordinary denoising paths and test-only gate hardening is valid, but the original issue explicitly requires a reproducible full-trajectory promotion flow for a stateful model and names LingBot-World. Those requirements remain unresolved.
