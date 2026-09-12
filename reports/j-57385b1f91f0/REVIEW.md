# Independent review of PR 2913

Candidate: https://github.com/amdpilot-org/sglang/pull/2913 at `ced86c26b014a23c5160fd5660d8d39a9981ddb2`

Upstream issue: https://github.com/sgl-project/sglang/issues/35333

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2945

## Recommendation

Request changes. The candidate is a useful partial framework and correctly hardens the three executable defects reported against parent PR 2748, but it does not fully resolve the original feature request.

The original feature is absent at recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: importing `runtime.utils.compile_trajectory` raises `ModuleNotFoundError`. At the exact candidate, the new module imports from the checkout and all 20 focused tests pass.

Independent reproduction against parent candidate `b9fdebd698e93283f02acc4095f1fcda71233f4e` confirmed that NaN checkpoint metrics and NaN output-adapter metrics were promoted, while equal tensor-valued terminal mappings raised `RuntimeError`. The exact reviewed candidate rejects both NaN cases and structurally compares the tensor-valued terminal mapping successfully.

However, the production integration required by the issue is still absent. All production callers invoke `_maybe_torch_compile(transformer)` without a `CompilePlanResolution`. No configuration/server/request path loads a `CompiledPlanManifest`, constructs `CompilePlanResolver`, derives the current `CompileWorkloadSignature`, or supplies the resolution to `DenoisingStage`. Consequently an uncovered production request still follows the pre-existing global compile flags instead of being forced eager. This is the central runtime safety contract, not merely missing coverage.

An additional independent malformed-gate case (`not_a_metric` in `tensor_thresholds`) raises `KeyError` instead of returning a rejected gate result. Manifest/gate schema validation is therefore not fail-closed for unsupported metric names.

## Execution evidence

- Base import reproduction: `ModuleNotFoundError` at the recorded base.
- Candidate focused tests: `20 passed`.
- Candidate source import: `/job/repo/python/sglang/multimodal_gen/runtime/utils/compile_trajectory.py`.
- Parent counterexamples: both NaN cases passed and tensor terminal state raised.
- Candidate adversarial checks: both NaN cases rejected; tensor terminal state passed structural comparison; unknown metric raised `KeyError`.
- Real GPU toy trajectory: AMD Instinct MI350X, gfx950, Torch `2.11.0+rocm7.2`; full four-step trajectory passed with maximum absolute error `1.1920928955078125e-07`; reset/short-history passed; injected step-1 perturbation was rejected. Torch emitted non-fatal Dynamo metrics serialization errors.

Raw logs and the adversarial script were preserved outside the revision-switching checkout at `/job/evidence-j-57385b1f91f0/`.

## Limits

No native source changed and the prepared environment declares no native build target, so a native rebuild was not applicable. One AMD Instinct MI350X (gfx950) was available. No production diffusion or stateful world-model weights were available. The GPU test is a deterministic Linear/tanh toy and does not validate a real architecture, semantic output quality, cache on/off, CFG serial/parallel, changing history regimes, reset isolation in a production pipeline, distributed topology, mutation/alias metadata, serving transport, or manifest-driven eager fallback.

Classification: partial fix plus test hardening; not a full original-issue fix. The corrected parent-review cases are verified, while the claimed broader feature remains unverified in real model paths and demonstrably absent from the production request path.
