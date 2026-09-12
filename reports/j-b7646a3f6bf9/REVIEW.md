# Correction generation 2

Candidate: https://github.com/amdpilot-org/sglang/pull/2913 at `ced86c26b014a23c5160fd5660d8d39a9981ddb2`

Independent review: https://github.com/amdpilot-org/sglang/pull/2983

The candidate's gate-metric counterexample was reproduced directly: declaring
`unsupported` as a tensor metric raised `KeyError`. A source audit also confirmed
that production code never constructed `CompilePlanResolver` and all real
`_maybe_torch_compile` calls omitted `resolved_plan`.

This correction preserves the candidate's manifest, capture, resolver, region
inventory, and numerical hardening. It adds an opt-in production configuration
path that loads manifests, defers regional compilation until request state is
available, derives the actual shape/steps/dtype/CFG/cache/parallel signature,
and resolves it before compilation. Compiled regional call wrappers are saved
and disabled for later uncovered requests, providing a real eager fallback
after a covered request has already compiled the model. The Hunyuan3D lazy-load
path uses the same resolver. Unknown tensor metrics now produce a rejected gate
result instead of an exception.

No production diffusion or world-model weights were available. Consequently,
the repository still has no promoted model manifest and this work does not
qualify real history-length regimes, cache variants, CFG variants, reset
isolation, mutation/alias metadata, semantic output metrics, or distributed
execution. The GPU evidence is the retained deterministic Linear/tanh Inductor
trajectory only; it is numerical/compiler-path evidence, not architecture or
semantic evidence.
