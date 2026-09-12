# Correction review for PR 2748

Reviewed candidate https://github.com/amdpilot-org/sglang/pull/2748 at exact commit
`b9fdebd698e93283f02acc4095f1fcda71233f4e`, using the independent review in
https://github.com/amdpilot-org/sglang/pull/2831 as untrusted diagnostic input.

The three executable counterexamples were independently reproduced before the
candidate was applied. A NaN checkpoint and a NaN output-adapter score both
passed the gate, while equal multi-element tensors in terminal state raised an
ambiguous tensor truth-value `RuntimeError`.

This correction preserves the candidate's schema, resolver, capture, region
inventory, and compilation behavior. It makes all non-finite measured metrics
fail closed and compares nested terminal-state mappings, sequences, and tensors
structurally. The focused suite now includes regressions for both behaviors.

The candidate remains incomplete relative to the original feature request. No
production server/configuration or request path loads manifests, constructs a
workload signature, invokes `CompilePlanResolver`, or supplies a resolved plan
to `DenoisingStage`. No model weights were available, so inventing a model
revision/state schema or claiming cache, CFG, reset, alias/mutation, semantic
output, or real diffusion/world-model qualification would be unsupported. The
GPU evidence is limited to the candidate's deterministic Linear/tanh toy.
