# Correction generation 2

Candidate: https://github.com/amdpilot-org/sglang/pull/2900 at
`a9ce297679dec8a6cb680ac3a3b68d7b94d39e3a`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2991.

The two concrete resolver counterexamples were reproduced against the exact
candidate: a report-only stale driver field was ignored, and a feasible,
product-consistent CFG degree of two was selected for a one-branch workload.
`failing-before.log` retains both failures.

The correction makes the recorded and current environment maps match exactly,
adds the accelerator driver version to runtime identity, and validates the
signature-derived CFG and SP constraints before selecting a recorded plan.
`focused-after.log` and `compatibility-after.log` retain passing results.

This remains a partial implementation of the upstream feature. There is no
calibration command, isolated end-to-end benchmark protocol, model
quality/trajectory executor, communication instrumentation, calibrated matrix,
or serving-pool router. No distributed/model-wrapping source was changed, so a
model-serving world-size-one fast path is not claimed. Model/backend-specific
TP, SP, CFG, and FSDP capabilities also are not available to this resolver and
remain the calibration producer's responsibility. One MI355X was visible, but
no model weights or multi-GPU allocation were used; the GPU probe only records
runtime identity and is not model-serving or numerical evidence.
