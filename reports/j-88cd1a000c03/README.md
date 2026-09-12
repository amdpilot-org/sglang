# DFlash2 dynamic-verification correction

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2743 at `9a7fdb8c7f520ec561be7c7f45a4a5352cc7a7a3`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2794.

The two source-level counterexamples reproduce on the exact candidate: it has no DFlash2 confidence-head module/loading/inference path and no DFlash2 STS option/loader/application path. `candidate-source-reproduction.log` records the audit, and `failing-before.log` runs the added regression tests against that exact checkout (5 failures). The candidate's valid selector fallback, SPS budgeting, ragged verification, graph bucketing, cutoff semantics, and compatibility fixes are preserved.

The correction adds an opt-in trained confidence head for checkpoints declaring `enable_confidence_head`, strict required-weight validation, selected-path head inference with optional Markov features, and `--speculative-dflash-confidence-sts-path` loading with gamma validation. Head output takes precedence over selector-lattice confidence; existing head-less checkpoints retain the selector fallback.

`passing-after.log` records the same five regression tests passing. `focused-tests-final.log`, `compatibility-tests.log`, and `ragged-compatibility.log` cover the corrected and preserved Python paths. `gpu-confidence-reference.log` compares calibrated head output with an independent PyTorch expression on one AMD Instinct MI355X. `gpu-kernel-parity.log` retains 22 independent GPU numerical-reference subtests for the shared scheduling kernels.

No compatible DFlash2 target/draft checkpoint or trained-head checkpoint was supplied or cached. Therefore real checkpoint loading, end-to-end sampling/grammar/continuous-batching/overlap/live-graph semantics, and mixed-workload throughput remain unverified. The tiny Llama fixture is not architecture-qualified for those claims and was not used. No native source changed, so no native rebuild was applicable.
