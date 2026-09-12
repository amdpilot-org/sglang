# EAGLE verify near-full KV investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/26399

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2490

The prepared `main` revision already contains the relevant design correction, so this change adds regression coverage and records the hardware limitation instead of duplicating a runtime fix.

In the reported v0.5.12 source, `EagleVerifyInput.verify` performs synchronous host conversions at lines 424-425 and 485-486 (`accept_index.tolist()`, `predict.tolist()`, and accepted-length `.tolist()` calls). The current implementation has moved verification to `run_eagle_verify`, which returns `predict` and `accept_lens` as device tensors. `GenerationBatchResult.copy_to_cpu` later passes both through `_async_d2h`, whose GPU path allocates pinned host memory, issues `copy_(..., non_blocking=True)`, and records the source tensor on the active stream for lifetime safety.

This mechanism arrived in upstream PR https://github.com/sgl-project/sglang/pull/29075 (merge commit `8e1988b746f7ee3072ea2b0199efb01d8273c2eb`), whose stated purpose includes speculative `accept_lens` and avoiding scheduler-thread stalls from pageable result D2H copies.

The added regression would reject the reported implementation's host conversion in the verify hot path. On the assigned gfx950 it also checks exact token and acceptance-length values after the real result-copy path for a two-request result, plus the empty-batch boundary.

Raw investigation artifacts are retained under `/tmp/amdpilot-repo-j-f2862992c228/evidence/`, including the v0.5.12 source excerpt, upstream commit/PR metadata, issue timeline, GPU test output, and the initial regression refinement output.

The full production report remains unverified: GLM-5.1 FP8 weights and eight H200 GPUs were unavailable, and a single gfx950 cannot reproduce TP8 topology or CUDA/H200 near-full-KV behavior.
