# Consolidated correction for PR 2550

Upstream issue: https://github.com/sgl-project/sglang/issues/26399

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2600

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2550 at `8e74147151c20c79e779e97978f1f5b5af31e62c`

Review parent: https://github.com/amdpilot-org/sglang/pull/2596

## Result

The review's two concrete counterexamples reproduce. The exact candidate suite
passes (`3 passed`), but its runtime assertions also pass when `_async_d2h` is
replaced at runtime with a pinned, blocking `copy_(..., non_blocking=False)`.
Its lexical AST check likewise reports no forbidden conversion when
`run_eagle_verify` calls a module-local helper containing `Tensor.tolist()`.

The consolidated regression preserves the candidate's useful real-GPU value,
pinned-memory, and empty-result coverage and closes both holes:

- it records the real `Tensor.copy_` calls made by `GenerationBatchResult` and
  requires both EAGLE result tensors to use `non_blocking=True` from GPU to CPU;
- it computes the module-local call graph reachable from `run_eagle_verify` and
  rejects `.cpu()`, `.tolist()`, or `.numpy()` in reachable helpers, with an
  independent helper-indirection boundary case.

A controlled production-source mutation to `non_blocking=False` fails both GPU
parameter cases, and the restored implementation passes all four tests on the
assigned gfx950. No runtime source correction was justified: prepared `main`
already returns EAGLE verify results as device tensors and stages them through
the pinned nonblocking `_async_d2h` boundary.

## Limitations

The reported end-to-end hang remains unverified. This environment has one AMD
Instinct MI350X/gfx950 with ROCm 7.2, not 8x H200 CUDA GPUs, and lacks GLM-5.1
FP8 weights. Therefore TP8 coordination, CUDA/H200 behavior, GLM-5.1 semantics,
80K-160K prompts, near-full KV pressure, and the watchdog hang were neither
reproduced nor ruled out. The tiny serving fixture cannot qualify those missing
properties and was not used as substitute evidence.

