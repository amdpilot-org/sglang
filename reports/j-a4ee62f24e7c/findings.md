# Independent review of PR 2550

Upstream issue: https://github.com/sgl-project/sglang/issues/26399

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2490

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2555

Candidate: https://github.com/amdpilot-org/sglang/pull/2550 at exact commit `8e74147151c20c79e779e97978f1f5b5af31e62c`.

## Recommendation

Request changes. The candidate is test-only hardening over a prepared base that already contains the relevant runtime redesign; it is not a candidate runtime fix and does not fully resolve or reproduce the original production issue. Its tests pass, but they do not establish the central claimed invariant that the result transfer is nonblocking.

## Findings

1. The candidate's GPU test cannot detect a regression from `copy_(..., non_blocking=True)` to `copy_(..., non_blocking=False)`. It synchronizes `copy_done` before checking only CPU placement, pinning, and values. All of those assertions also pass after a blocking pinned D2H copy. Therefore the test name and report overstate what was measured: it validates the real GPU copy path and values, but not absence of scheduler-thread blocking.
2. The source guard checks only direct `.cpu()`, `.tolist()`, and `.numpy()` attribute calls lexically inside `run_eagle_verify`. Moving the same synchronous conversion to a helper and calling that helper from `run_eagle_verify` passes the guard. The retained adversarial AST case demonstrates this counterexample.
3. The exact candidate regression passes unchanged on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` (`3 passed`). This is consistent with the candidate's prose that the base already contains the runtime correction, but means this is not failing-before/passing-after coverage against the required comparison base.

## What is supported

The original v0.5.12 source snapshot contains direct device-to-host conversions in `EagleVerifyInput.verify`: `accept_index.tolist()`, `predict.tolist()`, `num_correct_drafts.cpu()`, and accepted-length `.tolist()` calls. At the prepared base, `run_eagle_verify` instead returns `predict` and `accept_lens` as device tensors in `GenerationBatchResult`; `GenerationBatchResult.copy_to_cpu` routes them through `_async_d2h`, which allocates pinned memory and requests a nonblocking copy before the scheduler result processor converts the CPU tensors to lists. The candidate's three tests pass on the assigned gfx950 and correctly validate values, pinned allocation for nonempty tensors, the empty boundary, and absence of direct forbidden method calls in the current function body.

No native source changes are present in the candidate, so no native rebuild was applicable. Imports resolved to `/job/repo/python/sglang/...`, confirming tests used checkout Python sources rather than an installed SGLang package.

## Original-issue status and limitations

The recorded base already has a plausible fix for the reported synchronous conversions, and the candidate adds only incomplete regression coverage. The full issue remains unverified here: the environment provides one AMD Instinct MI355X (`gfx950`, ROCm 7.2), not eight H200 CUDA GPUs; GLM-5.1 FP8 weights are unavailable; TP8, 80K-160K prompts, near-full KV pressure, CUDA D2H behavior under memory saturation, and the 300-second watchdog hang were not reproduced. A successful small tensor transfer cannot prove that the production deadlock is eliminated.

Raw command output, candidate metadata/diff, source excerpts, import paths, and adversarial results are under `reports/j-a4ee62f24e7c/evidence/`.
