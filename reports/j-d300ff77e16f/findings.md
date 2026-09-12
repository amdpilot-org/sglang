# Independent review of PR 2617

Upstream issue: https://github.com/sgl-project/sglang/issues/26399

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2618

Candidate: https://github.com/amdpilot-org/sglang/pull/2617 at `c6a20683551a8513301dd49d93c12cf78494683f`

## Recommendation

Accept as test-only hardening, not as independent proof that the original
production issue is fully resolved.

The candidate changes no runtime or native source. It strengthens the existing
EAGLE D2H regression in two useful ways: it verifies the actual `Tensor.copy_`
calls use pinned GPU-to-host destinations with `non_blocking=True`, and it
follows direct module-local helper calls from `run_eagle_verify` when looking
for `.cpu()`, `.tolist()`, and `.numpy()` conversions.

On the exact candidate, the focused suite passed all four tests on the assigned
AMD Instinct MI350X/gfx950. A controlled source mutation from
`copy_(..., non_blocking=True)` to `copy_(..., non_blocking=False)` made both GPU
parameter cases fail, confirming the primary counterexample from the prior
review is now covered. The helper-indirection fixture also passes and would
detect its direct `run_eagle_verify -> sync_helper -> Tensor.tolist()` case.

## Original failure and prepared base

The recorded base is exactly the image-prepared checkout commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no checkout discrepancy.
That base already contains the runtime redesign: EAGLE verification returns
device tensors, and `GenerationBatchResult.copy_to_cpu()` stages result tensors
through pinned host memory using `_async_d2h` and `non_blocking=True`. Therefore
the original synchronous `THPVariable_tolist` failure is not present on the
prepared base and could not be reproduced there. Historical source inspection
identified the later async D2H implementation in commit `8e1988b746f7ee3072ea2b0199efb01d8273c2eb`.

## Independent boundary case

The candidate's AST walker is intentionally incomplete. A module-level alias
(`alias = sync_helper`) followed by `run_eagle_verify -> alias(tensor)` bypasses
the scanner even when `sync_helper` calls `tensor.tolist()`. Imported helpers,
callable objects, and other forms of dynamic dispatch are similarly outside its
scope. This is a remaining static-test counterexample, although no such alias
was found in the current `run_eagle_verify` path.

## Environment and architecture limits

The imported package and manager utility came from `/job/repo/python`, using
`/tmp/amdpilot-repo-j-d300ff77e16f/venv/bin/python`, Torch `2.11.0+rocm7.2`, HIP
`7.2.26015`, and one AMD Instinct MI350X/gfx950. No native source changed, so no
native rebuild was applicable.

This environment lacks GLM-5.1 FP8 weights and has one AMD GPU rather than eight
H200 CUDA GPUs. TP8 coordination, CUDA/H200 behavior, 80K-160K prompts,
near-full KV pressure, and the watchdog hang remain unverified. The focused GPU
test establishes the result-transfer call contract and values only; it cannot
establish full resolution of the reported production hang.
