# Investigation report

The reported B300/CUDA 590 multi-context driver hang could not be reproduced on
the assigned single AMD MI355X (gfx950). The checked-out implementation did,
however, retain the issue's prerequisite: the first target-verify graph setup
created DSA metadata buffers and returned before launching
`fused_dsa_target_verify_metadata`. Consequently its Triton module load remained
deferred until the first graph replay/request on every rank.

The correction removes only that early return. Initial graph setup now continues
through the same metadata update path used by replay, which launches/JIT-loads
the target-verify kernel during startup. Existing metadata continues to bypass
buffer construction.

Evidence:

- `failing-before.log`: the regression fails against the original control flow
  because the target-verify kernel receives zero calls.
- `passing-after-gpu.log`: the regression and the existing DSA metadata kernel
  suite pass (10 tests and 4 subtests). The kernel suite executes on GPU and
  compares outputs exactly with independent PyTorch constructions.
- `gpu.json`: assigned GPU and Torch/ROCm identity.
- `related-pr-search.json`, `pr-29498.json`, and `pr-32109.json`: related-change
  inspection. Those merged changes implement/optimize fused replay metadata but
  do not remove the initial-build return or warm the first module load.

Limitations: no NVIDIA B300, CUDA 590.48.01, GLM-5.2 weights, eight ranks, or
Mooncake PD deployment were available. Thus this validates the source-level
lazy-load prerequisite and the candidate warmup behavior, not recovery from or
full reproduction of the reported CUDA driver deadlock.
