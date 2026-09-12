# Investigation of sglang#33356

Upstream issue: https://github.com/sgl-project/sglang/issues/33356

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1902

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

Current main already contains the primary producer-side correction from upstream
PR #32467 (merge commit `8549cce11b878d2fbf814d5e27bcc5e626890e70`).
The fix removes five redundant initializations of `warp_min`/`warp_max` in
`plan_compress_prefill_kernel0`. With the fixed 1024-thread launch, all 32 warps
write their own scratch slot before the existing reduction barrier, so removing
the competing writes eliminates the reported write/write race.

No production change is proposed. The included regression calls the actual
GPU-input `plan_prefill` implementation and checks the independent invariant
`max(ragged_id) < sum(extend_lens)` for the issue's uniform control and both
ragged boundary shapes.

## Evidence

On the assigned single AMD Instinct MI350X/gfx950, current source produced zero
out-of-bound plans in 6,000 calls:

```text
uniform [4] * 96:               0 / 2000 bad, max 383, bound 383
ragged [4] * 72 + [3] * 24:    0 / 2000 bad, max 359, bound 359
ragged [3] * 104 + [2] * 24:   0 / 2000 bad, max 359, bound 359
```

As a negative control, the five removed lines were temporarily restored and the
same 6,000 calls were run with an isolated JIT cache. The old scheduling race did
not reproduce on gfx950. The checkout was then restored. This is expected to be
architecture/scheduling sensitive and is recorded honestly rather than treated
as proof that the historical source was correct.

The existing DSV4 plan suite also passed: 5 tests and 82 subtests.

## Related changes inspected

- #32467 is merged and present in this base; it is the issue-specific producer
  correction.
- #33795 remains open and adds a Full CUDA Graph post-warmup completion point.
- #34286 remains open and proposes the analogous Breakable CUDA Graph boundary.
  Both explicitly describe separate capture-ordering hardening and are not the
  primary B300 producer correction. They were not duplicated here.

## Limitations

The assigned environment has one gfx950 under ROCm 7.2. It does not have eight
B300/B30Z GPUs, CUDA 13, the DeepSeek-V4-Pro-DSpark weights, or the original
TP8 topology. Therefore this work does not reproduce or qualify the full model
startup/capture failure, host SIGSEGV modes, cross-rank behavior, or downstream
CUDA surfacing kernels. The direct fixture validates only GPU plan generation.
