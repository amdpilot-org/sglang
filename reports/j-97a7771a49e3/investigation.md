# Investigation: sglang#31023

Upstream issue: https://github.com/sgl-project/sglang/issues/31023

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2327

## Finding

The prepared base already contains the source correction merged as upstream PR
https://github.com/sgl-project/sglang/pull/32467.  In
`python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`, every warp writes its
own `warp_max` and `warp_min` reduction slot and the existing block barrier
publishes those writes before warp 0 consumes the slots.  The old redundant
initialization, which could race with those writes, is absent.

No production source change was justified.  This PR adds the missing focused
GPU regression for the real GPU-input `plan_prefill` path.  It checks the two
ragged shapes from the issue, the uniform control, and independent partial-warp,
full-1024-thread-block, and zero-extend boundaries.  The oracle requires the
valid write-plan ragged IDs to be exactly `range(sum(extend_lens))`.

## GPU evidence

The prepared device was one AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, with
PyTorch 2.11.0+rocm7.2 and ROCm 7.2.26015.

The fixed implementation completed 9,000 direct planner launches with no bad
plans.  In particular:

```text
uniform [4] * 96:                 0 / 2000 bad, max ragged_id 383 of 383
ragged [4] * 72 + [3] * 24:      0 / 2000 bad, max ragged_id 359 of 359
ragged [3] * 104 + [2] * 24:     0 / 2000 bad, max ragged_id 359 of 359
partial-warp boundary:            0 / 1000 bad
1024-thread boundary:             0 / 1000 bad
zero-extend boundary:             0 / 1000 bad
```

Raw output is retained outside the worktree at
`/tmp/amdpilot-repo-j-97a7771a49e3/evidence/fixed_gpu_planner_reproduction.log`.

For a local failing-before attempt, the five redundant scratch-initialization
lines removed by #32467 were temporarily restored and the two reported ragged
shapes were each run 2,000 times.  They did not fail on gfx950 (0/4,000 bad).
That source experiment was reverted before the committed change.  Its raw log
is `/tmp/amdpilot-repo-j-97a7771a49e3/evidence/historical_unpatched_gpu_planner.log`.
This does not contradict the timing-sensitive NVIDIA B300 evidence, but means a
local failing-before reproduction is unavailable on the assigned AMD GPU.

## Limitations

- Only one gfx950 GPU was assigned, so TP8, cross-rank behavior, NVIDIA B300
  scheduling, CUDA Graph capture/replay, and NCCL failure surfacing were not
  reproduced.
- DeepSeek-V4-Pro-DSpark weights were unavailable.  No full-model or semantic
  accuracy claim is made.
- The deterministic tiny Llama serving fixture is unrelated to this DSV4
  architecture-specific planner kernel and therefore was not used as a
  substitute reproduction.
- No native FlyDSL component was changed or rebuilt.
