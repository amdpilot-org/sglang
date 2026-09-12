# Investigation report: mamba track index during TARGET_VERIFY

Upstream issue: https://github.com/sgl-project/sglang/issues/34786

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1484

## Finding

The prepared `main` checkout already contains the reported crash fix. Upstream
PR #27998 (merge commit `1c4892d7bb5f0967aa54cd67b4923493f850a36e`)
changed `set_mamba_track_indices_from_reqs` to map an unallocated request's
`mamba_next_track_idx=None` to the initial ping-pong position 0. The current
implementation retains that guard in
`python/sglang/srt/managers/schedule_batch.py`.

It also contains the broader lazy/spec implementation from upstream PR #30437
(merge commit `7c9257529f0d4140bfdcc1f38a4b7b46c35562cd`):
`spec_prepare_for_decode` invokes `mamba_lazy_spec_prepare`, and
`prepare_mamba_track_for_verify` consumes the resulting per-request track plan.
Thus no additional runtime correction is justified at this base.

## Reproduction and regression

`raw/upstream_pr27998.diff` is the fetched upstream fix diff. Running its exact
pre-fix tensor expression with a fresh request (`mamba_next_track_idx=None`)
reproduced `TypeError: 'NoneType' object cannot be interpreted as an integer`;
see `raw/pre_fix_reproduction.log`.

The added GPU regression calls the actual checked-out helper with a device-side
ping-pong mapping. It covers:

- the reported fresh-request `None` state, selecting slot 0 without mutating the
  request's allocation state;
- a mixed batch containing `None`, slot 1, and slot 0;
- an explicit lazy-spec plan overriding request-local positions.

All cases passed on the assigned AMD Instinct MI350X (`gfx950`). The result is
selection/gather correctness for the scheduling helper, not a full-model
semantic or distributed serving qualification.

## Limitations

The reported Qwen3.6-27B-FP8 weights and two NVIDIA RTX 3090 GPUs were not
available. Consequently, the original two-GPU model-serving workload was not
rerun. The single-gfx950 test exercises the actual scheduling helper and GPU
gather but does not establish Qwen model accuracy, NVIDIA-specific behavior, or
multi-GPU behavior. No native code changed, so no native rebuild was applicable.
