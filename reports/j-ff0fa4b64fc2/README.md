# Independent review of PR 1706

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1706 at exact
commit `461a16aa4fd3436ce6882fc53dc9f00f5fb97a11`.

Upstream issue: https://github.com/sgl-project/sglang/issues/34786

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1742

## Finding

Recommendation: **accept as test-only hardening**. The candidate changes no
runtime or native source. Its added regression exercises real scheduler/cache
methods for fresh lazy allocation, speculative boundary planning, TARGET_VERIFY
index construction, and confirmed-boundary promotion/freeing. It passed on the
assigned AMD Instinct MI350X (`gfx950`). Independent adversarial cases covering
no-boundary behavior, allocation failure, pending-slot reuse, and an exact
window-edge crossing also passed.

This does **not** fully resolve the original issue by itself. The recorded base
already contains the runtime `None -> 0` guard and the broader lazy speculative
lifecycle fixes. On that base, the historical unguarded tensor expression still
reproduces the reported `TypeError`, while the actual checked-out helper safely
selects slot 0 and leaves the request field unmodified. The candidate adds
meaningful regression coverage for that existing source fix; it is not a new
runtime correction and it does not provide end-to-end serving proof.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: historical
  expression raised `TypeError`; actual `set_mamba_track_indices_from_reqs`
  returned GPU index `101` using position 0.
- Exact candidate: `4 passed` in
  `test/registered/unit/managers/test_mamba_track_indices.py`.
- Independent adversarial review: `4 passed` using production methods and GPU
  tensors.
- Adjacent lifecycle suite: `3 passed, 2 subtests passed`.
- Imports resolved to `/job/repo/python/sglang/...`; Torch was
  `2.11.0+rocm7.2`. No native/C++/HIP source changed, so rebuilding native code
  was not applicable.

Raw command output is retained in `raw/`. The external adversarial harness used
during detached-candidate review is retained as `adversarial_mamba_review.py`.

## Remaining limitations

No fresh-request NEXTN TARGET_VERIFY request was served end to end on a
hybrid-mamba model. Qwen3.6-27B-FP8 weights and the reported two RTX 3090 GPUs
were unavailable. NVIDIA, multi-GPU, hybrid-mamba kernel semantics, model
accuracy, HTTP serving, and distributed execution therefore remain unverified.
The AMD `gfx950` scheduler/cache tests cannot substitute for those checks.
