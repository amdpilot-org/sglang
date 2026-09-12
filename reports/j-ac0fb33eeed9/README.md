# Independent review of PR 1838

Upstream issue: https://github.com/sgl-project/sglang/issues/33642

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1882

Candidate: https://github.com/amdpilot-org/sglang/pull/1838 at
`074a97cefd38979cd1d35f5cb424c201914512bc`.

## Recommendation

Accept the narrow source-level fix. It removes the demonstrated early return
that deferred `fused_dsa_target_verify_metadata` until the first graph replay.
This is a partial, issue-directed fix rather than proof that the complete
reported deployment no longer hangs.

`fully_resolves_original` is false because the assigned environment has one AMD
Instinct MI350X (`gfx950`, ROCm 7.2), not eight NVIDIA B300 (`sm_103`) GPUs with
CUDA driver 590.48.01. GLM-5.2 weights, Mooncake PD disaggregation, and an
eight-rank TP/DP/EP deployment were also unavailable. In particular, moving
module loading into startup does not by itself prove that eight contexts cannot
deadlock if they load the module concurrently during startup.

## Source and artifact checks

The prepared checkout was clean at the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference between
the image-prepared checkout and the requested failing-before revision. The
candidate was fetched from `refs/pull/1838/head`, resolved exactly to the
requested SHA, temporarily checked out detached, and the checkout was returned
to `amdpilot/j-ac0fb33eeed9` before this report was committed.

Imports during candidate testing resolved to:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/layers/attention/dsa_backend.py`
- `/job/repo/python/sglang/kernels/ops/attention/dsa_metadata.py`

The interpreter was
`/tmp/amdpilot-repo-j-ac0fb33eeed9/venv/bin/python`, with Torch
`2.11.0+rocm7.2` and HIP `7.2.26015`. The candidate changes Python control flow
and a Python test only; it changes no native C++/FlyDSL source, so no native
rebuild was applicable. The real Triton kernel was executed from the repository
source on gfx950.

## Evidence

On the base, the candidate regression produced one failure and one pass. The
failure was the issue-specific assertion: initial target-verify graph setup
called `fused_dsa_target_verify_metadata` zero times. The existing DSA metadata
kernel suite independently passed on the GPU (8 tests and 4 subtests), isolating
the observed defect to warmup/control flow rather than numerical kernel output.

At the exact candidate commit:

- The candidate regression passed: 2 tests.
- The existing GPU DSA metadata suite passed: 8 tests and 4 subtests. These
  execute decode, target-verify, and draft-extend Triton kernels and compare
  outputs with PyTorch references.
- Independent review-only adversarial tests passed: 4 tests covering first
  construction and cached reuse for batch sizes 1, 2, and 7, plus initial
  decode-mode setup. The same tests failed 4/4 on the base.

Raw logs, the candidate diff, upstream issue snapshot, related-PR search, and
the independent test source are preserved outside the revision-switching
checkout at `/job/review-evidence-j-ac0fb33eeed9/`.

## Remaining counterexamples and limitations

- No B300/sm_103 or CUDA 590.48.01 was available, so `cuModuleLoadData` behavior
  on the reported driver was not exercised.
- Only one GPU was assigned; simultaneous module loading by eight independent
  scheduler contexts was not reproduced at request time or startup.
- No GLM-5.2 weights or Mooncake PD deployment were available, so the complete
  first EAGLE verify serving path was not run.
- The issue reports an internal fork; behavior unique to that fork cannot be
  qualified from the prepared mirror base.
- The candidate establishes that startup reaches the kernel and loads its
  module in this environment. It does not establish that moving the same
  concurrent load earlier is sufficient for the NVIDIA driver failure.

