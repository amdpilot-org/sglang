# Corrected independent review of PR 1881

Reviewed exact candidate `b083b511e7dcc34ac3fd22b8b980bd6006c64037`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

- Upstream issue: https://github.com/sgl-project/sglang/issues/38408
- Candidate task mirror: https://github.com/amdpilot-org/sglang/issues/1814
- Review deliverable mirror: https://github.com/amdpilot-org/sglang/issues/1884
- Candidate: https://github.com/amdpilot-org/sglang/pull/1881
- Prior review being corrected: https://github.com/amdpilot-org/sglang/pull/1944

## Recommendation

Accept the candidate as an intentionally bounded, test-only mitigation. It
implements the original issue's proposed process-group supervision, fresh
per-attempt caches, deadline, and one retry, and its focused regression passes.
It does not fix the underlying compiler/lock/kernel stall or establish that the
intermittent H200 failure itself is gone, so `fully_resolves_original` is false.

The provenance allegation in PR 1944 is incorrect. Issue 1814 is the source
task that produced PR 1881 and was created before the candidate. Issue 1884 was
created later for PR 1944's review deliverable and was not a citation
requirement for the candidate. PR 1881 also cites historical review issue 1008
as its source task requested.

## Independent evidence

The image-prepared checkout exactly matched the recorded base. The base file
directly invoked `unittest.main()` without an internal deadline. A controlled
pre-result stall at that boundary remained alive until an external three-second
timeout killed it (exit 137). The actual base file entered the MXFP4 JIT build
and failed after running four tests because the assigned ROCm compiler cannot
find the CUDA-only `cuda_bf16.h` header.

On the exact candidate, four focused supervisor and lock tests passed in 33.30
seconds. An independent final-attempt injection bounded both workers and
returned 124. A live `_build_lock` owner still held a contender until an
external timeout killed the probe; this is the unchanged production-lock
boundary, not a regression introduced by the candidate. The self-test worker
exits before unittest discovery, so that regression proves supervisor mechanics
only.

The actual candidate file used fresh caches and reached `/opt/rocm/bin/hipcc`
with `--offload-arch=gfx950:sramecc+:xnack-`, then failed on `cuda_bf16.h`.
Thus the source/JIT path was exercised, but no MXFP4 GPU kernel executed and the
roughly 1% NVIDIA H200 stall could not be statistically reproduced.

## Environment and native scope

- Python: `/tmp/amdpilot-repo-j-132605abc3e2/venv/bin/python`
- SGLang import: `/job/repo/python/sglang/__init__.py`
- Torch: `2.11.0+rocm7.2` from `/opt/venv/lib/python3.12/site-packages/torch/`
- GPU: AMD Instinct MI355X, `gfx950:sramecc+:xnack-`
- NVIDIA H200: unavailable
- Native rebuild: not applicable; the candidate changes Python tests and reports only

Raw output is retained under `raw/`; fetched issue/PR metadata and exact diffs
are preserved outside the checkout in `/job/review-evidence-j-132605abc3e2/`.
