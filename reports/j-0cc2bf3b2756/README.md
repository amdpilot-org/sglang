# Independent review of PR 1410 at `74a54568046e3adaa9182fd609eab498a714bd36`

Recommendation: **request changes**. The candidate is a partial race fix, not a full resolution of the original issue.

The exact candidate passes its focused regression and correctly rejects an older successful probe when a re-registration and later failure recreate the same failure count. The recorded base fails that regression as expected. Imports were verified to resolve `sglang` and `MooncakeKVManager` from `/job/repo/python`, using the prepared interpreter.

An independent ordering case still fails. In the registration path, the candidate first writes `decode_kv_args_table[mooncake_session_id]` and only afterward calls `_mark_session_registered()`. A transfer thread can therefore observe and use the newly published arguments, fail, and add the session to `failed_sessions`; the subsequent registration bookkeeping unconditionally discards that failure. The reproducer in `raw/adversarial_candidate.py` expresses this source ordering and fails on the exact candidate. Moving or combining state transitions would be an implementation decision for the candidate author; this review does not modify the candidate.

The candidate also does not identify the cause of recurring transfers failing in the reported 5P3D GLM5.2 NVIDIA/XCCL RoCEv2 deployment. Its recovery probe may improve availability after some transient failures, but that is not evidence that the original deployment defect is resolved.

## Evidence

- `raw/candidate_regression_on_base.log`: 4 failed, 10 passed, 3 subtests passed on base `358c163`; this establishes failing-before behavior.
- `raw/candidate_regression.log`: 14 passed, 3 subtests passed on exact candidate `74a5456`.
- `raw/candidate_adversarial.log`: one independent case fails and the reviewed equal-count ABA case passes.
- `raw/base_imports.log` and `raw/candidate_imports.log`: measured Python, Torch/ROCm, repository import paths, and visible GPU.
- `raw/candidate_source_lines.log`: exact candidate source ordering relevant to the remaining race.
- `raw/compileall.log` and `raw/diff-check.log`: both commands exited zero.

## Environment limits

The prepared host exposes one AMD Instinct MI350X (`gfx950`) with Torch `2.11.0+rocm7.2`. The report requires NVIDIA compute capability 8.6, XCCL, eight GPUs per node, a multi-node 5P3D layout, RoCEv2, and GLM5.2 weights. Those were unavailable, so no full deployment reproduction or GPU execution is claimed. No native files changed, making a native rebuild inapplicable.

Upstream issue: https://github.com/sgl-project/sglang/issues/37022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1448

Candidate: https://github.com/amdpilot-org/sglang/pull/1410
