# Independent review of PR 1829

Upstream issue: https://github.com/sgl-project/sglang/issues/33656

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1871

Candidate: https://github.com/amdpilot-org/sglang/pull/1829 at
`67d1d2f2dc265f1475651e35986b430aab3a960a`.

## Finding

The candidate correctly fixes a real, narrowly scoped accounting defect in
`HybridCacheController._page_backup`: on an MLA follower rank that deliberately
has no local writes, an empty `backup_transfers` set must be treated as
successful rather than reporting zero completed tokens. The candidate's focused
test failed on the recorded base (`0 != 128`) and passed on the exact candidate.
Independent cases also confirmed that `None`, an empty list, and filtered
replicated transfers are successful, while missing, short, and partially false
results for selected rank-local transfers remain failures.

This does not fully verify or reproduce the original issue. The reported
contract is a successful large cached-prefix restore followed by deterministic
`TAIL_K_SWA` position corruption (`512` stored versus `8448` expected) and NaN
sampling. The candidate changes only backup completion accounting and adds no
test that restores a FULL+SWA DeepSeek-V4 prefix, rebuilds the FULL-to-SWA
mapping, or checks written SWA positions. On the base, the demonstrated defect
reports the follower backup as incomplete; that can suppress or discard cached
state, but the submitted evidence does not show how it creates a successful
restore with a wrong SWA position. Thus this is a partial fix / hardening of an
issue-adjacent prerequisite, not proof that the original production corruption
is resolved.

## Environment and scope

The prepared checkout exactly matched the recorded base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The candidate was temporarily
checked out detached and source import was confirmed from
`/job/repo/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`.
The diff contains Python only, so no native rebuild was applicable.

The available machine has one AMD gfx950 GPU with ROCm 7.2 and Torch
2.11.0+rocm7.2. It does not provide the reported 8x NVIDIA H20 CUDA 13 topology,
DeepSeek-V4-Flash-0731 weights, DSPARK production configuration, or hours-long
workload. No GPU execution was used because the changed path and tests are
host-side accounting; an unrelated GPU smoke would not validate the issue.

Raw outputs and the reviewed diff are retained in `evidence/`.
