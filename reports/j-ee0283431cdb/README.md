# MXFP4 JIT hang correction generation 2

Candidate parent PR: https://github.com/amdpilot-org/sglang/pull/973 at
`f7d3ecf469e2a685b261ff1893152fe087087895`

Independent review parent PR: https://github.com/amdpilot-org/sglang/pull/1055

Upstream issue: https://github.com/sgl-project/sglang/issues/38408

Current mirror issue: https://github.com/amdpilot-org/sglang/issues/1814

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/1008

## Finding and correction

The review claims were reproduced against the exact candidate before it was
applied to the prepared base. The candidate's deterministic supervisor test
passes, and the complete JIT-cache suite passes after integration. A separate
process holding `_build_lock` alive kept a contender blocked until an external
two-second timeout returned 124. This is expected: the production lock remains
unbounded while its owner lives, while the registered MXFP4 test is bounded by
the candidate's isolated-process supervisor.

The supervisor self-test deliberately exits its successful worker before
unittest discovery, so it validates timeout, process-group termination, fresh
caches, and retry mechanics without claiming JIT or GPU coverage. A separate
run of the actual registered file reached the real JIT compiler on the assigned
gfx950 device. It invoked `/opt/rocm/bin/hipcc` with
`--offload-arch=gfx950:sramecc+:xnack-`, then failed because the NVIDIA-specific
source includes unavailable `cuda_bf16.h`. No MXFP4 GPU kernel executed.

No further source correction is justified from this environment. This branch
therefore preserves candidate `f7d3ecf` unchanged and corrects the deliverable
metadata by including the current mirror issue and both parent PR links.

## Failing before / passing after

Before candidate `f7d3ecf`, the registered test directly called
`unittest.main()` and had no internal deadline, isolated process group, fresh
per-attempt caches, retry, or deterministic supervisor regression. After the
candidate, `test_mxfp4_registered_test_retries_a_stalled_worker_attempt` passes
and the full `test_jit_cache.py` suite reports 45 passed. The live-owner
boundary remains blocked (external timeout exit 124), demonstrating that the
change is narrowly test-scoped rather than a speculative production-lock
timeout.

Formatting validation could not be rerun because the prepared interpreter has
no `ruff` module and no `ruff` executable. The exact preserved candidate
already contains passing ruff evidence under `reports/j-51be59b6eebc/raw/`.

Raw output is retained under `raw/`.
