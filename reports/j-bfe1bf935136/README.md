# Independent review of PR 1881

Reviewed exact candidate `b083b511e7dcc34ac3fd22b8b980bd6006c64037`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

- Upstream issue: https://github.com/sgl-project/sglang/issues/38408
- Required mirror issue: https://github.com/amdpilot-org/sglang/issues/1884
- Candidate: https://github.com/amdpilot-org/sglang/pull/1881
- Candidate parent: https://github.com/amdpilot-org/sglang/pull/973 at
  `f7d3ecf469e2a685b261ff1893152fe087087895`
- Prior independent review: https://github.com/amdpilot-org/sglang/pull/1055

## Verdict

Request changes. The implementation is a valid, bounded, test-only mitigation,
but it does not fully resolve or reproduce the intermittent H200 failure. The
exact PR body also names mirror issue `1814`, rather than the required current
mirror issue https://github.com/amdpilot-org/sglang/issues/1884. It does include
the separately required historical review issue `1008`.

## Evidence

The prepared checkout exactly matched the recorded base. On that base, the
registered test had no internal supervisor; a fault-injected pre-result stall
survived until the independent three-second external timeout killed it. The
actual base test entered the real JIT path and failed after 19.85 seconds because
HIP compilation of the CUDA-only source could not find `cuda_bf16.h`.

On the exact candidate, four focused regressions passed in 38.04 seconds,
including the deterministic first-attempt stall/retry test and three advisory
lock boundary tests. An actual candidate run used a fresh
`SGLANG_JIT_CACHE_DIR`, invoked `/opt/rocm/bin/hipcc` for
`gfx950:sramecc+:xnack-`, and reached the same `cuda_bf16.h` failure. This proves
the source and JIT path used by the candidate, but no MXFP4 GPU kernel executed.

Independent short-deadline probes confirmed that both attempts are bounded and
return 124. They also show the deadline includes Python/Torch/module startup:
with 0.5-1 second deadlines, even the nominally successful self-test attempt can
time out before reaching the self-test branch. This is not a counterexample to
the production 120-second value, but it defines the mechanism's boundary.

The production `_build_lock` remains an unbounded blocking `fcntl.flock` while
its owner is alive. The candidate correctly tests orphan/dead-owner recovery;
it does not change production lock behavior. The supervisor self-test exits
before unittest discovery and therefore does not validate MXFP4 compilation or
kernel execution.

## Environment

- Python: `/tmp/amdpilot-repo-j-bfe1bf935136/venv/bin/python`
- SGLang import: `/job/repo/python/sglang/__init__.py`
- Torch: `2.11.0+rocm7.2` from
  `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`
- GPU: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`
- Candidate native changes: none; no native rebuild was applicable
- NVIDIA H200: unavailable

Raw logs and fetched issue/PR metadata were preserved outside the revision
switch at `/job/review-evidence/`.
