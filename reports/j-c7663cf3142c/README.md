# Independent review of PR 973

Candidate: https://github.com/amdpilot-org/sglang/pull/973

Exact commit: `f7d3ecf469e2a685b261ff1893152fe087087895`

Upstream issue: https://github.com/sgl-project/sglang/issues/38408

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1008

## Verdict

`request_changes`. The candidate is test-only hardening, but its code fully
implements the original issue's requested operational mitigation: the
registered MXFP4 file runs each attempt in a new session/process group, gives
each attempt fresh SGLang and Torch extension caches, enforces a 120-second
deadline, kills the timed-out process group, and retries once. It does not fix
the underlying compiler or kernel stall; it prevents such a stall from
consuming the enclosing 30-minute CI shard.

The requested change is metadata-only: the candidate PR body does not contain
the required current mirror issue URL
`https://github.com/amdpilot-org/sglang/issues/1008`. It instead names mirror
issues 907 and 839. No candidate code was modified in this review.

## Independent evidence

- The prepared checkout exactly matched recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365` before review.
- The exact candidate commit was checked out detached and all imports resolved
  to `/job/repo/python/sglang`, including the MXFP4 operation and JIT loader.
- On the base, a cold-cache run of the real registered MXFP4 file reached the
  in-tree JIT source and `/opt/rocm/bin/hipcc`, then failed because the
  NVIDIA-specific source includes unavailable `cuda_bf16.h`.
- On the candidate, the three focused regressions passed, including the
  registered-file supervisor regression: `3 passed`.
- The complete candidate JIT-cache test file passed: `45 passed`.
- An independent terminal-boundary case with a 0.001-second deadline timed out
  both attempts, printed two distinct fresh-cache roots, killed each process
  group, and returned 124.
- A live holder of production `_build_lock` still kept an independent waiter
  blocked until an external 3-second kill. This is expected: the candidate
  contains that condition at the registered-test process boundary rather than
  adding a timeout to the general-purpose build lock.
- Ruff could not be rerun with the mandated prepared interpreter because that
  environment has no `ruff` module.

## Architecture and native-code limits

The assigned device is AMD Instinct MI350X, gfx950, with ROCm 7.2 and PyTorch
2.11.0+rocm7.2. The issue was reported on NVIDIA H200. The actual candidate
worker reached `hipcc` with `--offload-arch=gfx950:sramecc+:xnack-`, but failed
at `expert_pack_mxfp4.cuh:14` because `cuda_bf16.h` is unavailable. Therefore
no MXFP4 GPU kernel executed, no numerical result was produced, and neither
H200 behavior nor the intermittent approximately 1% stall was reproduced.

The candidate changes only Python tests and report files. No production or
native source changed, so no native rebuild was applicable.

Detailed raw logs were preserved outside the revision-switching checkout at
`/job/review-evidence-j-c7663cf3142c/`.
