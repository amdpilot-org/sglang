# MXFP4 JIT hang correction generation 1

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/799 at
`4f47bef5921d53d3a40186d03225601411530334`

Independent review: https://github.com/amdpilot-org/sglang/pull/871

Upstream issue: https://github.com/sgl-project/sglang/issues/38408

Mirror issue: https://github.com/amdpilot-org/sglang/issues/907

## Finding

The candidate's advisory-lock regressions are valid: an ownerless lock file is
ignored and the kernel releases a lock when its owner dies. They cover the
stale/dead-owner failure fixed by the existing `load_jit` migration, but not a
live compiler or kernel stall. On the exact candidate, a second `_build_lock`
acquisition remained blocked until the independent external timeout exited 124.
The candidate also changed only the generic JIT-cache test; it did not bound the
registered MXFP4 test, isolate its process group, create fresh per-attempt JIT
caches, or retry it. Its PR body references mirror issue 757 rather than the
review-required issue 839.

## Correction

The registered MXFP4 file now supervises a clean worker process. Each attempt
gets a new `SGLANG_JIT_CACHE_DIR` and `TORCH_EXTENSIONS_DIR`, runs in a new
session/process group, and has a 120-second deadline. A timeout kills the whole
group, including compiler descendants, and retries once with another fresh
cache. Ordinary test failures are returned immediately rather than hidden by a
retry. The candidate's two valid advisory-lock regressions are retained.

The deterministic supervisor regression runs the registered MXFP4 executable,
forces its first worker to stall, observes the internal timeout and process-group
kill, and verifies that its second fresh-cache worker succeeds. Before this
correction, the registered file had no such supervisor or internal deadline.

## Hardware limitation

The assigned device is AMD Instinct MI355X (`gfx950`), not the reported H200.
The actual cold registered test reached `/opt/rocm/bin/hipcc` with
`--offload-arch=gfx950:sramecc+:xnack-`, then the NVIDIA-specific source failed
at missing `cuda_bf16.h`. No MXFP4 GPU kernel executed. This does not validate
the H200 kernel numerics or statistically reproduce the intermittent H200
stall; it only confirms that the corrected registered entry reaches its real
JIT path with fresh cache isolation on the available system.

Raw commands and outputs are retained in `raw/`.
