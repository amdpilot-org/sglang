# Independent review of PR 2113

Candidate: `3ac707e08537890da1a90372c58f2718e5b8789a`

Upstream issue: https://github.com/sgl-project/sglang/issues/32507

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2054

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2147

## Verdict

Request changes. The candidate is a functional partial fix, not test-only hardening: a freshly compiled native library copies the reported `N=16` vector exactly on the assigned gfx950 GPU, and independent non-specialized sizes work through 512. However, the implementation replaces the old three-size restriction with an undocumented `N <= 512` restriction. `N=513` is still routed unconditionally through `copy_to_gpu_no_ce` by the production OffloaderV2/DeepEP path and fails with `copy_to_gpu_no_ce supports at most 512 elements`. No production validation was found that makes 512 the maximum valid local-expert count. This does not meet the issue's stated arbitrary-valid-count contract.

## Evidence

The recorded base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`; it matched the prepared checkout. Its actual native source was HIP-translated and rebuilt with `/opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx950`, then loaded directly as a focused Torch extension. The base copied N=32, 64, and 72 exactly and rejected N=16 and N=17 with `unexpected N`.

The checkout was then temporarily detached at the exact candidate commit and its changed `copy.cu` was independently rebuilt into a separate extension cache. Exact integer comparisons passed for N=1, 15, 16, 17, 31, 32, 33, 64, 72, 127, 255, 256, 511, and 512. N=513 and N=1024 were rejected by the new cap. Invalid dtype, shape, contiguity, and output-size cases were also rejected. The generated HIP sources and complete compiler/runtime logs are retained in the job evidence directory outside the checkout; copies of the raw logs are included here.

The prepared source import resolves SGLang code from `/job/repo/python`, while `sgl_kernel` resolves to the prepared ROCm egg under `/opt/venv/lib/python3.12/site-packages`; that installed native package does not register `copy_to_gpu_no_ce`. Therefore both revisions were validated using their checked-out native source, not the wheel.

## Limitations

The assigned architecture was one AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not NVIDIA H800/CUDA. The GLM-5.2 weights and two-node TP16/EP16 DeepEP topology were unavailable. Consequently, this review verifies the failing and fixed native transfer behavior but does not claim a full model-serving, NVIDIA compilation, or distributed reproduction.
