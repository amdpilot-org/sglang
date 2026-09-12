# Independent review of PR 1662

Reviewed exact candidate commit `ef54a0c02a8fd820d7de9ee63a7fe643b6180406` against upstream issue https://github.com/sgl-project/sglang/issues/35257 and mirror issue https://github.com/amdpilot-org/sglang/issues/1700.

## Finding

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduced the tie-buffer overflow as a value-exactness failure while still returning valid unique indices. At the first overflow boundary (2049 candidates), 11 of 2048 selected values differed from `torch.topk`; larger overflowing bands produced 1867-2048 mismatches.

At the exact candidate, the same cases had zero value-multiset mismatches on the assigned AMD Instinct MI355X (gfx950). Independent cases covering k=512/1024/2048, shuffled arrival order, negative values, mixed row lengths, positive subnormals, two-value overflow, exact ties, and signed zero also had zero mismatches. The candidate's full top-k v2 test file passed 289/289 tests.

Source inspection confirms that CUDA no longer dispatches any row to the known-inexact `TopKCluster`: `kEnableClusterPath` is false and gates `use_cluster`, so the original batch=1, N=262144 shape is routed to `TopKStreaming`. On gfx950, a fresh private JIT cache rebuilt and executed the changed native source from `/job/repo/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` and `/job/repo/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`; Python imports resolved to `/job/repo/python`.

## Recommendation and limitation

Recommendation: **unverified**. No runtime counterexample was found, and the candidate is a plausible complete correctness-first source fix. However, the original failure and the final dispatch change are CUDA/B200-specific. This host has ROCm 7.2 and gfx950 only, no `nvcc`, and cannot compile or execute CUDA thread-block clusters. Therefore CUDA compilation, the B200 N=262144 fixture, and CUDA fallback performance remain unverified. Disabling the cluster path may regress long-row/small-batch CUDA performance.

Raw logs and the exact independent harnesses are preserved outside the checkout at `/job/review-evidence-j-77beeebbd696/`.

