# Investigation of sglang issue 30936

Upstream issue: https://github.com/sgl-project/sglang/issues/30936

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2340

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The prepared source already contains the issue-specific solution. No additional
code change is justified from the available evidence.

The reported failure is an NVCC/cicc exit 139 while compiling
`deepseek_v4/topk_v2.cuh` for Hopper during DeepSeek-V4 CUDA-graph capture. The
upstream discussion independently narrowed it to the Hopper cluster kernels.
Upstream PR https://github.com/sgl-project/sglang/pull/32910 then recorded a
direct CUDA 13.2 reproduction and fixed the compiler trigger: a cluster-mapped
DSMEM pointer stored in `problem.out` reached a later load in
`problem_transform`. Keeping that mapped alias in a temporary copy prevented
the CUDA 13.x optimizer crash without changing which shared-memory bytes the
elected rank consumed.

Current main has since refactored this code, but retains the same invariant in
`python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`: immediately before the
epilogue load it asserts that the elected rank's `problem.out` is its local
`s_topk_indices`. The adjacent source comment explicitly identifies the CUDA
13.1+ cicc failure, issue #32830, and the earlier #32910 workaround. Raw copies
of the original fix and current source are retained under `raw/`.

## Validation

The assigned accelerator is an AMD Instinct MI355X (`gfx950`) using Torch
2.11.0+rocm7.2. Three real-GPU top-k cases passed against the independent
`torch.topk` reference:

- sequence 8192, the Register2 upper boundary;
- sequence 8193, the Register4 lower boundary, with a permuted page table;
- sequence 16385, the Streaming lower boundary.

These tests establish that the current HIP top-k implementation behaves
numerically at independent dispatch boundaries. They do not exercise the
CUDA-only cluster branch guarded by `#ifndef USE_ROCM` and therefore are not
evidence that NVCC is fixed.

## Exact blockers

`command -v nvcc` returned 1. The environment contains ROCm 7.2, not CUDA, and
the only assigned GPU is gfx950 rather than H800/sm_90. The reported model
weights are absent, and the TP=8/EP=8 reproduction requires eight NVIDIA GPUs.
Consequently, this investigation could not rerun either the failing-before and
passing-after CUDA 13.1 compiler regression or the full serving command. The
tiny Llama transport fixture would not exercise the DeepSeek-V4 top-k JIT
translation unit and was therefore not substituted for the reported bug.

Raw evidence:

- `raw/upstream-issue-30936.txt`
- `raw/pr-32910.json`
- `raw/pr-34167.json`
- `raw/original-fix.diff`
- `raw/current-topk-small-batch.txt`
- `raw/environment.txt`
- `raw/gfx950_topk_boundaries.log`
