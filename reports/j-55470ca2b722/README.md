# Investigation of fast_topk_v2 candidate-buffer overflow

Upstream issue: https://github.com/sgl-project/sglang/issues/36807

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1091

## Result

The reported CUDA implementation remains present in
`python/sglang/kernels/aot/csrc/elementwise/topk.cu`: candidates after
`SMEM_INPUT_SIZE` are still omitted from both the shared candidate array and the
refinement histogram. The assigned machine has one AMD gfx950 GPU and cannot execute
or validate that CUDA kernel.

Current main already contains a separate ROCm implementation. `setup_rocm.py` builds
`csrc/elementwise/topk.hip` instead of `topk.cu`; that implementation uses the
cooperative kernel in `include/hip/dsa_topk_coop.cuh`. Its documented and implemented
overflow path rescans the full row under the resolved radix prefix rather than ranking
the truncated tie buffer. The installed native extension contains the corresponding
`coop_topk_kernel<2048, 12, 4096, 1024>` symbol (see `native_symbols.log`).

On the assigned gfx950, the issue's standard-normal case produced threshold buckets
of 4194--4464 entries at 64 x 256K, exceeding the 4096 tie capacity, while returning
the exact `torch.topk` value multiset in all 64 rows. A 4 x 1M independent long-row
case also returned exact results with bucket populations of 4986--5213. The 64 x 64K
below-boundary control was exact as well. Raw numerical output is retained in
`reproduce_topk_overflow.log`.

No CUDA correction is proposed from an AMD-only run. Doing so would leave the changed
kernel unexecuted and would not meet the issue's requirement for failing-before and
passing-after CUDA evidence. The original CUDA behavior therefore remains unverified
on this host and the CUDA source still exhibits the reported truncation mechanism.

## Reproduction

Run with the prepared interpreter and the assigned GPU:

```bash
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-55470ca2b722/venv/bin/python \
  reports/j-55470ca2b722/reproduce_topk_overflow.py
```

The focused checked-in boundary tests were also run on the GPU; their output is in
`pytest_topk_boundaries.log`.
