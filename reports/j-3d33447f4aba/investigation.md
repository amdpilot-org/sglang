# Investigation: DeepSeek-V4 long-context top-k illegal access

Upstream issue: https://github.com/sgl-project/sglang/issues/34718

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1524

## Finding

The prepared `main` checkout already contains the narrow correction identified for
this failure class by merged upstream PR
https://github.com/sgl-project/sglang/pull/34167 (merge commit
`accc51c6dbe5ad59110b0fc405a979be9b61fd68`). No additional source correction is
justified in this checkout.

The reported `fp8_paged_mqa_logits` frame can surface a sticky asynchronous CUDA
error from the preceding top-k kernel. The issue's observed raw-token band maps to
the fused top-k v2 dispatch, and PR #34167 established a CUDA 13.1+ SM90a compiler
mislowering when a single pointer selected between distributed shared memory
(`problem.out`) and CTA-local shared memory (`smem->tmp_out`). That corrupted top-k
indices later consumed by sparse attention.

Current source in
`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` retains the
fix: primary and non-primary scatter destinations are in separate control-flow
branches. It also retains a warning against merging those address spaces. Current
`test/registered/kernels/ops/attention/test_topk_v2.py` independently compares
selected indices with `torch.topk` and covers the 65,536/65,537 cluster boundary,
98,304 and 131,072 lengths, fused/persistent batch boundaries, and page-table
variants.

## Local validation

The assigned device is one AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not an
NVIDIA Hopper device. Five focused GPU cases passed against the independent
`torch.topk` reference: C4 lengths 65,536, 65,537, 98,304, and 131,072 at k=512,
plus the long-context ragged case. Raw output is in
`raw/gfx950_topk_boundaries.log`.

This is useful boundary coverage, but it does not execute the CUDA-only
`TopKCluster`/DSMEM code (`#ifndef USE_ROCM`) and therefore cannot reproduce or
runtime-qualify the SM90a compiler fix. DeepSeek-V4-Flash-0731 weights, eight H100
GPUs, TP=8, CUDA 13.1+, and compute-sanitizer were unavailable. No tiny-Llama
serving smoke was run because it cannot exercise the DeepSeek-V4 indexer or the
CUDA cluster kernel and would not qualify the reported bug.

## Conclusion

Outcome: `candidate_verified`. Source inspection and the merged fix's documented
H200 before/after evidence show that current main contains the relevant solution;
available gfx950 numerical boundary tests pass. Full issue-specific runtime
verification remains blocked by architecture, topology, and model availability.
