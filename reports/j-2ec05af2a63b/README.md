# MI300X decode-sized block and KV workspace study

## Scope

This report is the decode-sized, recurrent-state follow-up requested by amdpilot-org/sglang issue 390. It is distinct from the already-prefill-sized study on branch `amdpilot/j-5a3c88eb5781`.

The reduced block uses:

1. `sglang.kernels.ops.layernorm.fused_add_rmsnorm`
2. `sglang.kernels.ops.kvcache.reshape_and_cache_flash`
3. `sglang.kernels.ops.activation.silu_and_mul`
4. `torch.nn.functional.linear` as the supported bf16 projection control

The public SGLang projection wrappers were probed first. On this HIP stack they are unavailable:

- `dsv3_fused_a_gemm`: `AttributeError: module 'sgl_kernel' has no attribute 'dsv3_fused_a_gemm'`
- `fp8_scaled_mm`: `NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'`

No projection result is claimed for those unsupported public wrappers.

## Candidate

The real cases were run against candidate commit `d806acd0bd8312b38112362453f285b32d3d3c7c7`, which already fixes the AITER RMSNorm argument contracts. This report does not duplicate or republish that fix.

The candidate's relevant correction is:

```text
rmsnorm2d_fwd_with_add(out, input, residual, residual_out, weight, eps)
```

Without that candidate, `main` swaps `residual_in` and `residual_out`, causing both in-place outputs to be wrong on gfx942.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Candidate source: `/tmp/sglang-candidate-d806/python/sglang/__init__.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`

## Early installed-source baseline

`baseline-first.json` records the first bounded GPU execution before cloning or editing the delivery checkout. It is explicitly not proof for later checkout changes.

- Installed SGLang source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- First GPU execution elapsed time: `2.241546483710408` seconds
- Case: 128 tokens, hidden 4096, intermediate 8192, bf16
- Independent fp32 RMSNorm plus Torch SiLU composition
- Gate: `allclose(atol=0.02, rtol=0.02)`
- Result: pass, max absolute error `0.03125`
- Timing: CUDA events, 3 warmups, 10 timed iterations

## Workload

The block uses:

- Hidden size: 2048
- Attention heads: 8
- Head size: 128
- Intermediate size: 4096
- Dtype: bf16
- KV block size: 16
- Synthetic weights: 62,918,656 bytes, below 4 GiB

The six supported cases are:

```text
(batch, context) = (1,64), (1,256), (4,64), (4,256), (16,64), (16,256)
```

Each case performs:

- 3 checked recurrent warmup forwards
- 12 checked and timed recurrent forwards
- 3 warmups and 12 timed calls per operator
- CUDA/HIP event timing
- Independent Torch references for norm, projection, activation, cache, and whole chain

## Numerical and contract gates

The unchanged gates are:

- Norm/residual: `allclose(rtol=0.02, atol=0.02)`
- Projection control: `allclose(rtol=0.01, atol=0.01)`
- Activation: `allclose(rtol=0.02, atol=0.02)`
- Cache: exact
- Whole chain: normalized RMSE `<= 0.02`

All six cases pass. Whole-chain normalized RMSE ranges from `0.00855074729770422` to `0.010337656363844872`.

All steps also preserve:

- bf16 dtype
- input and residual storage pointers
- in-place norm mutation contract
- exact cache writes

## Warm results

Median and p95 whole-chain times are in milliseconds.

| Batch | Context | Chain median | Chain p95 | Norm | QKV | KV write | Gate | Activation | Down |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.6385 | 0.6742 | 0.1241 | 0.0475 | 0.0822 | 0.0464 | 0.0300 | 0.0457 |
| 1 | 256 | 0.6185 | 0.6568 | 0.1211 | 0.0464 | 0.0812 | 0.0469 | 0.0291 | 0.0464 |
| 4 | 64 | 0.6317 | 0.6428 | 0.1235 | 0.0484 | 0.0834 | 0.0467 | 0.0302 | 0.0463 |
| 4 | 256 | 0.6270 | 0.9757 | 0.1205 | 0.0461 | 0.0825 | 0.0449 | 0.0289 | 0.0447 |
| 16 | 64 | 0.6281 | 0.8589 | 0.1178 | 0.0456 | 0.0828 | 0.0463 | 0.0281 | 0.0447 |
| 16 | 256 | 0.6413 | 0.6728 | 0.1224 | 0.0460 | 0.0822 | 0.0461 | 0.0307 | 0.0457 |

## Memory

Predictions use logical tensor bytes plus PyTorch's 512-byte allocator granularity for the slot tensor. Measured values are live-allocation deltas after one-time backend warmup.

| Batch | Context | Predicted setup | Measured setup | Predicted peak | Measured peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 270,848 | 270,848 | 313,856 | 314,368 |
| 1 | 256 | 1,057,280 | 1,057,280 | 1,100,288 | 1,100,800 |
| 4 | 64 | 1,081,856 | 1,081,856 | 1,253,888 | 1,254,400 |
| 4 | 256 | 4,227,584 | 4,227,584 | 4,399,616 | 4,400,128 |
| 16 | 64 | 4,325,888 | 4,325,888 | 5,014,016 | 5,014,528 |
| 16 | 256 | 16,908,800 | 16,908,800 | 17,596,928 | 17,597,440 |

The maximum measured live allocation is 160,207,872 bytes, far below the 48 GiB limit. The 512-byte differences are allocator granularity for the slot tensor.

## Reproduction

From a checkout containing this report:

```bash
git worktree add --detach /tmp/sglang-candidate-d806 d806acd0b
cd /tmp/sglang-candidate-d806
PYTHONPATH="$PWD/python" /opt/venv/bin/python \
  /path/to/this/checkouts/reports/j-2ec05af2a63b/benchmark_mi300_block.py \
  --output /path/to/this/checkouts/reports/j-2ec05af2a63b/mi300-block-results.json
```

The command records candidate commit `d806acd0bd8312b38112362453f285b32d3d3c7c7` and the actual imported source paths.

## Limits and boundaries

- No full model weights or checkpoint were downloaded.
- No node-wide state was modified.
- No upstream issue, PR, or comment was posted or changed.
- No unbounded loop, sleep loop, or synthetic GPU burn was used.
- Total benchmark wall time was 12.039587509818375 seconds.

## Left undone

- The unsupported public SGLang projection wrappers were not replaced or fabricated.
- The already-existing AITER norm fix was not duplicated in this PR.
- The benchmark is report-only on `main`; it requires candidate `d806acd0b` for the corrected public AITER norm path.
