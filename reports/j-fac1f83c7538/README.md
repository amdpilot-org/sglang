# MI300X backend comparison for the reduced SGLang block

## Scope

This is the backend-comparison follow-up to amdpilot-org/sglang issue 398. It
does not repeat that issue's eager-versus-captured replay study. The read-only
upstream context was sgl-project/sglang issue 29630; its description and
comments were inspected without posting or changing anything upstream.

The reduced block is:

1. `sglang.kernels.ops.layernorm.rmsnorm`
2. `sglang.kernels.ops.attention.dsv4.linear_bf16_fp32`
3. `sglang.kernels.ops.activation.silu_and_mul`
4. `sglang.kernels.ops.attention.dsv4.linear_bf16_fp32`

The two compared execution paths are the installed AITER ROCm path and the
Torch/ROCm path. No fallback is relabeled as AITER.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, 304 AITER compute units.
- Qualified local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Delivery checkout: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AITER source: `/sgl-workspace/aiter/aiter`.
- AITER native modules loaded during the run include `module_rmsnorm_quant.so`,
  `module_activation.so`, and `module_custom.so`.

The installed-source baseline was recorded before delivery-clone changes in
`/job/baseline-first.json`. Its source commit was
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. The first selector returned
pytest code 5 because it selected no tests; the meaningful supported control,
`test_rmsnorm[True-dtype0-1-256]`, passed in 16.767598 seconds. That baseline is
not evidence for later checkout changes.

## Code fix

The checkout's AITER adapters used stale AITER signatures:

- `RMSNormOp.forward_aiter` called `rmsnorm2d_fwd(out, input, weight, eps)`.
  The installed AITER API is `rmsnorm2d_fwd(input, weight, epsilon)` and
  returns a tensor.
- `FusedAddRMSNormOp.forward_aiter` passed `residual_out` and `residual` in the
  wrong positions. The installed API is
  `rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)`.

Both adapters now use the installed signatures while preserving SGLang's
public `out=` and in-place contracts. The existing real-GPU parity suite and
a new RMSNorm `out=` contract test pass.

## Benchmark

The harness is `benchmark_backend_block.py`. It uses locally generated bf16
data and weights, three workload cases, five warmup iterations, and 20 timed
iterations per measurement. Weights are about 147 KiB per case, well under the
4 GiB limit. Timing uses CUDA events around complete operator or block calls.
There are no unbounded loops, sleep loops, checkpoint downloads, or synthetic
GPU burn.

The cases are `(tokens, hidden, intermediate)`:

- `(4, 256, 128)`
- `(8, 256, 128)`
- `(16, 256, 128)`

These shapes are deliberately small enough for AITER's native `skinny` GEMM
default. An earlier exploratory shape was rejected because AITER logged
`using torch solution:0`; that result is not counted as an AITER projection.

### Median timings

| Backend | Tokens | RMSNorm | Projection 1 | SiLU-and-mul | Projection 2 | Block | Operator sum | Block minus sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AITER | 4 | 0.042057 | 0.054225 | 0.034800 | 0.053463 | 0.185347 | 0.184546 | 0.000801 |
| AITER | 8 | 0.041476 | 0.054525 | 0.034559 | 0.053824 | 0.190619 | 0.184384 | 0.006235 |
| AITER | 16 | 0.042117 | 0.053604 | 0.034720 | 0.053664 | 0.187592 | 0.184104 | 0.003488 |
| Torch | 4 | 0.070784 | 0.031332 | 0.028846 | 0.030511 | 0.167686 | 0.161473 | 0.006213 |
| Torch | 8 | 0.070222 | 0.030610 | 0.028025 | 0.030249 | 0.170553 | 0.159107 | 0.011446 |
| Torch | 16 | 0.069821 | 0.032094 | 0.029067 | 0.034179 | 0.185869 | 0.165161 | 0.020708 |

All times are milliseconds. The complete-block time is close to the sum of
individually timed operators; the remaining difference is small and includes
allocation and launch effects. These tiny blocks are launch-overhead sensitive
and are not model-level throughput claims.

### Dispatch and correctness

- AITER fused-op trace: `layernorm.rmsnorm -> aiter` and
  `activation.silu_and_mul -> aiter`.
- AITER projection dispatch: native `skinny`, solution index 2, for both
  projections in every case.
- Torch fused-op trace: `layernorm.rmsnorm -> torch` and
  `activation.silu_and_mul -> torch`.
- Torch projection dispatch: `torch.mm` through the ROCm/cuBLAS-backed
  `aten::mm` CUDA registration.
- Both projections' public raw output is fp32; the chain casts that output to
  bf16 before activation and the final block output is bf16.
- RMSNorm and activation `out=` return the provided tensor, leave inputs
  unchanged, and preserve dtype. Projection weights are unchanged and a new
  output tensor is returned.
- The independent Torch composition passes
  `torch.allclose(atol=0.02, rtol=0.02)` for all six backend/case results.
  Raw max absolute differences and complete sample lists are in
  `aiter-results.json` and `torch-results.json`.

## Reproduction

From the repository root:

```bash
PYTHONPATH=python /opt/venv/bin/python \
  reports/j-fac1f83c7538/benchmark_backend_block.py \
  --backend aiter --output /tmp/aiter-results.json

PYTHONPATH=python /opt/venv/bin/python \
  reports/j-fac1f83c7538/benchmark_backend_block.py \
  --backend torch --output /tmp/torch-results.json

PYTHONPATH=python /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py
```

## Boundaries and unfinished work

- The installed AITER tuning table did not provide a native config for the
  initially explored larger block shapes; AITER selected Torch there. Those
  shapes were excluded rather than mislabeled.
- The benchmark covers only skinny projection shapes that are genuinely native
  in this installed AITER build. It does not claim performance for production
  hidden sizes or tuned large-GEMM shapes.
- The block is intentionally reduced and tiny, so timing is useful for
  per-operator attribution and dispatch comparison, not end-to-end model
  throughput.
- No CUDA/HIP graph capture is attempted here; that was the distinct scope of
  issue 398.
