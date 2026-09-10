# MI300X chunk-partition reduced-block report

## Scope

This is the distinct chunk-partitioning follow-up to amdpilot-org/sglang issue 382. It uses the public `sglang.kernels` operators from read-only sgl-project/sglang issue 29630, one assigned gfx942 GPU, locally generated bf16 weights, and exactly six workload cases. No checkpoint or model weights were downloaded.

The tested mirror base is `0084030179bfba86bfeb6d43f7997d4076329d2c`. Upstream PR 33095 was inspected; it changes the production row-strided RMSNorm path and is unrelated to the two public-operator AIter call-order defects fixed here.

## Installed-source baseline

Before cloning or editing, the installed source at `/sgl-workspace/sglang` was tested with:

```bash
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /sgl-workspace/sglang/test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py::test_rmsnorm \
  /sgl-workspace/sglang/test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py::test_gated_activation
```

The first real GPU execution took 16 seconds wall time (13.15 seconds reported by pytest). All 12 gated-activation cases passed. All six RMSNorm cases failed before comparison because `forward_aiter` passed the output tensor where AIter expected `input`, causing the weight to occupy the `epsilon` argument. The full receipt is in `/job/baseline-first.json`; it is installed-source evidence only and is not proof for the mirror checkout.

## Operator fixes

Two AIter call contracts in `python/sglang/kernels/ops/layernorm/__init__.py` were corrected:

- `RMSNormOp.forward_aiter` now calls `rmsnorm2d_fwd(input, weight, eps)` and copies into the optional public `out` tensor.
- `FusedAddRMSNormOp.forward_aiter` now passes `residual` before `residual_out`, matching AIter's schema and preserving the public in-place mutation contract.

The existing GPU parity gates were unchanged. After the fix, all six `test_rmsnorm` cases and all four `test_fused_add_rmsnorm` cases passed.

## Reduced block

Each forward is:

1. public `sglang.kernels.ops.layernorm.fused_add_rmsnorm`, mutating input and residual;
2. bf16 up projection with Torch `F.linear`;
3. public `sglang.kernels.ops.activation.silu_and_mul`, returning a new tensor;
4. bf16 down projection with Torch `F.linear`.

The projection boundary is explicit. `sglang.kernels.ops.gemm.fp8_scaled_mm` has only CUDA/SM90 AOT and Torch backends in its priority list; neither is HIP-eligible on gfx942. No SGLang projection result is fabricated. `F.linear` is used only as the supported projection control.

Dimensions are hidden size 4096 and intermediate size 8192. Synthetic weights total 201,326,592 bytes (192 MiB), below the 4 GiB limit. The six legal cases are:

| Tokens | Partition rows |
|---:|---|
| 1024 | unchunked |
| 1024 | 256 / 512 / 256 |
| 4096 | unchunked |
| 4096 | 1024 / 2048 / 1024 |
| 16384 | unchunked |
| 16384 | 4096 / 8192 / 4096 |

## Numerical and mutation contracts

The unchanged bf16 gate is `atol=2e-2`, `rtol=2e-2`. It is applied to:

- public fused-add RMSNorm final state versus an independent Torch composition;
- public SiLU-and-mul versus `F.silu(gate) * up` on identical input;
- chunked output and final state versus unchunked output and final state.

All compared tensors remain bf16. `fused_add_rmsnorm` mutates input and residual in place; `silu_and_mul` returns a new tensor and leaves its input unchanged. Whole-chain error against the independent Torch composition is recorded as raw max/mean absolute and relative error, but is not gated because accumulated bf16 projection and activation differences are not a per-operator gate. No numerical gate was changed.

## Results

Timing uses CUDA/HIP events after three warmups and ten finite measurements. The full run took 16 seconds. The first case includes 0.79 seconds of cold AIter import/JIT wall time; later cold cases are approximately 0.007–0.011 seconds.

| Case | Chain ms | Tokens/s | TFLOP/s | Peak allocated | Norm / up / act / down share |
|---|---:|---:|---:|---:|---|
| 1024 unchunked | 0.527 | 1,943,376 | 391.25 | 0.535 GiB | 12.4% / 53.2% / 4.8% / 29.5% |
| 1024 chunked | 0.655 | 1,562,791 | 314.63 | 0.541 GiB | 23.6% / 42.9% / 5.2% / 28.3% |
| 4096 unchunked | 1.642 | 2,495,148 | 502.34 | 1.354 GiB | 8.5% / 58.5% / 4.1% / 28.9% |
| 4096 chunked | 1.877 | 2,182,213 | 439.34 | 1.378 GiB | 12.9% / 54.1% / 4.6% / 28.4% |
| 16384 unchunked | 6.023 | 2,720,353 | 547.68 | 4.635 GiB | 5.0% / 59.4% / 4.4% / 31.2% |
| 16384 chunked | 6.701 | 2,444,845 | 492.21 | 4.729 GiB | 5.1% / 59.1% / 4.4% / 31.4% |

Chunking reduced measured throughput by approximately 19.6% at 1024 tokens, 12.6% at 4096 tokens, and 10.2% at 16384 tokens. Projection dominates the whole-chain attribution at larger token counts. The norm share is sensitive to small-token chunking because each chunk adds a kernel launch. Peak allocated memory stayed far below the 48 GiB live-allocation limit.

Raw timings, backend boundary, source/native paths, GPU identity, image identity, commands, and per-case numerical observations are in `reports/j-f81a8affcf55/chunk-partition-results.json`.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /job/sglang/test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py::test_rmsnorm \
  /job/sglang/test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py::test_fused_add_rmsnorm

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  /job/sglang/test/registered/kernels/benchmark/layernorm/bench_chunk_partition.py \
  --output /job/sglang/reports/j-f81a8affcf55/chunk-partition-results.json
```

## Environment

- GPU: one AMD Instinct MI300X, gfx942, 206,141,652,992 bytes VRAM.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, `2.9.1+rocm7.2.0.git7e1940d4`.
- SGLang source: `/job/sglang/python/sglang/__init__.py`.
- `sgl_kernel` Python/native: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py` and `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- AIter RMSNorm native: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

No upstream issue, PR, or comment was posted or modified.
