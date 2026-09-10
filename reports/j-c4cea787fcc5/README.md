# Bounded MI300X INT8 dense-linear backend comparison

## Scope

This is a distinct INT8 follow-up to mirror issue 399. It does not repeat the completed FP8 AITER-versus-Triton study in mirror PR 462 or the eager-versus-graph-replay scope of issue 399. No kernel or dispatch code is changed.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, Torch capability `(9, 4)`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP: `7.2.26015-fc0010cf6a`
- SGLang checkout: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Installed AITER: `c16d44b93a528b2a4bfd6d8d3409116d465872a9`

## Installed-source baseline

Before cloning or editing, the installed AITER FP8 block-scale test ran successfully:

```bash
timeout 120s /opt/venv/bin/python \
  /sgl-workspace/aiter/op_tests/test_gemm_a8w8_blockscale.py \
  -d bf16 -m 1 -nk 512,1024 --ck_preshuffle False
```

The independent dequantized reference used `atol=0.01`, `rtol=0.01`. CK and ASM both passed, as did CK and CKTile split-K checks. The first GPU execution completed in 13.984 seconds. Native paths included:

- `/sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_blockscale.so`
- `/sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_blockscale_cktile.so`
- `/sgl-workspace/aiter/hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_128x128.co`

This baseline is installed-source environment context only and is not evidence for later checkout changes. The complete record is in `/job/baseline-first.json`.

## Method

- Workloads: `M=1,64,1024,4096`, `N=2048`, `K=2048`
- Inputs: locally generated synthetic `torch.int8` tensors
- Scales: locally generated `torch.float32` per-token and per-output-column scales
- Output: `torch.float16`
- Backends:
  - direct AITER CK dispatch through `aiter.gemm_a8w8_CK`
  - direct SGLang Triton dispatch through `triton_scaled_mm`
- Reference: independent FP32 dequantized matmul of the identical operands and scales
- Numerical gates, unchanged from the installed tests:
  - AITER INT8: `rtol=0.01`, `atol=0.01`
  - SGLang INT8: `rtol=0.15`, `atol=0.10`
- Timing: three warmups followed by 30 CUDA-event measurements per backend and case
- Complete-block timing includes each wrapper's output allocation; the Triton weight transpose is prepared once outside the timed block
- Profiling: one real call per backend and case with Torch's CUDA profiler

The study uses four workload cases, well below the six-case limit. Each weight is 4,194,304 bytes, well below 4 GB. Peak allocated memory is 381,706,240 bytes. The complete run took 4.131 seconds.

## Results

All eight backend/case comparisons passed their unchanged numerical gates. Outputs were finite and used independent output addresses.

| M | AITER median ms | Triton median ms | Triton/AITER | AITER p95 ms | Triton p95 ms | Max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.036384 | 0.052261 | 1.436 | 0.058375 | 0.056450 | 0.007736 |
| 64 | 0.047710 | 0.051940 | 1.089 | 0.065993 | 0.054125 | 0.014099 |
| 1024 | 0.074452 | 0.067355 | 0.905 | 0.083392 | 0.069882 | 0.015572 |
| 4096 | 0.090850 | 0.105343 | 1.160 | 0.098988 | 0.108210 | 0.015644 |

The maxima above are the larger of the two backend errors for each case; both backends produced the same maxima. AITER was faster at `M=1`, `64`, and `4096`; Triton was faster at `M=1024`. These are bounded single-GPU observations, not a general performance claim.

## Native dispatch

- AITER Python: `/sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_gemm_a8w8.so`
- AITER profiled kernel family: `ck::kernel_gemm_xdl_cshuffle_v3_multi_d`
- SGLang Triton source: `/job/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py`
- SGLang profiled kernel: `scaled_mm_kernel`
- Triton cache: `/tmp/sglang-cache-j-c4cea787fcc5/triton`

The job-private Triton cache contains three `scaled_mm_kernel.hsaco` variants for the heuristic tile configurations. The raw JSON records every path.

## Reproduce

```bash
cd /job/sglang
PYTHONPATH=$PWD/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-c4cea787fcc5/triton \
/opt/venv/bin/python reports/j-c4cea787fcc5/run_int8_backend_study.py \
  reports/j-c4cea787fcc5/results.json
```

## Limitations

- One MI300X, one `N=K=2048` shape family, and four M values
- Synthetic operands only; no checkpoint or full-model weights
- Timing evidence only; no pass/fail performance gate
- No multi-GPU, MoE, CUDA-graph, or production-model claim
- No upstream issue, PR, or comment was posted or modified
