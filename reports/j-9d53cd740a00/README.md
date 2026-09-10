# Bounded INT8/FP8 dense-linear prefill study

## Scope

This is a prefill-sized throughput reduced-block study for the installed
INT8/FP8 dense-linear path on one assigned MI300X (`gfx942`). It is distinct
from earlier standalone operator boundary probes.

- Upstream issue 15194, `[Roadmap] Quantization Modifications`, is open.
- Relevant merged upstream PR 17449 adds MXFP8 support and the Triton dense
  linear path used here.
- Open upstream PRs 26846 and 26942 are package-layout and GGUF refactors,
  not dense-linear throughput candidates.
- Upstream context: read-only `sgl-project/sglang` issue 15194.
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Branch: `amdpilot/j-9d53cd740a00`.
- GPU: one AMD Instinct MI300X, architecture `gfx942`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.

## Environment paths

- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`
- Installed SGLang source: `/sgl-workspace/sglang`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- Installed AOT native module:
  `/sgl-workspace/sglang/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-310/sgl_kernel`

## Installed-source baseline

The first GPU execution objective was completed before any mirror checkout
changes. The installed numerical test was:

```bash
cd /sgl-workspace/sglang
SGLANG_USE_AITER=0 /opt/venv/bin/python -m pytest -q \
  test/registered/quant/test_triton_scaled_mm.py -s
```

Result: `1 passed, 5 subtests passed in 39.03s`.

The direct `torch._scaled_mm` path was unsupported on this ROCm build and
returned:

```text
RuntimeError: CUDA error: HIPBLAS_STATUS_NOT_SUPPORTED when calling `HIPBLAS_STATUS_NOT_SUPPORTED`
```

The supported neighboring control used
`sglang.kernels.ops.quantization.fp8_kernel.triton_scaled_mm` with an INT8
case of `M=128`, `K=128`, `N=1024`. It passed the unchanged gate
`rtol=0.15`, `atol=0.10`, with cold first-forward time `0.835 s` and warm
throughput `0.858 TFLOPS` over 100 measured real forwards.

## Benchmark command

```bash
cd /job/sglang
/opt/venv/bin/python benchmark/bench_triton_scaled_mm_prefill.py \
  --output reports/j-9d53cd740a00/results.json
```

The benchmark uses:

- Kernel: `sglang.kernels.ops.quantization.fp8_kernel.triton_scaled_mm`
- Input/weight dtypes: `int8` and `float8_e4m3fn`
- Output dtype: `bfloat16`
- Scale dtype: `float32`
- Reference: independent FP32 dequantized matmul
- Numerical gates, unchanged from the installed test:
  - INT8: `rtol=0.15`, `atol=0.10`
  - FP8: `rtol=0.25`, `atol=0.15`
- Cold timing: wall clock around the first real forward for each dtype
- Warm timing: CUDA events after 10 warmup real forwards
- Measured real forwards: 100 for `M < 4096`, 30 for `M >= 4096`
- Maximum cases: 6
- Weight limit: 4 GB
- Total live allocation limit: 48 GB
- Wall limit: 7200 seconds

## Results

- INT8, `M=128`, `K=2048`, `N=2048`: cold `0.811 s`, warm `0.00384 s`,
  `27.99 TFLOPS`, max abs error `0.00195`, mean relative error `0.00143`,
  gate passed.
- INT8, `M=1024`, `K=2048`, `N=2048`: cold `0.477 s`, warm `0.00435 s`,
  `197.58 TFLOPS`, max abs error `0.00195`, mean relative error `0.00143`,
  gate passed.
- INT8, `M=8192`, `K=2048`, `N=2048`: cold `0.000761 s`, warm `0.00375 s`,
  `550.06 TFLOPS`, max abs error `0.00370`, mean relative error `0.00143`,
  gate passed.
- FP8, `M=128`, `K=2048`, `N=2048`: cold `2.842 s`, warm `0.02699 s`,
  `3.98 TFLOPS`, max abs error `0.000122`, mean relative error `0.00141`,
  gate passed.
- FP8, `M=1024`, `K=2048`, `N=2048`: cold `2.466 s`, warm `0.03006 s`,
  `28.57 TFLOPS`, max abs error `0.000122`, mean relative error `0.00141`,
  gate passed.
- FP8, `M=8192`, `K=2048`, `N=2048`: cold `0.00191 s`, warm `0.03637 s`,
  `56.69 TFLOPS`, max abs error `0.000122`, mean relative error `0.00141`,
  gate passed.

Peak allocated memory across all cases was `671,129,600` bytes, well below the
48 GB total live allocation limit. Each generated weight was `4,194,304` bytes,
well below the 4 GB weight limit. The complete benchmark run completed in
`15.07` seconds of wall time.

## Notes

- No upstream issue, PR, or comment was posted or modified.
- No full model weights were downloaded.
- No node-wide state was modified.
- No artificial burn, unbounded loop, or sleep loop was used.
- The installed-source baseline is environment context only and is not proof
  for later checkout changes.
