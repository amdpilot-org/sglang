# gfx942 FP8 reduced-block latency study

## Scope

This is a bounded, one-MI300X experiment for decode-sized latency using the
supported native PyTorch FP8 dense linear kernel. It is distinct from earlier
standalone operator boundary probes:

- Six token/batch cases: `M = 1, 2, 4, 8, 16, 32`.
- A real `LlamaMLP` block forward from
  `python/sglang/srt/models/torch_native_llama.py`.
- Four recurrent block steps per sequence.
- Three warmup sequences and ten timed sequences per case.
- Independent dequantized reference comparison after every step.
- CUDA-event median and p95 latency across all timed steps.
- Synthetic packed FP8 weights and float32 scales; no model weights are
  downloaded.

The upstream context is `sgl-project/sglang` issue `15194`. The merged
quantization-refactor PRs listed there are already represented in this mirror’s
`main`; the open AutoRound and GGUF PRs target unrelated layouts and were not
duplicated.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- Source import: `/job/sglang/python/sglang/__init__.py`.
- Native Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.

## First GPU baseline

The installed `sgl_kernel` wheel exposes Python wrappers for
`int8_scaled_mm` and `fp8_scaled_mm`, but the corresponding CUDA ops are not
registered in this ROCm build:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'int8_scaled_mm'
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fp8_scaled_mm'
```

The meaningful supported neighboring control is PyTorch’s native
`torch._scaled_mm` FP8 path. It uses `float8_e4m3fnuz` on gfx942 and passed an
independent dequantized reference comparison for `M=1` and `M=128` with
`K=512`, `N=128`. The first successful GPU control completed in about five
seconds. Raw details are in `baseline-first.json`.

## Reduced-block benchmark

Run:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  test/registered/kernels/benchmark/gemm/bench_gfx942_fp8_reduced_block.py \
  --output /job/gfx942-fp8-reduced-block.json
```

The benchmark uses:

- Input: `bfloat16`, dynamically quantized per row to `float8_e4m3fnuz`.
- Packed weight: `float8_e4m3fnuz`.
- Weight scale: `float32`, one scale per output channel.
- Output: `bfloat16`.
- Gate-up GEMM: `M x 1024 x 1024`.
- Down GEMM: `M x 512 x 1024`.
- Accuracy gate: finite output, `max_abs_error <= 1e-4`, and
  `max_relative_error <= 0.02` against an independent dequantized reference.
- Weight limit: under 4 GB per case.
- Live allocation limit: under 48 GB.
- Wall limit: 7200 seconds.

All six cases passed. Peak allocated memory was about 93 MB, and the complete
run finished in about 4.4 seconds. Median block latency stayed near 240–256 µs
across the tested `M` values; p95 ranged from about 286–328 µs. The full raw
measurements are in `results.json`.

## Limitations

- This study uses synthetic weights and a single real MLP block, not a full
  model or production checkpoint.
- The installed `sgl_kernel` CUDA scaled-mm ops are unavailable on this ROCm
  wheel, so the benchmark uses the native PyTorch FP8 path rather than an
  sglang custom op.
- The recurrent sequence is intentionally short and bounded; it is not an
  artificial burn or an unbounded loop.
