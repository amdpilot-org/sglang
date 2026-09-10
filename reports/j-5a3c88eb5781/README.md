# MI300X reduced-block kernel study

## Scope

This study exercises the public SGLang operators from issue 29630 as a prefill-sized reduced block:

1. `sglang.kernels.ops.layernorm.rmsnorm`
2. per-token FP8 quantization plus `aiter.gemm_a8w8`
3. `sglang.kernels.ops.activation.silu_and_mul`

The public projection operator `sglang.kernels.ops.gemm.fp8_scaled_mm` is also probed. On this ROCm stack it fails with:

```text
RuntimeError: Only multiplication of row-major and column-major matrices is supported by cuBLASLt
```

The Aiter GEMM is therefore labeled as a supported neighboring projection control, not as proof that the public SGLang projection operator works on ROCm.

## Environment

- GPU: one AMD Instinct MI300X, gfx942 (`capability [9, 4]`, 304 CUs)
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd8dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Checkout: `amdpilot-org/sglang` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- SGLang source: `/job/sglang/python/sglang/__init__.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Aiter: `/sgl-workspace/aiter/aiter/__init__.py`
- Job-private cache: `/tmp/sglang-cache-j-5a3c88eb5781`

## Operators and dtypes

- Input and norm weight: `torch.bfloat16`
- Projection operands: `torch.float8_e4m3fnuz`
- Projection scales: `torch.float32`
- Projection and final output: `torch.bfloat16`
- Norm epsilon: `1e-6`
- Hidden size: `4096`
- Projection output before gated activation: `8192`
- Final activation output: `4096`

## Bounded cases

The benchmark uses exactly six token cases:

```text
2048, 4096, 8192, 16384, 32768, 65536
```

Each case uses:

- one synchronized cold chain call,
- three warmups,
- twenty timed calls for norm, projection, activation, and the whole chain,
- CUDA/HIP events for warm timing,
- one independent fp32/dequantized Torch reference.

The locally generated projection weight is 33,554,432 bytes, below the 4 GiB limit. Peak allocated memory across all cases is 11,455,471,616 bytes, below the 48 GiB live-allocation limit. The measured benchmark wall time was 5.93 seconds; the task limit was 7200 seconds.

## Numerical and contract gates

The unchanged accuracy gate is:

```text
normalized_rmse <= 0.02
```

All six cases pass. The normalized RMSE range is `0.01985774375498295` through `0.019901975989341736`.

All cases also pass:

- actual output dtype equals `torch.bfloat16`,
- input remains unchanged,
- explicit norm output matches the returned norm output,
- explicit activation output matches the returned activation output.

## Warm results

| Tokens | Chain ms | Norm ms | Projection ms | Activation ms | Tokens/s | Projection TFLOP/s |
---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.2976 | 0.0395 | 0.2848 | 0.0180 | 6,881,527 | 482.7 |
| 4096 | 0.5529 | 0.0326 | 0.5307 | 0.0363 | 7,408,626 | 517.9 |
| 8192 | 1.0807 | 0.0344 | 1.0108 | 0.0616 | 7,580,578 | 543.9 |
| 16384 | 2.0656 | 0.0585 | 1.9470 | 0.1197 | 7,931,889 | 564.7 |
| 32768 | 4.2564 | 0.1117 | 4.0575 | 0.2537 | 7,698,475 | 542.0 |
| 65536 | 9.0350 | 0.2901 | 8.5517 | 0.5411 | 7,253,587 | 514.3 |

Projection timing includes per-token quantization and the Aiter GEMM. Whole-chain timing includes all three stages. Raw values, cold timings, memory counters, paths, commands, and limits are in `results.json`.

## Reproduction

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  test/registered/kernels/benchmark/reduced_block/bench_norm_fp8_projection_activation.py \
  --output reports/j-5a3c88eb5781/results.json
```

The ROCm RMSNorm wrapper fix is validated by:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py::test_rmsnorm
```

This produced six passing tests.

## Early installed-source baseline

`baseline-first.json` records the first bounded GPU execution from the preinstalled source before cloning or editing the delivery checkout. It is explicitly not proof for later checkout changes.

Key facts:

- installed SGLang: `0.5.18.dev20260826+g937af8538b`
- installed source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- first GPU execution elapsed time: `3.612058241851628` seconds
- one cold synchronized call, three warmups, twenty timed iterations
- 512 tokens, hidden 4096, projection output 8192
- normalized RMSE gate: `<= 0.02`
- measured normalized RMSE: `0.018313298001885414`
- warm mean chain time: `0.18898550271987916 ms`
- peak allocated memory: `484517888` bytes

The installed default ROCm RMSNorm path failed because the wrapper passed an output tensor where Aiter expected the float epsilon argument. The baseline then used the public operator's supported Torch backend as the neighboring control. The installed public FP8 projection also failed because its native op was absent from the installed ROCm `sgl_kernel` wheel.

## Upstream context

Read-only context came from `sgl-project/sglang` issue 29630, including its current description, fourteen comments, and related changes. The issue records that the `sglang.kernels.ops` migration is complete and that `sglang.jit_kernel` was retired by merged PR 32072. No upstream issue, PR, or comment was posted or modified.

## Limitations

- This is a reduced prefill-sized block, not an end-to-end model benchmark.
- The projection control is Aiter rather than the unsupported public SGLang FP8 projection on this ROCm stack.
- The Aiter GEMM reports no tuned configuration for these shapes and uses its default config.
- No full model weights were downloaded.
