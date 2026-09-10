# FP8 online/prepared linear GPU control

## Scope

- Upstream context: sgl-project/sglang issue 15194.
- Candidate already merged upstream: sgl-project/sglang PR 26415, commit `92247054e61ba4992a301bd87d0382a4a2fe444a`.
- The current implementation was not changed. This adds a real-GPU regression test for the already-working online and prepared FP8 linear paths.
- Tested mirror `main` commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, serial `692440003992`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Checkout source: `/job/sglang/python/sglang`
- AOT kernel package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Job-private JIT cache: `/tmp/sglang-cache-j-7c1bf47bb6c5`

## Installed-source baseline

The first GPU execution used the installed source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`:

```bash
/opt/venv/bin/python -m pytest \
  /sgl-workspace/sglang/test/registered/kernels/ops/quantization/test_per_tensor_quant_fp8.py::test_jit_per_tensor_quant_compare_implementations \
  -q --no-header -p no:cacheprovider
```

The installed JIT path is unsupported on this stack. Its kernel accepts `uint8`, while the test allocates `float8_e4m3fn`:

```text
RuntimeError: Tensor match failed for Tensor dtype=float8_e4m3fn
at /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/gemm/per_tensor_quant_fp8.cuh:118
- Root cause: Dtype value [float8_e4m3fn] not in the allowed options: [uint8]
```

The supported neighboring control used `input_to_float8` and `torch._scaled_mm` with `torch.float8_e4m3fnuz` (`fp8_max=224`). For shape `(M=128, N=128, K=256)`, online and prepared codes were bit-equal, outputs were bit-equal, and the independent float32 dequantized matmul had maximum absolute error `3.0517578125e-05` and maximum relative error `0.0066225165501236916`.

## Persistent checkout result

Command:

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-7c1bf47bb6c5 \
/opt/venv/bin/python -m pytest \
  test/registered/unit/layers/quantization/test_fp8_online_prepared_linear.py \
  -q --no-header -p no:cacheprovider
``+

Result: `1 passed` in `15.95s`; complete process wall time was `19.205467128s`.

The real `Fp8LinearMethod` control used a known `(128, 256)` BF16 weight and `(128, 256)` BF16 activation. The online path quantized the weight to per-tensor `float8_e4m3fnuz`. The prepared checkpoint used the same codes; on FNUZ it stored e4m3fn codes with half the runtime scale, which loading normalized back to the identical FNUZ representation and scale.

Raw metrics:

- Online weight scale: `0.0020054408814758062`
- Prepared weight scale: `0.0020054408814758062`
- Weight codes bit-equal: `true`
- Weight scales bit-equal: `true`
- Online/prepared linear outputs bit-equal: `true`
- Dynamic activation scale: `0.00188446044921875`
- Independent dequantized-matmul maximum absolute error: `0.000244140625`
- Independent dequantized-matmul maximum relative error: `0.007194244768470526`

## Numerical gates

The regression test keeps the gates explicit and unchanged:

- Prepared and online FP8 weight codes must be bit-equal.
- Prepared and online weight scales must be bit-equal.
- Prepared and online linear outputs must be bit-equal.
- Actual linear output must satisfy `torch.testing.assert_close(..., rtol=2e-2, atol=1e-2)` against an independent float32 dequantized activation-by-weight matmul.

No full model weights were downloaded, no node-wide state was modified, and no synthetic GPU work was used.
