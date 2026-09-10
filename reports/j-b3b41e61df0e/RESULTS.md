# gfx942 INT8 linear investigation

## Scope

This change covers shared symmetric W8A8 INT8 dense-linear inference only. It does
not change AWQ/GPTQ packed weights, MTP selection, MoE quantization, or the global
quantization scheme layout.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, UUID
  `34333965-3031-6163-3332-323164383438`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Persistent checkout: `/job/sglang`, base `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Persistent Python source: `/job/sglang/python/sglang`

## Installed-source baseline

The preinstalled editable source was `/sgl-workspace/sglang` at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, imported from
`/sgl-workspace/sglang/python/sglang/__init__.py`.

Command:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python -m pytest -q \
  test/registered/unit/layers/quantization/test_int8_linear_methods.py::TestW8A8Int8Linear::test_channel -s
```

Result: all three channel subtests failed at the kernel boundary with
`NameError: name 'int8_scaled_mm' is not defined`. The first GPU execution took
30.72 seconds. The installed ROCm `sgl_kernel` native module is
`/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`,
and it does not register `torch.ops.sgl_kernel.int8_scaled_mm`.

The neighboring supported control, `TestBlockInt8Linear::test_block`, passed all
three subtests (including odd `M=5`) in 17.47 seconds. It dispatched
`_per_token_group_quant_int8` and `_w8a8_block_int8_matmul`, with a Triton
`float32` accumulator.

## Persistent-checkout result

Command:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/layers/quantization/test_int8_linear_methods.py -s
```

Result: `5 passed, 6 subtests passed` in 15.35 seconds. The unchanged numerical
gate remains `rtol=5e-2, atol=1e-1` against an independent dequantized `float32`
matmul.

For the profiled odd-row case `(M=5, N=160, K=336)`:

- `W8A8Int8LinearMethod` maximum absolute error: `0.004586130380630493`
- `CompressedTensorsW8A8Int8` maximum absolute error: `0.004586130380630493`
- Mean absolute error for both: `0.0010702040744945407`
- The two scheme outputs are exactly equal.
- Actual dispatch: `_per_token_quant_int8` (3.968 us) and `scaled_mm_kernel`
  (5.171 us).
- Kernel source: `/job/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py`
  and `/job/sglang/python/sglang/kernels/ops/quantization/int8_kernel.py`
- `scaled_mm_kernel` uses a Triton `int32` accumulator for INT8 inputs and applies
  the `float32` per-token and per-output-channel scales after accumulation.
- Weights remain unpacked INT8 `[N, K]` checkpoints, transposed to `[K, N]` at the
  kernel boundary.

Both supported symmetric per-channel representations use dynamic per-token
activation quantization and a `float32` weight scale of shape `[N, 1]`. A zero
weight scale produces an all-zero output in both paths.

## Zero-point boundary

The tested compressed-tensors case is `input_symmetric=True`; it has no activation
zero point. Asymmetric activation zero-point support remains unchanged and is not
claimed by this change. In particular, the existing TODO for an AZP-aware scaled MM
remains, so zero-offset behavior outside the documented symmetric representation
is not tested or inferred.
