# gfx942 AWQ/GPTQ scheme-kernel evidence

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, UUID `34333965-3031-6163-3332-323164383838`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Mirror source: `/job/sglang`, commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AWQ Triton source: `/job/sglang/python/sglang/kernels/ops/quantization/awq_triton.py`.
- Native artifact used by the installed-source baseline: `/job/.cache/sglang/triton/IVPZWFXAMUPZSNKQKGQJQAGEX7OQSSB2ICQRBCHNPX4JTFX2LBNQ/awq_gemm_kernel.hsaco`.

## Installed-source baseline

The first successful GPU case used the preinstalled source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python - <<'PY'
# AWQ Triton GEMM, shape 8x128 @ 128x32, group_size=128, fp16.
PY
```

- Reference: independent Torch int32 unpack with AWQ reverse order `[0,4,1,5,2,6,3,7]`, zero/scale expansion, dequantization, and `torch.matmul`.
- Gate: `torch.testing.assert_close(rtol=4e-2, atol=4e-2)`.
- Result: PASS; max absolute error `0.125`, max relative error `0.01869918778538699`.
- Timing: three warmups, one synchronized `time.perf_counter` interval; kernel `0.00008606 s`; first successful process wall time `7.384959 s`.

## Mirror boundary test

Added `TestAWQTriton::test_gemm_known_boundaries` in `test/registered/quant/test_awq_dequant.py`. It uses one small packed linear weight set with:

- `K=128`, `N=32`, `M=8`.
- Weight codes alternating the 4-bit boundaries `0` and `15`.
- Zero-point boundaries `0`, `8`, and `15`.
- Group sizes `32`, `64`, and `128`.
- Per-group scales spanning `0.01` to `0.05`.

The test compares:

1. AWQ dequantization plus `torch.matmul` (shared kernel path).
2. Fused `awq_gemm_triton` (format-specific kernel path).
3. An independent Torch dequantization reference.

Gates:

- Dequantized weights and shared-path output: `rtol=1e-6, atol=1e-6`.
- Fused GEMM: `rtol=2e-2, atol=2e-2`.

Command:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/quant/test_awq_dequant.py::TestAWQTriton::test_gemm_known_boundaries \
  -p no:cacheprovider
```

Result: `1 passed, 9 subtests passed in 14.56s`.

Raw prototype max absolute errors (fused GEMM minus Torch reference; shared path was exactly `0.0` in every case):

| group size | zero 0 | zero 8 | zero 15 |
|---:|---:|---:|---:|
| 32 | 0.009372711 | 0.005053520 | 0.009067535 |
| 64 | 0.009866714 | 0.005149841 | 0.007179260 |
| 128 | 0.010654449 | 0.006002426 | 0.011240959 |

## Unsupported paths on gfx942

- Preinstalled GPTQ AOT linear: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'gptq_gemm'`.
- Preinstalled AWQ JIT dequantization: `hipcc` fails at `awq_dequantize.cuh:154` with `unknown type name '__nv_bfloat16'`.
- Preinstalled AWQ AOT dequantization: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'awq_dequantize'`.
- Mirror AWQ Marlin repack (`test_awq_marlin_repack_correct[16-1-1-4]`): `hipcc` fails on `__cvta_generic_to_shared`, `cudaFuncSetAttribute`, and `cudaFuncAttributeMaxDynamicSharedMemorySize`.
- Mirror GPTQ Marlin GEMM (`test_gptq_marlin_gemm[False-mnk_factors0--1-quant_type0-64-128]`): `hipcc` fails on `__cvta_generic_to_shared`, invalid asm constraint `l`, and `nv_bfloat162`.

No global refactor was made. The change is limited to the focused AWQ boundary test.
