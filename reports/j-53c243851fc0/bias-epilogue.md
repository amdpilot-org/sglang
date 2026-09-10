# gfx942 FP8 scaled GEMM bias epilogue evidence

## Scope

This records the bias-epilogue contract for `gemm.fp8_scaled_mm` on one AMD Instinct
MI300X (`gfx942`). It does not redesign the kernel namespace discussed in
read-only upstream issue 29630; that RFC reports its namespace migration as
complete. No upstream issue, pull request, or comment was modified.

## Environment

- Delivery base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- GPU: one AMD Instinct MI300X, CUDA capability `(9, 4)`, gfx942.

## Paths

- Public wrapper: `python/sglang/kernels/ops/gemm/__init__.py`.
- Explicit native backend: `Fp8ScaledMMOp.forward_native`, resolved through
  `get_kernel("gemm.fp8_scaled_mm", KernelBackend.TORCH)`.
- Native Torch operation: `aten::_scaled_mm`, backed by Torch/HIP and hipBLASLt.
- Independent reference: float32 `torch.mm`, per-row/per-column scaling, and bias
  addition in `test/registered/kernels/ops/gemm/test_fp8_scaled_mm_bias.py`.
- Installed AOT native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- ROCm AOT registration source: `python/sglang/kernels/aot/csrc/common_extension_rocm.cc`.

## Installed-source baseline

The installed source was revision `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. Its
installed AOT wheel loaded but did not register `torch.ops.sgl_kernel.fp8_scaled_mm`,
so the public AOT call failed with `AttributeError`. The supported neighboring
control was `torch._scaled_mm`.

The control used ROCm `float8_e4m3fnuz` inputs and BF16 output. Zero and finite bias
matched the independent float32 reference exactly for row counts 1, 3, and 17.
FP16 output was unsupported with:
`hipblaslt rowwise _scaled_mm only supports BFloat16 output but got Half`.

Timing method was three warmup calls followed by ten synchronized wall-clock calls
for shape `(17, 128, 128)`. Mean was `0.06888331845402718 ms`, minimum
`0.05536619573831558 ms`, and maximum `0.08356664329767227 ms`. First GPU
execution elapsed `10.365161058492959` seconds from process start. The complete
installed-source baseline is labeled as such in `/job/baseline-first.json`; it is
not proof for checkout changes.

## Checkout validation

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/gemm/test_fp8_scaled_mm_bias.py -v
```

Result: 9 passed in 11.55 seconds.

The six supported cases compare public dispatch, explicit native dispatch, and the
independent float32 reference for zero/finite bias and row counts 1, 3, and 17.
All output shapes were `(rows, 128)` and all outputs were BF16. The unchanged
numerical gate is `rtol=2e-2, atol=2e-2`; observed maximum absolute error was
`0.0` for every public/native, public/reference, and native/reference comparison.

The remaining three cases verify rejection of a 127-element bias, a FP16 bias with
BF16 output, and FP16 output on HIP. The ROCm AOT source does not register
`fp8_scaled_mm`, so the supported gfx942 public path is its native Torch fallback.
