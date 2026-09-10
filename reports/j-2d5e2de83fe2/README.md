# gfx942 compiled fused-MoE reuse investigation

This report extends the prior MoE dispatch investigation with the uncovered
`torch.compile(dynamic=False)` fused-MoE execution representation. It does not
repeat the original `bench_one_batch` backend-initialization trigger and does
not change production code.

## Scope

- Issue context: `sgl-project/sglang` issue 36395.
- Prior follow-up: `amdpilot-org/sglang` issue 218 / PR 287.
- Operation: `benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py::fused_moe_torch`.
- Device: one assigned AMD Instinct MI300X (`gfx942`).
- Shape sequence: token counts `[1, 2, 4]`, eight experts, top-k two.
- Dtypes: BF16 and FP16.
- Timing: one cold call per dtype/shape after a Dynamo reset with a fresh
  job-private Inductor cache, then the mean of five warm calls.
- Reference: an independent float32 per-token/per-expert PyTorch loop.
- Storage: sentinel tensors surround each cold call, inputs are cloned for an
  aliasing check, and input/output addresses are recorded.
- Dispatch: CUDA/ROCm profiler kernel names are recorded for each warm case.
- Unsupported FP8: `use_fp8_w8a8=True` must raise `AssertionError`; it is not
  forced through.

## Reproduce

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-2d5e2de83fe2/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-2d5e2de83fe2/torch-inductor
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-2d5e2de83fe2/sglang-jit
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-2d5e2de83fe2/tvm-ffi
/opt/venv/bin/python reports/j-2d5e2de83fe2/compiled_moe_reuse.py
```

The script writes `gpu-evidence.json` beside this README. The committed JSON is
the raw result from the delivery checkout; rerunning will overwrite it with a
new timing sample.

## Result

All six supported BF16/FP16 cases passed the independent reference gate. The
bounded cold/warm matrix was:

| dtype | tokens | cold (ms) | warm mean (ms) |
| --- | ---: | ---: | ---: |
| BF16 | 1 | 4855.113 | 0.241 |
| BF16 | 2 | 4512.753 | 0.216 |
| BF16 | 4 | 4981.439 | 0.239 |
| FP16 | 1 | 2328.762 | 0.202 |
| FP16 | 2 | 4152.917 | 0.214 |
| FP16 | 4 | 3957.582 | 0.193 |

The unsupported FP8 variant raised:

```text
AssertionError: Fp8_w8a8 fused_moe is not supported for torch compile
```

The maximum absolute reference differences were below `1e-6`. Actual native
kernel names, input/output addresses, and raw timings are retained in
`gpu-evidence.json`.

## Environment

- Image reference: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`.
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Triton: `3.7.0+amd.rocm7.2.0.git89002410`.
- Delivery base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Limitations

- This is a small synthetic kernel investigation, not a full model benchmark.
- `dynamic=False` specializes shape and dtype; it does not claim a static output
  address contract. Addresses are recorded as evidence rather than asserted.
- The benchmark module imports FlashInfer only for its unused benchmark helper.
  When FlashInfer is absent, the report loader supplies a no-op helper so the
  actual SGLang compiled path can still be imported and executed.
- No full model weights, toolchain replacement, unbounded stress, or GPU burn
  are used.
