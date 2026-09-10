# gfx942 activation-clamp parity report

## Scope

- Campaign: `repo-e2e-20260909`
- Upstream context: `sgl-project/sglang` issue 16255 and candidate PR 37769.
- Candidate head tested: `b5fdca8dc5def8c0e41d79d2c29c8f3995d0cc6e`.
- Delivery base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- GPU: one assigned AMD Instinct MI300X (`gfx942`).
- Stack: `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`.

This is MoE activation-clamp dispatch parity. It does not change or evaluate MLA
normal/absorbed attention algebra, and it is not a plain activation namespace rename.

## Installed-source baseline

The first GPU execution was a supported small SGLang activation control before any
mirror checkout modification:

```bash
/opt/venv/bin/python - <<'PY'
from sglang.kernels.ops.activation.activation import run_activation
# shape (2, 3, 512), bf16, silu, explicit F.silu reference,
# 3 warmups and 10 timed calls with torch.cuda.Event.
PY
```

- Callable: `sglang.kernels.ops.activation.activation.run_activation`
- Python path: `/sgl-workspace/sglang/python/sglang/kernels/ops/activation/activation.py`
- Installed source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` (dirty environment context)
- First GPU execution wall time: `33.774500 s`
- Mean CUDA-event time: `0.030133399367332458 ms`
- Explicit-reference max absolute error: `0.03125`
- Gate: `torch.allclose(atol=1e-2, rtol=1e-2)`

The complete installed-source evidence is in job-level `baseline-first.json`. It is
clearly labeled as installed-source context and is not proof for later checkout changes.

## Current-main control

The current mirror's non-filtered HIP Triton MoE branch selected
`silu_and_mul_clamp`, whose JIT build failed on gfx942 before a useful numerical
result:

```text
fatal error: 'cuda_fp8.h' file not found
RuntimeError: Failed to build JIT module sgl_kernel_jit_dpsk_v4_silu_and_mul_clamp_bf16_t_false
```

The affected Triton MoE dispatch also contains the explicit platform assertion fixed by
candidate PR 37769. No synthetic burn, unbounded loop, sleep loop, or repeated GPU work
was used.

## Corrected gfx942 parity probe

After applying the candidate correction to the delivery branch, the configured
DeepSeek-style `swiglu_limit=10.0` boundary was exercised in bf16:

- Requested boundary values: `[-10.03125, -10.0, -9.96875, -0.0, 0.0, 9.96875, 10.0, 10.03125]`
- bf16 values: `[-10.0, -10.0, -10.0, -0.0, 0.0, 10.0, 10.0, 10.0]`
- Shape: `[8, 256]` (`8 x 128` gate and up halves)

The independent reference preserved the model-specific clamp order and dtype:

```python
gate = gate.clamp_max(torch.tensor(limit, dtype=torch.bfloat16, device="cuda"))
up = up.clamp(
    torch.tensor(-limit, dtype=torch.bfloat16, device="cuda"),
    torch.tensor(limit, dtype=torch.bfloat16, device="cuda"),
)
expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
```

### Fused path

- Callable: `sglang.kernels.ops.moe.fused_moe_triton_kernels.act_and_mul_triton`
- Python path: `/job/sglang/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`
- Native path: `act_and_mul_kernel` Triton gfx942 kernel in that source file
- Filter-expert path: enabled
- Max absolute error: `0.0`
- Max relative error: `0.0`
- Bit mismatches: `0 / 1024`
- Mean CUDA-event time: `0.02387090027332306 ms`

### Separate path

- Callables: `_clamp_swiglu_inputs_` plus `sgl_kernel.silu_and_mul`
- Python paths:
  - `/job/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
  - `/opt/venv/lib/python3.10/site-packages/sgl_kernel/elementwise.py`
- Native paths: ATen clamp ROCm kernels plus `torch.ops.sgl_kernel.silu_and_mul.default`
- Filter-expert path: disabled
- Max absolute error: `0.0`
- Max relative error: `0.0`
- Bit mismatches: `0 / 1024`
- Mean CUDA-event time: `0.02926749885082245 ms`

Both paths matched each other bit-for-bit (`0 / 1024` mismatches) and passed
`torch.testing.assert_close(atol=1e-2, rtol=1e-2)`.

An additional saturation probe used extreme negative gate values. Both real paths
still matched each other exactly, while the independent Torch formula differed by one
bf16 ULP on `128 / 2048` elements because the native SiLU implementation uses a
different fast-math rounding. That probe is not a clamp-order or clamp-dtype failure.

## Validation

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_CACHE_DIR=/tmp/sglang-cache-j-a8af813d6085
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-a8af813d6085/jit
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-a8af813d6085/triton
/opt/venv/bin/python -m pytest test/registered/unit/layers/moe/test_fused_moe_swiglu_clamp.py -q
```

Result: `3 passed` in `17.61 s` (`20.466800 s` wall time).

All caches were job-private under `/tmp/sglang-cache-j-a8af813d6085`, outside
`/job/sglang`. No model weights were downloaded and no upstream issue, PR, or comment
was posted or modified.
