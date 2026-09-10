# MI300X norm/projection/activation block check

## Result

The structured-input run found and fixed a real defect in the public
`sglang.kernels.ops.layernorm.fused_add_rmsnorm` AITER backend. The wrapper passed
AITER's `residual_out` buffer where `residual_in` was expected, and passed the input
residual where AITER expected the output residual. This produced finite but badly
wrong norm and residual values.

The fix preserves AITER's documented call order:

```text
rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)
```

Before the fix, one bounded BF16 reproduction had:

- normalized-output maximum absolute error: `6.3984375`
- residual-output maximum absolute error: `6.875`
- native-reference gate: failed

After the fix, all six structured cases pass the unchanged BF16 gate
`atol=2e-2, rtol=2e-2`, remain finite, and return `torch.bfloat16`. Raw evidence is
in `prefix-failure.json`, `results.json`, and `run.log`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, CUDA capability `(9, 4)`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Checkout SGLang import: `/job/sglang/python/sglang/__init__.py`
- Torch import: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- AITER import: `/sgl-workspace/aiter/aiter/__init__.py`
- AITER native core: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- AITER RMSNorm native module: `/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so`
- `sgl_kernel` import: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`

The installed-source baseline was captured before cloning or editing the mirror. It
ran four selected real-GPU SiLU cases from the existing activation test in
`17.396 s` and is recorded in `/job/baseline-first.json`. That baseline is
installed-source evidence only and is not proof for checkout changes.

## Operators and reference

The reduced block uses:

1. `sglang.kernels.ops.layernorm.fused_add_rmsnorm`
2. `sglang.srt.layers.linear.ReplicatedLinear` with
   `UnquantizedLinearMethod` and `SGLANG_USE_AITER=1`
3. `sglang.kernels.ops.activation.silu_and_mul`

The independent reference computes in float32 inside the norm, casts the new
residual and normalized value to BF16, uses `torch.nn.functional.linear` for the
projection, computes SiLU in float32, casts the gate activation to BF16, and
multiplies by the BF16 up-projection half. This follows the existing public
operator references and keeps the documented BF16 stability contract.

The projection is a supported SGLang projection layer, but AITER's `tgemm` reports
`using torch solution:0` for `M=16, N=2048, K=512`. No native AITER GEMM result is
claimed for this shape. The JIT tiny-GEMM path is also not a fabricated control:
its test explicitly skips HIP and requires SM90+, and the installed
`sgl_kernel` does not expose `dsv3_fused_a_gemm`.

## Structured cases

All cases use identical shapes: 16 tokens, 512 hidden values, a 2048-wide
projection, and a 1024-wide gated activation. Synthetic tensors total
`2,294,784` bytes, well below 4 GB. No checkpoint or model weights were
downloaded.

| Case | Complete-output max abs error | Gate |
|---|---:|---:|
| zeros | `0.0` | pass |
| tiny finite | `2.86102294921875e-06` | pass |
| mixed magnitudes | `0.0234375` | pass |
| cancellation | `3.0517578125e-05` | pass |
| sparse skew | `0.0234375` | pass |
| random baseline | `0.0234375` | pass |

Maximum relative error is reported in `results.json`, but it is not used as the
gate because several expected values are near zero. The gate is the existing
BF16 `torch.allclose(atol=2e-2, rtol=2e-2)` contract.

## Contracts

All checked contracts pass:

- `fused_add_rmsnorm` returns `None`.
- `fused_add_rmsnorm` mutates both `input` and `residual` in place.
- Projection preserves its input tensor and weight.
- `silu_and_mul(input, out)` returns the supplied `out` object.
- `silu_and_mul` leaves its input unchanged when `out` is supplied.
- Every complete-block output is `torch.bfloat16`.
- Every complete-block output is finite.

## Timing attribution

Timing uses CUDA events with five warmups and 30 measured launches. Input resets
for the mutating norm are synchronized outside the timed interval. Values below
are mean milliseconds.

| Case | Norm | Projection | Activation | Operator sum | Whole chain | Chain minus sum |
|---|---:|---:|---:|---:|---:|---:|
| zeros | 0.088754 | 0.064939 | 0.019386 | 0.173079 | 0.185506 | 0.012427 |
| tiny finite | 0.088131 | 0.068481 | 0.019958 | 0.176570 | 0.186257 | 0.009688 |
| mixed magnitudes | 0.087285 | 0.064788 | 0.019542 | 0.171615 | 0.185638 | 0.014023 |
| cancellation | 0.086191 | 0.061776 | 0.018996 | 0.166962 | 0.182001 | 0.015039 |
| sparse skew | 0.087059 | 0.062672 | 0.019358 | 0.169090 | 0.180678 | 0.011588 |
| random baseline | 0.086576 | 0.063279 | 0.018969 | 0.168824 | 0.180779 | 0.011955 |

The whole-chain cost is approximately 6–9% above the sum of independently timed
operators. This is launch and composition overhead, not an unbounded throughput
claim.

## Reproduction

Use one visible MI300X and keep the JIT cache outside the repository:

```bash
mkdir -p /tmp/sglang-cache-j-1184a345bbd6
SGLANG_USE_AITER=1 \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-1184a345bbd6 \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python reports/j-1184a345bbd6/gpu_block_check.py \
  --output reports/j-1184a345bbd6/results.json
```

Focused regression:

```bash
SGLANG_USE_AITER=1 \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-1184a345bbd6 \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_fused_add_rmsnorm_aiter_hip.py
```

Existing public-operator parity control:

```bash
SGLANG_USE_AITER=1 \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-1184a345bbd6 \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py \
  -k fused_add_rmsnorm
```

Observed results:

- Focused regression: 1 passed.
- Existing fused-add parity selection: 6 passed, 20 deselected.
- Six-case block harness: all cases passed.

## Scope and limits

- Upstream issue 29630 was read only; no upstream issue, PR, or comment was changed.
- The public `sglang.kernels.ops` surface from that RFC was exercised directly.
- Only one MI300X was used.
- The projection backend selected AITER's Torch solution for this shape, so the
  projection timing is not evidence of a native AITER GEMM kernel.
- No full model, checkpoint, framework replacement, node-wide state change, GPU
  burn, or unbounded repetition was used.
