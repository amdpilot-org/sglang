# gfx942 activation namespace validation

## Scope

This records the runtime validation for the public `silu_and_mul` and
`gelu_and_mul` operators after the `sglang.kernels` namespace migration
described in read-only upstream issue `sgl-project/sglang#29630`.

The installed source was used only for the first baseline. The persistent
mirror checkout under `/job/sglang` was used for the code change and all
post-change GPU validation.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, 304 multiprocessors.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP runtime reported by Torch: `7.2.26015-fc0010cf6a`.
- Python: `/opt/venv/bin/python`.
- Image: operator-provided local image ID
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Installed SGLang baseline: `0.5.18.dev20260826+g937af8538b`.
- Installed public callable:
  `/sgl-workspace/sglang/python/sglang/kernels/ops/activation/__init__.py`.
- Installed AOT Python wrapper:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/elementwise.py`.
- Native AOT op: `torch.ops.sgl_kernel.silu_and_mul.default`.
- JIT source: `python/sglang/kernels/jit/csrc/elementwise/activation.cuh`.

## Installed-source baseline

The first real public call completed in `65.53 ms`, including module import and
first JIT/AOT dispatch. Public dispatch selected `KernelBackend.AOT` on HIP.
Timing used CUDA events with three warmups and the mean of ten measured calls.

The baseline used 4096 rows and output widths 8, 64, and 4096. Public dispatch,
direct `sgl_kernel`, and the native op followed the same path. The independent
reference computed gate and up slices in float32 on CPU and cast back to the
input dtype.

| dtype | width | public µs | direct AOT µs | native op µs | Torch ref µs | max abs diff |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 8 | 18.13 | 14.68 | 12.04 | 85.39 | 0.00390625 |
| FP16 | 64 | 25.45 | 20.96 | 17.08 | 107.53 | 0.00390625 |
| FP16 | 4096 | 30.69 | 29.75 | 29.10 | 186.81 | 0.0078125 |
| BF16 | 8 | 18.17 | 15.50 | 14.22 | 85.63 | 0.03125 |
| BF16 | 64 | 22.42 | 19.03 | 15.55 | 104.60 | 0.03125 |
| BF16 | 4096 | 34.05 | 32.91 | 32.37 | 189.81 | 0.0625 |

The BF16 differences are low-precision rounding differences, not dispatch
disagreement; public, direct AOT, and native results had the same maxima.

Unsupported tail controls were recorded without relaxing the gate:

- Output widths 1 and 3 raised the AOT pointer-alignment `ValueError`.
- Output width 4 exposed the sharper wrapper gap: the AOT Python wrapper
  accepted the 16-byte input row, but the native kernel launched zero threads
  for the 8-byte output row and left the supplied `out` unchanged.
- The installed ROCm/Torch build also intermittently raised
  `hipErrorInvalidConfiguration` for tiny GPU `torch.randn`, clone, fill, and
  elementwise comparison launches. CPU-generated inputs and CPU comparisons
  were used as the meaningful neighboring control; all measured operator calls
  remained on the GPU.

The direct JIT path was then forced explicitly on gfx942. It compiled under
`hipcc`, preserved the supplied `out` object, and matched the independent
reference for both FP16 and BF16. This disproved the wrapper metadata that
marked JIT as CUDA-only. AOT remained faster on the measured HIP shapes, so
the fix preserves AOT as the HIP default.

Representative silu timings were:

| dtype | width | AOT µs | JIT µs |
|---|---:|---:|---:|
| FP16 | 8 | 8.17 | 12.03 |
| FP16 | 64 | 11.57 | 16.67 |
| FP16 | 4096 | 29.68 | 32.03 |
| BF16 | 8 | 8.30 | 11.34 |
| BF16 | 64 | 11.49 | 17.13 |
| BF16 | 4096 | 32.26 | 32.35 |

The zero-row candidate `sgl-project/sglang#23636` was also checked. Public and
direct zero-row calls already returned the supplied empty output in this
installed ROCm wheel, so that underlying fix was not duplicated.

## Fix

The public activation wrapper now:

1. marks gated-activation JIT as CUDA and HIP capable;
2. keeps AOT as the HIP default through a platform-aware candidate order;
3. rejects AOT output widths below the 16-byte vector width instead of
   silently launching zero threads and leaving `out` unchanged.

Explicit JIT remains available for A/B testing on HIP without changing the
production default.

## Reproduction

From the repository root:

```bash
export PYTHONPATH="$PWD/python"
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-7459dc445c62

AMD_SERIALIZE_KERNEL=1 /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_kernels_namespace.py

AMD_SERIALIZE_KERNEL=1 /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/activation/test_activation.py \
  -k public_namespace_activation

AMD_SERIALIZE_KERNEL=1 /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py \
  -k gated_activation
```

Observed on the assigned MI300X:

- namespace metadata suite: 24 passed.
- public activation parity and unsupported-width suite: 28 passed.
- every-backend gated-activation GPU parity suite: 12 passed.

## Limitations

- The installed AOT wheel itself still contains the width-4 zero-thread launch;
  this change prevents the public SGLang wrapper from reaching that invalid
  native launch.
- The JIT width gate is left to its existing native runtime check. The public
  test covers supported JIT widths 8, 64, and 4096.
- No upstream issue, PR, or comment was posted or modified.
