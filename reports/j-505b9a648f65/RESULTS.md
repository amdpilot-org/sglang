# MI300X grouped quantized GEMM investigation

## Scope

This investigation used one assigned AMD Instinct MI300X (gfx942) with the qualified
Torch/ROCm stack. It exercised the direct grouped FP8 GEMM path with synthetic expert
groups, without routing maps, dense checkpoint loading, model weights, or a whole-MoE
framework.

Upstream context is sglang issue 15194, a broad quantization roadmap. Its linked PRs were
read, including the merged MXFP8 CUTLASS MoE work in PR 17449 and the scheme/kernel-split
refactors. No linked change already fixes the ROCm pre-quantized activation dtype bug found
here. No upstream issue, PR, or comment was posted or changed.

## Installed-source first baseline

The first useful GPU baseline used the supported neighboring ROCm operator
`torch._scaled_grouped_mm`, not the unsupported native `sgl_kernel` CUTLASS grouped FP8
op. The baseline is recorded in `baseline-first.json` and is explicitly labeled as an
installed-source result, not proof for later checkout changes.

The direct installed `sgl_kernel.fp8_blockwise_scaled_grouped_mm` dispatch fails with:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fp8_blockwise_scaled_grouped_mm'
```

The existing test collector also fails on ROCm because its NVIDIA skip predicate compares
`torch.version.cuda` (`None`) with a string. The native op remains unavailable on gfx942;
that test predicate was left unchanged in this focused change.

## Persistent-checkout result

The persistent mirror checkout was cloned from `main` at commit
`0084030179bfba86bfeb6d43f7997d4076329d2c`. The direct Triton grouped FP8 GEMM path
accepted pre-quantized CUDA `float8_e4m3fn` activations, but re-quantized ROCm
`float8_e4m3fnuz` activations because the check hard-coded the CUDA dtype. This discarded
the caller-provided activation scales and produced incorrect results.

The fix compares against the platform FP8 dtype selected by `fp8_dtype`. The focused
regression constructs manual group boundaries and covers:

- empty experts;
- one-token groups;
- mixed small groups;
- activation block-scale broadcasting;
- weight block-scale broadcasting;
- sentinel-filled output detection;
- a finite output bound;
- independent float32 dequantized `torch.mm` references.

The numerical gates are unchanged: exact equality (`rtol=0`, `atol=0`), no sentinel values
remaining, all finite values, and output magnitude no greater than 2048.

## Raw results

Post-fix raw results are in `post-patch.json`. Timing uses one real dispatch followed by
20 bounded CUDA-event timed warm dispatches.

| Case | Groups | Max reference difference | First dispatch | Warm mean |
| --- | --- | ---: | ---: | ---: |
| empty and one-token | `[0, 1, 2, 3]` | 0.0 | 779.52 ms | 0.07987 ms |
| mixed small groups | `[0, 1, 0, 2, 1, 0, 3, 1]` | 0.0 | 0.253 ms | 0.07409 ms |

The first case includes Triton kernel compilation. The second case reuses the compiled
kernel. No synthetic burn, unbounded loop, sleep loop, or repeated occupancy work was used.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-505b9a648f65/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-505b9a648f65/inductor

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/moe/test_fused_moe_group_boundaries.py

/opt/venv/bin/python \
  reports/j-505b9a648f65/benchmark_group_boundaries.py
```

## Environment

- GPU: one AMD Instinct MI300X, capability `(9, 4)`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Python: `/opt/venv/bin/python`, Python `3.10.12`.
- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Triton native: `/opt/venv/lib/python3.10/site-packages/triton/_C/libtriton.so`.
- Installed `sgl_kernel` native: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Image reference supplied by the operator: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`.
- Operator-supplied local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Container hostname is not image identity.

## Limits

The native CUTLASS `sgl_kernel` grouped FP8 op was not run numerically because it is not
registered for gfx942. The CDNA4 `tl.dot_scaled` MXFP8 grouped kernel is gfx95-only and was
not used as a claimed MI300X result. No full model weights or additional framework stack
were downloaded.
