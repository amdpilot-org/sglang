# gfx942 representation validation for issue 35096

## Scope and identities

- Upstream context: `sgl-project/sglang` issue 35096 and candidate PR 35097, tested at upstream commit `6a7fa215d172afcd66559dcb9751dbca4a403013`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`main`).
- Source: `/job/sglang/python/sglang/kernels/ops/attention/deepseek_v4_rope.py`.
- Python: `/opt/venv/bin/python`; checkout import: `/job/sglang/python/sglang`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; Triton: `3.7.0`; HIP: `7.2.26015-fc0010cf6a`.
- Native paths: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`, `/opt/rocm/lib/libamdhip64.so.7`, and `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- GPU: one AMD Instinct MI300X, `gfx942`, 206,141,652,992 bytes of VRAM.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. The image ID is operator-provided; this container has no Docker socket, and hostname is not used as image identity.

## Installed-source first baseline

The first GPU baseline used the preinstalled source at `/sgl-workspace/sglang`, not the later checkout. It is recorded in `/job/baseline-first.json` and is not proof for checkout changes.

For `[4, 8, 6]` FP32 contiguous input, the independent PyTorch pair-wise cos/sin reference reported:

- Flat kernel: maximum absolute delta `2.4748899936676025`, 2 elements differing above `1e-5`.
- Per-token kernel: maximum absolute delta `0.0`, 0 elements differing above `1e-5`.
- Sentinel-protected suffix: 0 changed elements.
- First GPU execution elapsed time: `1.2479643179103732` seconds.
- Timing method: 3 warmups and 10 synchronized repetitions; clone excluded from the timed interval.

Actual dispatch was `apply_rotary_emb_flat_kernel` with grid `(2,)` and `apply_rotary_emb_triton_kernel` with grid `(4, 8, 1)`.

## Uncovered base representation case

Before applying the candidate fix, the mirror base commit `0084030179bfba86bfeb6d43f7997d4076329d2c` was tested with a `[4, 8, 6]` FP32 packed view using last-dimension stride 2:

- Shape: `[4, 8, 6]`; strides: `[96, 12, 2]`.
- Active maximum absolute delta versus the independent reference: `11576.1123046875`.
- Active elements differing above `1e-5`: 34 of 192.
- Sentinel elements changed: 96 of 192.

This confirms the original bug also corrupts a supported non-contiguous packed representation, not only contiguous storage.

## Fixed-checkout representation matrix

The candidate fix was applied to the mirror checkout. The registered test command was:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-bcc2b24da7fc/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-bcc2b24da7fc/inductor
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_deepseek_v4_rope_flat.py -q --tb=short
```

Result: `60 passed, 1 warning in 7.20s`. The warning is pytest's existing unknown `asyncio_mode` config option.

The focused representation command also passed:

```bash
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_deepseek_v4_rope_flat.py -q -k 'packed_view or noncontiguous_head_stride' --tb=short
```

Result: `20 passed, 40 deselected, 1 warning in 8.67s`.

The bounded matrix used widths `{6, 96, 192}`, layouts `{unpacked, packed_last_dim, strided_head}`, FP32, `[4, 8]` shape, 3 warmups, and 10 synchronized repetitions. Every flat and per-token result had:

- Maximum absolute delta versus the independent reference: `0.0`.
- Mean absolute delta versus the independent reference: `0.0`.
- Elements differing above `1e-5`: 0.
- Sentinel elements changed: 0.

Flat-kernel median timings ranged from `0.04412233829498291` ms to `0.04655495285987854` ms. Raw samples and per-case medians are in `reports/j-bcc2b24da7fc/gfx942-representation-validation.json`.

Actual dispatch summary:

- `apply_rotary_emb_flat_kernel`, grid `(2,)`: 126 launches.
- `apply_rotary_emb_triton_kernel`, grid `(4, 8, 1)`: 48 launches.

## Contracts

- Aliasing: `apply_rotary_emb_triton` returns the same tensor object and mutates it in place.
- Static graph address: the kernel receives `x` and frequency pointers plus runtime strides; no contiguity assumption is added.
- Dtype: FP32 timing is recorded here; FP16 and BF16 numerical gates are covered by the registered test file.
- Unsupported variants: none were forced. All tested non-contiguous and packed views are supported by the stride-aware kernel.

## Limitations

- This result is for one MI300X (`gfx942`) under ROCm 7.2. It does not reproduce the NVIDIA B200 environment or timings in upstream PR 35097.
- `compute-sanitizer` is not installed in this image, so the upstream CUDA memory-check observation was not repeated.
- No model-level evaluation or full model weights were downloaded or run.
- No upstream issue, PR, or comment was posted or modified.
