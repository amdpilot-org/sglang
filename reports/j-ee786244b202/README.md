# Diffusion upsample GPU validation

## Scope

This change validates the local spatial resize used by the Stable Diffusion
`Upsample2D` path. It does not test denoising scheduler arithmetic or a full
model generation.

The operator uses `torch.nn.functional.interpolate(..., mode="nearest")`.
`align_corners` is not exposed for nearest-neighbor interpolation, so the test
checks the nearest-neighbor coordinate convention against an independent CPU
Torch reference.

## Environment

- GPU: one AMD Instinct MI300X, capability `(9, 4)`.
- Image: local ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Installed SGLang source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Native Torch libraries:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so` and
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Baseline

The installed-source baseline is recorded in `/job/baseline-first.json`. It is
not proof for later checkout changes.

- First relevant GPU execution elapsed time: `0.74570187414065` seconds.
- Timing method: `time.perf_counter` through synchronization for the first
  call; CUDA events with three warmups and ten timed calls for medians.
- FP32 nearest, bilinear, and bicubic cases matched the CPU reference within
  `2e-2` absolute and relative tolerance.
- BF16 cases also matched within `2e-2`; the largest observed absolute error
  was `0.012296915054321289` for bicubic with `align_corners=True`.
- Supported tails included `1x1`, `3x5`, and `17x23` outputs.

## Checkout validation

The persistent mirror checkout adds
`python/sglang/multimodal_gen/test/unit/test_stable_diffusion_upsample2d.py`.
It isolates the resize by replacing `Upsample2D.conv` with `nn.Identity`, then
compares GPU output with a CPU Torch reference for:

- FP32 and BF16.
- Default `2x` upsampling.
- Explicit output sizes `13x17`, `1x1`, `3x5`, and `17x23`.

The numerical gate is exact equality (`rtol=0`, `atol=0`) because
nearest-neighbor interpolation only selects source values.

Reproduction command:

```bash
/opt/venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stable_diffusion_upsample2d.py -q
```

Raw result:

```text
10 passed, 2 warnings in 0.76s
```

## Upstream context

Read-only context was taken from `sgl-project/sglang` issue `23494`, including
its current description and comments. No upstream issue, pull request, or
comment was modified. No already-working upstream fix was duplicated.
