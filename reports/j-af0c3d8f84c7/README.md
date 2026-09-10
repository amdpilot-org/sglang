# MI300X timestep embedding validation

## Scope

This investigation covers the GPU timestep/noise-schedule transform only. It does not test the later denoising update or model-level generation.

- Campaign: `repo-e2e-20260909`
- Job: `j-af0c3d8f84c7`
- GPU: one AMD Instinct MI300X, CUDA capability `(9, 4)`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Python source: `python/sglang/kernels/ops/diffusion/modulate/timestep_embedding_jit.py`
- Native source: `python/sglang/kernels/jit/csrc/diffusion/timestep_embedding.cuh`
- Diffusion wrapper: `python/sglang/multimodal_gen/runtime/layers/visual_embedding.py`
- Job-private JIT cache: `/tmp/sglang-cache-j-af0c3d8f84c7/jit`

## Installed-source baseline

The installed source was revision `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, imported from `/sgl-workspace/sglang/python/sglang`. Its first timestep JIT execution could not build on ROCm because `cuda_runtime.h` was unavailable. The exact error and a supported PyTorch GPU formula control are recorded in `/job/baseline-first.json`. The baseline is installed-source evidence only and is not proof for checkout changes.

The first successful post-fix GPU test, including JIT compilation, completed in `19.60 s`.

## Commands

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-af0c3d8f84c7/jit \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/diffusion/test_modulate.py \
  -k timestep_embedding
```

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-af0c3d8f84c7/jit \
/opt/venv/bin/python /tmp/validate_timestep.py
```

The independent reference used `mpmath` with 80 decimal digits. Timing used CUDA events around 20 complete calls after three warmups.

## Results

- Existing numerical gates: `793 passed, 226 deselected`; unchanged `atol=rtol=1e-3`.
- Scalar versus eight-example broadcast at `t=0.375`: exact equality, maximum absolute error `0.0`.
- Endpoint maximum absolute error: `0.0` at `t=0`, `4.57e-8` at `t=1`, `2.90e-5` at `t=999`, and `4.14e-5` at `t=1000`.
- Supported dtype conversion maximum absolute error: `4.14e-5` for float32, float16, bfloat16, and int64-to-float32.
- Caller input storage, data pointer, byte size, and values were preserved.
- ROCm diffusion wrapper path was active and matched diffusers exactly.
- Mean complete GPU call time: `0.0075053 ms`.

Raw results are in `reports/j-af0c3d8f84c7/gpu-validation.json`.

## Upstream context

Read-only review covered sgl-project/sglang issue 23494 and related timestep changes in PRs 12995, 16766, 17658, 32045, and 35114. Issue 23494 is the AMD roadmap issue and contains no direct timestep fix. The current main still had the ROCm include and device-matcher failures reproduced here, so this change does not duplicate an already-working upstream fix.

No upstream issue, PR, or comment was modified. No model weights were downloaded and no node-wide state was changed.
