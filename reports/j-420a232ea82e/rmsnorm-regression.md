# AMD RMSNorm regression evidence

The pinned environment did not reproduce an implementation defect, so this
change adds focused ROCm regression coverage without changing production code.

## Environment

- GPU: AMD Instinct MI350X (`gfx950:sramecc+:xnack-`)
- Python: `/tmp/amdpilot-repo-j-420a232ea82e/venv/bin/python`
- Torch: `2.11.0+rocm7.2`
- HIP: `7.2.26015`
- Python implementation: `/job/repo/python/sglang/srt/layers/layernorm.py`
- AITER Python package: `/sgl-workspace/aiter/aiter/__init__.py`
- Loaded native modules:
  - `/tmp/amdpilot-repo-j-420a232ea82e/cache/aiter/module_aiter_core.so`
  - `/tmp/amdpilot-repo-j-420a232ea82e/cache/aiter/module_rmsnorm_quant.so`

## Independent reference and API checks

The reference converts activations (and residuals, when supplied) to float32,
computes the residual sum, mean square, reciprocal square root, normalization,
and weight multiplication independently with Torch, then casts to the activation
dtype for comparison. The tests cover fp16 and bf16 with 1, 7, and 31 rows at a
hidden size of 128.

The plain path uses the legitimate noncontiguous view `storage[:, ::2]`. The
ROCm/AITER module accepts this layout by materializing a contiguous kernel input
and returns a separate output. The fused-residual path likewise returns separate
output and residual tensors. Tests verify numerical accuracy and that both
caller-owned input tensors remain unchanged.

## Commands

All commands set caches outside `/job/repo`:

```bash
env PYTHONPATH=/job/repo/python SGLANG_USE_AITER=1 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/j-420a232ea82e-torch-cache \
  TRITON_CACHE_DIR=/tmp/j-420a232ea82e-triton-cache \
  XDG_CACHE_HOME=/tmp/j-420a232ea82e-xdg \
  /tmp/amdpilot-repo-j-420a232ea82e/venv/bin/python \
  /job/rmsnorm-evidence/prechange_reproducer.py

env PYTHONPATH=/job/repo/python SGLANG_USE_AITER=1 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/j-420a232ea82e-torch-cache \
  TRITON_CACHE_DIR=/tmp/j-420a232ea82e-triton-cache \
  XDG_CACHE_HOME=/tmp/j-420a232ea82e-xdg \
  PYTEST_ADDOPTS='-o cache_dir=/tmp/j-420a232ea82e-pytest-cache' \
  /tmp/amdpilot-repo-j-420a232ea82e/venv/bin/python -m pytest \
  test/manual/layers/test_rmsnorm_amd.py -v
```

Raw outputs are retained at `/job/rmsnorm-evidence/prechange-output.txt` and
`/job/rmsnorm-evidence/pytest-output.txt`.

An initial collection attempt through the broad `test_layernorm.py` module was
blocked by a pre-existing root-owned `/tmp/aiter_configs` directory while an
unrelated MoE import tried to refresh tuned GEMM configuration. Its raw output
is retained at `/job/rmsnorm-evidence/pytest-collection-blocker.txt`; the focused
test is isolated from that unrelated dependency chain.

This operator-level result does not qualify model-serving behavior.
