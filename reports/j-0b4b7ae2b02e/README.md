# FlyDSL fused residual norm investigation on gfx942

## Result

The requested compatibility fix is already present in the current mirror. The
working implementation at
`python/sglang/kernels/ops/diffusion/norm/fused_residual_norm_flydsl.py` is the
FlyDSL v0.3.0 stable-public-API migration from upstream PR 36349, merged as
commit `1fb85053e74c3263c79457a14f589002e3cb8c31`. No kernel rewrite or
replacement is needed, and this branch does not duplicate that fix.

The qualified image's preinstalled SGLang source is older and still contains the
private/removed-API implementation. It works only because its
`flydsl.expr.buffer_ops` import failure falls back to AITER's source-tree copy.
With AITER removed from `sys.path`, that old module fails to import. The
mirror's stable implementation imports from the installed FlyDSL 0.3.1 wheel
alone and passes the same bounded gfx942 tests.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Container hostname (not used as image identity): `banff-cyxtera-cx57-5`
- GPU: one assigned AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, compute
  capability `9.4`, serial `692440004359`, GUID `39656`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP/ROCm: `7.2.26015-fc0010cf6a`, `/opt/rocm-7.2.0`
- FlyDSL: `0.3.1`
- Working mirror commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Image source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`

## API surface and paths

- Working Python module:
  `/job/sglang/python/sglang/kernels/ops/diffusion/norm/fused_residual_norm_flydsl.py`
- Image Python module:
  `/sgl-workspace/sglang/python/sglang/kernels/ops/diffusion/norm/fused_residual_norm_flydsl.py`
- Installed FlyDSL Python package:
  `/opt/venv/lib/python3.10/site-packages/flydsl/__init__.py`
- Installed FlyDSL native extensions:
  `/opt/venv/lib/python3.10/site-packages/flydsl/_mlir/_mlir_libs`
- Torch native libraries:
  `/opt/venv/lib/python3.10/site-packages/torch/lib`
- AITER native module used by the image fallback:
  `/sgl-workspace/aiter/aiter/jit/build/module_aiter_core/build/module_aiter_core.so`
- ROCm HIP runtime:
  `/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200`

The installed FlyDSL no longer exports `flydsl.expr.buffer_ops`. The image's old
module imports `flydsl._mlir`, raw MLIR dialect builders,
`flydsl.compiler.kernel_function.CompilationContext`, and AITER's source-tree
`buffer_ops` fallback. The mirror imports only public `flydsl.expr`,
`flydsl.expr.gpu`, `flydsl.expr.math`, `flydsl.expr.rocdl`, and
`flydsl.compiler` symbols. A manifest audit confirms every required symbol is
present in the relevant FlyDSL `__all__` export list; the raw audit is in
`raw/stable_api_manifest_audit.log`.

## Commands

All caches were kept outside the repository in
`/tmp/sglang-cache-j-0b4b7ae2b02e`.

```bash
export HOME=/tmp/sglang-cache-j-0b4b7ae2b02e/home
export XDG_CACHE_HOME=/tmp/sglang-cache-j-0b4b7ae2b02e/xdg
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-0b4b7ae2b02e/triton
export PYTHONPATH=/job/sglang/python
export HIP_VISIBLE_DEVICES=0
export SGLANG_USE_ROCM_FLYDSL=1

/opt/venv/bin/python -m pytest -q -s \
  test/registered/kernels/ops/diffusion/test_norm_flydsl.py

/opt/venv/bin/python reports/j-0b4b7ae2b02e/probe_gfx942.py \
  --output /tmp/sglang-cache-j-0b4b7ae2b02e/logs/probe_mirror_stable.json

PYTHONPATH=/sgl-workspace/sglang/python \
/opt/venv/bin/python /job/sglang/reports/j-0b4b7ae2b02e/probe_gfx942.py \
  --output /tmp/sglang-cache-j-0b4b7ae2b02e/logs/probe_image_private.json
```

## Numerical gates

The registered suite's gates were unchanged:

- Residual output: `atol=5e-2`, `rtol=5e-2`
- Normalized/scale-shift output: `atol=1.0`, `rtol=5e-2`

The complete registered FlyDSL norm suite passed on gfx942:

```text
22 passed, 1 warning in 12.86s
```

The bounded probe used the same gates and an independent Torch reference. Both
the image fallback implementation and the mirror stable implementation passed:

| Implementation | Dtype | D | Norm | Residual max abs | Output max abs |
|---|---:|---:|---|---:|---:|
| Mirror stable | BF16 | 5120 | RMS | 0.0 | 0.125 |
| Mirror stable | BF16 | 5120 | LayerNorm | 0.0 | 0.125 |
| Mirror stable | FP16 | 5120 | RMS | 0.0 | 0.125 |
| Mirror stable | FP32 | 5120 | RMS | 0.0 | 0.125 |
| Mirror stable | BF16 | 10240 | RMS | 0.0 | 0.125 |
| Image fallback | BF16 | 5120 | RMS | 0.0 | 0.125 |
| Image fallback | BF16 | 5120 | LayerNorm | 0.0 | 0.125 |
| Image fallback | FP16 | 5120 | RMS | 0.0 | 0.125 |
| Image fallback | FP32 | 5120 | RMS | 0.0 | 0.125 |
| Image fallback | BF16 | 10240 | RMS | 0.0 | 0.125 |

For one seeded BF16, D=5120, RMS case, the mirror output and residual were
bit-exact to the image fallback implementation:

```text
y bit_exact=True max_abs=0.0
residual_out bit_exact=True max_abs=0.0
```

## Dtype and tail behavior

- BF16 is the native compute/output dtype.
- FP16 and FP32 inputs are accepted by the public wrapper, converted to BF16
  before the kernel, and return BF16. Only these dtypes were tested; no claim is
  made about every convertible Torch dtype.
- Supported feature dimensions are multiples of `5120`. D=`5120` and D=`10240`
  passed.
- D=`5119`, D=`5121`, and D=`10239` were rejected with
  `AssertionError: FlyDSL fused_residual_norm requires D % 5120 == 0`.
- The public `eps` argument remains intentionally ignored; the kernel uses its
  fixed `1e-6` epsilon. This behavior is unchanged by PR 36349.
- When `weight` is supplied, the current kernel adds `bias` for both RMS and
  LayerNorm. The registered tests pass `bias=None` for RMS, so this observed
  behavior was modeled in the independent reference and left unchanged.

## Raw artifacts

- `raw/pytest.log`
- `raw/pytest.xml`
- `raw/probe_mirror_stable.json`
- `raw/probe_image_private.json`
- `raw/stable_wheel_only_import.log`
- `raw/image_private_without_aiter.log`
- `raw/environment_and_direct_comparison.log`
- `raw/source_provenance.log`
- `raw/stable_api_manifest_audit.log`
- `raw/native_paths.log`

## Left undone

No source change was made because the requested stable-API replacement is
already merged and validated. The `eps` behavior and RMS-with-bias behavior were
recorded but not changed, because both are pre-existing semantics outside this
compatibility task. No model weights were downloaded, no framework stack was
replaced, and no upstream issue, PR, or comment was posted or modified.
