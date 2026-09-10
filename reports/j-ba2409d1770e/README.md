# gfx942 Gemma3 RMSNorm output-reuse investigation

## Result

This is a distinct execution-representation and reuse follow-up to amdpilot-org/sglang issue 209. It does not repeat that issue's rank-flattening or mixed-dtype trigger, which mirror pull request 342 already reports.

The persistent checkout's actual HIP dispatch for issue 32807's `Gemma3RMSNorm` is:

```text
Gemma3RMSNorm.forward_hip -> Gemma3RMSNorm.forward_native
```

The method has no `out` parameter and no documented safe output-buffer reuse variant. A preallocated, sentinel-filled candidate is not accepted or modified. Passing `out=` fails immediately with:

```text
TypeError: Gemma3RMSNorm.forward_hip() got an unexpected keyword argument 'out'
```

This is honest negative evidence, not a forced reuse implementation. No production code was changed.

## Fresh-output evidence

The checkout probe used three distinct input batches with shapes `(1, 1152)`, `(64, 1152)`, and `(257, 1152)`, bfloat16 activations, bfloat16 weights, and `eps=1e-6`.

Every fresh output:

- matched an independent CPU float64 reference exactly (`max_abs_diff=0.0`, `mean_abs_diff=0.0`, `max_rel_diff=0.0`);
- stayed finite;
- preserved bfloat16 dtype and `cuda:0` placement;
- did not alias its input or the sentinel-filled candidate;
- left the input unchanged;
- left the sentinel-filled candidate unchanged;
- used a distinct output address from the other probe shapes.

The profiler recorded the real native Torch dispatch, including bfloat16-to-float copies, `pow`, mean reduction, `rsqrt`, multiplication, and the final float-to-bfloat copy. Raw names are in `gfx942-results.json`.

## Bounded timing matrix

Timing used three warmups and ten measured calls per shape, with one CUDA event pair around each real `forward_hip` invocation. No sleep loops, synthetic burn, or unbounded repetition were used.

| shape | mean | min | max |
|---|---:|---:|---:|
| `(1, 1152)` | 0.132 ms | 0.125 ms | 0.160 ms |
| `(64, 1152)` | 0.116 ms | 0.109 ms | 0.143 ms |
| `(257, 1152)` | 0.117 ms | 0.112 ms | 0.126 ms |

The reuse arm is not timed because the operation has no supported reuse API.

## Installed-source first baseline

The first GPU execution was completed promptly with the preinstalled interpreter and saved as `baseline-first.json`. It is labeled separately and is not proof for this checkout.

The installed `sgl_kernel` wheel exposes the Python wrappers but does not register the native norm operations on this qualified ROCm stack:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'gemma_rmsnorm'
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'rmsnorm'
```

The supported neighboring control was Torch-native eager Gemma RMSNorm. Its fresh and reused explicit-output control cases passed an independent CPU float64 reference, and its bounded timing was approximately 0.18 ms per call. Unsupported output dtype, shape, and device variants failed clearly. Mixed activation and weight dtypes were not forced through the missing native operation.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, serial `692440003964`, GUID `25408`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Installed source context: `/sgl-workspace/sglang`
- Persistent checkout base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Checkout operation: `/job/sglang/python/sglang/srt/layers/layernorm.py`

The Triton cache was kept outside the repository at `/tmp/sglang-cache-j-ba2409d1770e/checkout-triton`. No full model weights were downloaded, no toolchain or Torch/ROCm stack was replaced, and no node-wide state was modified.

## Reproduce

From the repository root on one assigned MI300X:

```bash
PYTHONPATH="$PWD/python" \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-ba2409d1770e/checkout-triton \
timeout 120s /opt/venv/bin/python \
  reports/j-ba2409d1770e/reproduce_gfx942.py
```

The script performs three correctness calls, three bounded timing arms, one unsupported reuse attempt, and one profiler dispatch capture. It writes `gfx942-results.json`.

The adjacent existing control is:

```bash
PYTHONPATH="$PWD/python" \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-ba2409d1770e/checkout-triton \
timeout 120s /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_minimax_m3_rmsnorm.py::test_gemma_rmsnorm_matches_reference \
  --disable-warnings
```

It passed all six cases.

## Boundaries

- Upstream issue sgl-project/sglang 32807 and open upstream pull request 32670 were read only; no upstream issue, pull request, or comment was changed.
- Open upstream pull request 32670 already covers the CUDA-side rank and mixed-dtype dispatch changes. This report does not duplicate that work.
- Mirror pull request 342 already covers issue 209's rank-flattening and mixed-dtype scope.
- The installed CUDA-native `sgl_kernel` norm operations were not rebuilt because that would replace or augment the qualified Torch/ROCm stack.
- No static graph capture was forced onto an undocumented output-reuse path.
- No aliasing was promoted to a supported contract.

## Raw results

- `baseline-first.json`: installed-source first GPU baseline, concrete missing-op errors, Torch control, and bounded timing.
- `gfx942-results.json`: persistent-checkout fresh-output correctness, dispatch, timing, sentinel checks, and unsupported reuse error.
