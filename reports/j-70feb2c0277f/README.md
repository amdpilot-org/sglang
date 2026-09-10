# gfx942 Gemma3 RMSNorm flatten/unflatten investigation

## Scope and conclusion

This is the separate follow-up to amdpilot-org/sglang issue 209 using one assigned AMD Instinct MI300X (`gfx942`). The read-only upstream context is sgl-project/sglang issue 32807 and open candidate PR 32670.

No production code was changed. The actual ROCm dispatch, `Gemma3RMSNorm.forward_hip`, delegates to `forward_native`. Across rank-2 `(37, 256)` and rank-3 `(37, 4, 256)` views, BF16/FP16 activations, and matching versus FP32 weights, all eight cases passed an independently derived float64 reference, remained finite, and had maximum absolute difference `0.0` after casting back to the activation dtype. For all four rank-3 dtype combinations, direct output was exactly equal to output computed on the flattened leading dimensions and restored with `reshape`.

The explicit fused CUDA path cannot be validated on this installed ROCm stack. The `sgl_kernel` wheel exposes the Python wrappers but not the `torch.ops.sgl_kernel.gemma_rmsnorm` or `rmsnorm` registrations. Direct calls therefore fail with `AttributeError` from the installed wheel, while checkout imports fail earlier with `NameError` because the ROCm conditional import does not bind the CUDA-only symbol. This is an unsupported-boundary result, not evidence that flattening is incorrect.

## Environment

- Image reference: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, serial `692440004420`, node ID `3`, GUID `39656`
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Installed `sgl_kernel`: `0.4.6.post1`, `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Installed source context: `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, dirty preinstalled working tree
- Persistent mirror base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Preserved upstream candidate: sgl-project/sglang PR 32670 head `701c8ac088938d97a963d48dfa8d4a792d59b2da`, parent `4e5a05148a2b3cc55eadbf48ff39c99a94546a35`

The installed-source baseline is labeled separately and is not proof for checkout changes. No full model weights were downloaded, no framework or Torch/ROCm stack was replaced, no node-wide state was changed, and no synthetic GPU burn, unbounded loop, or sleep loop was used.

## Commands

Installed-source first GPU attempt:

```bash
cd /sgl-workspace/sglang
timeout 120s /opt/venv/bin/python -m pytest -q \
  python/sglang/kernels/aot/tests/test_norm.py::test_gemma_norm -x --disable-warnings
```

It failed in 7.817 seconds with `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'gemma_rmsnorm'`. The adjacent `test_norm` control failed in 6.976 seconds with the same class of error for `rmsnorm`. The supported control was therefore the real `Gemma3RMSNorm.forward_hip` Torch path:

```bash
PYTHONPATH=/job/sglang/python \
PROBE_OUTPUT=/job/main-0084030-hip-gpu-results.json \
timeout 120s /opt/venv/bin/python /job/baseline_probe.py
```

Persistent checkout and candidate preservation:

```bash
git clone --depth 1 --branch main https://github.com/amdpilot-org/sglang.git /job/sglang
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch --no-tags upstream pull/32670/head:refs/remotes/upstream/pr-32670
git switch --detach 701c8ac088938d97a963d48dfa8d4a792d59b2da
```

Explicit CUDA-path probes:

```bash
PYTHONPATH=/job/sglang/python \
PROBE_SOURCE_COMMIT=701c8ac088938d97a963d48dfa8d4a792d59b2da \
PROBE_OUTPUT=/job/candidate-701c8ac0-gpu-results.json \
timeout 120s /opt/venv/bin/python /job/checkout_probe.py

git switch main
PYTHONPATH=/job/sglang/python \
PROBE_SOURCE_COMMIT=0084030179bfba86bfeb6d43f7997d4076329d2c \
PROBE_OUTPUT=/job/main-0084030-cuda-gpu-results.json \
timeout 120s /opt/venv/bin/python /job/checkout_probe.py
```

## Numerical and timing method

The input was a deterministic finite adversarial matrix containing zero, `1e-4`, positive, and negative values. The independent reference was computed in float64 as:

```python
scale = torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps)
expected = (x.double() * scale * (1.0 + weight.double())).to(x.dtype)
```

The unchanged numerical gates were:

- `torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)`
- `torch.isfinite(actual).all()`
- rank-3 equivalence: `torch.equal(actual, unflattened)`

Timing used three warmups, ten timed calls, CUDA events, and one synchronization after the timed loop. It measured the real operation only and was not repeated to occupy the GPU.

## Raw results

- `baseline-first.json`: installed-source baseline, unsupported native-op errors, supported Torch fallback timings, and all eight reference comparisons.
- `main-0084030-hip-gpu-results.json`: persistent mirror `main` actual HIP dispatch; all eight cases pass and all rank-3 flatten/unflatten comparisons are exactly equal.
- `candidate-701c8ac0-gpu-results.json`: upstream PR 32670 explicit CUDA path. FP32-weight cases fall back and pass; matching half-precision cases are unsupported because the Gemma op is absent.
- `main-0084030-cuda-gpu-results.json`: mirror `main` explicit CUDA path. Rank-2 is unsupported; rank-3 direct output falls back and passes, but its rank-2 flatten control is unsupported, so fused flatten/unflatten cannot be established on this stack.

## Decision

The operation contract and numerical gates are preserved by the actual ROCm path, and no mismatch was demonstrated. Changing `Gemma3RMSNorm` would therefore not be justified by this gfx942 evidence. Upstream PR 32670 already covers the CUDA-side rank guard and mixed-dtype fallback; it was preserved and tested as a candidate rather than duplicated. A CUDA-native rebuild was outside this job's bounded scope because it would require replacing or rebuilding the qualified Torch/ROCm stack.
