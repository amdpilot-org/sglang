# Gemma3 RMSNorm MI300X investigation

## Scope

This report covers sgl-project/sglang issue 32807 and upstream pull request 32670 on one assigned AMD Instinct MI300X (gfx942). The delivery intentionally does not duplicate the open upstream fix. It records the tested candidate commit, native ROCm behavior, an explicit supported control, numerical comparisons, and bounded timings.

- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Upstream candidate: `701c8ac088938d97a963d48dfa8d4a792d59b2da` from sgl-project/sglang pull request 32670
- Candidate parent: `4e5a05148a2b3cc55eadbf48ff39c99a94546a35`
- GPU: AMD Instinct MI300X, capability `9.4`, 206141652992 bytes HBM
- Qualified image declared by the operator: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Declared local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP runtime `7.2.26015-fc0010cf6a`

The container cannot query the host image store, so the image identity above is the operator-declared identity rather than an independently measured digest.

## Installed-source baseline

The early baseline used the preinstalled source, not the later mirror checkout, and is therefore not proof for checkout changes.

- Installed sglang: `0.5.18.dev20260826+g937af8538b`
- Installed Python source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed layernorm source: `/sgl-workspace/sglang/python/sglang/srt/layers/layernorm.py`
- Installed native package: `sgl_kernel 0.4.6.post1` at `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Installed ROCm Triton control: `/sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/minimax_m3_rmsnorm.py`
- Normal module dispatch: `Gemma3RMSNorm.forward_hip`
- Explicit `forward_cuda` attempt: `NameError: name 'gemma_rmsnorm' is not defined`
- Direct `sgl_kernel.gemma_rmsnorm` attempt: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'gemma_rmsnorm'`

The first GPU call began 2.587 seconds after baseline process start and its first synchronization completed 2.592 seconds after process start. Timing used CUDA events with five warmups and fifty timed calls, one call per iteration.

| Path | Microseconds per call |
| --- | ---: |
| Installed module native dispatch, 2-D BF16 | 93.209 |
| Direct ROCm Triton Gemma RMSNorm, 2-D BF16 | 39.652 |
| Installed module native dispatch, 3-D BF16 | 92.488 |
| Direct ROCm Triton Gemma RMSNorm, 3-D BF16 | 42.953 |

The complete installed-source record is retained outside the repository at `/job/baseline-first.json`.

## Mirror-main results

The persistent checkout is `/job/sglang`. Its normal `Gemma3RMSNorm.forward()` dispatch resolves to `forward_hip`, which delegates to `forward_native`. This is a real native fallback on gfx942, not the CUDA fused path described by the upstream issue.

The validator covered:

- 2-D hidden states and rank-3 Q/K views `[tokens, heads, head_dim]`
- Contiguous, transposed, and last-dimension-sliced rank-3 views
- BF16 and FP16 activations
- Matching half-precision weights and FP32 weights
- The BF16 residual path
- An independent Torch reference using FP32 mean-square, reciprocal square root, Gemma `(1 + weight)` scaling, and activation-dtype output

All 17 module cases and all direct ROCm Triton control cases passed the unchanged numerical gate `torch.allclose(..., rtol=1e-2, atol=1e-2)`. No output contained NaN or Inf. The largest direct-control difference was `0.0009765625` in FP16 cases. The normal module output was bit-identical to the independent reference in every non-residual case because both used the same Torch native implementation.

Timing used the same bounded CUDA-event method on mirror main:

| Path | Microseconds per call |
| --- | ---: |
| Module native dispatch, 2-D BF16 | 62.527 |
| Direct ROCm Triton Gemma RMSNorm, 2-D BF16 | 23.357 |
| Module native dispatch, 3-D BF16 | 59.068 |
| Direct ROCm Triton Gemma RMSNorm, 3-D BF16 | 23.136 |

Raw results are in `mi300x-main.json`.

## Candidate PR 32670 results

The preserved candidate commit changes `Gemma3RMSNorm.forward_cuda` to:

- Fall back to native Torch for unsupported activation dtypes, mismatched weight dtypes, mismatched devices, or a mismatched final dimension.
- Fuse rank-2 inputs directly.
- Flatten leading dimensions for rank-3 Q/K views, invoke the fused kernel, and restore the original shape.
- Preserve the existing in-place residual contract.

Run unmodified on gfx942, the candidate's four focused Gemma3 tests produced 72 errors. Its older `forward_hip` path calls `forward_cuda`, but the CUDA-only globals `gemma_rmsnorm` and `gemma_fused_add_rmsnorm` are not imported on ROCm. The resulting failures are `NameError`, not numerical failures.

The candidate was then exercised with the installed ROCm Triton Gemma RMSNorm kernel substituted for the missing CUDA globals. The residual substitute also copied the returned normalized and pre-norm tensors back into the input and residual tensors to preserve the CUDA kernel's in-place contract. With that explicit substitute, all four focused candidate tests passed:

```text
Ran 4 tests in 2.005s
OK
SUBSTITUTE_RESULT tests=4 errors=0 failures=0
```

The same substitute harness produced these dispatch counts across the 17-case matrix:

- Matching-dtype 2-D and rank-3 cases: one `rmsnorm` call each
- FP32-weight BF16 and FP16 activation cases: zero kernel calls, native fallback
- Residual case: one `fused_add_rmsnorm` call
- All module and direct-control comparisons passed the unchanged `1e-2` gate
- No output contained NaN or Inf

The largest candidate substitute difference was `0.015625` in the BF16 residual case, within the unchanged gate. Raw results are in `mi300x-candidate-32670.json`.

## Architecture-specific limitations

- `sgl_kernel.gemma_rmsnorm` and `sgl_kernel.gemma_fused_add_rmsnorm` are unavailable in this qualified ROCm stack, so the upstream CUDA kernel cannot be executed natively on gfx942.
- Mirror-main normal dispatch is `forward_hip -> forward_native`, so the candidate's `forward_cuda` change is not reached through normal module dispatch on this architecture.
- The candidate commit predates the current mirror-main dispatch refactor. In that older code, `forward_hip` calls `forward_cuda`, which causes the missing-global `NameError` on ROCm.
- The ROCm Triton control is a meaningful supported neighbor, not proof of CUDA `sgl_kernel` behavior. It performs FP32 normalization math and supports arbitrary rank through `reshape`.
- The residual substitute adds copies to emulate the CUDA in-place contract. Its timings are intentionally omitted; only the native-dispatch and direct-control paths were timed.
- No full model weights were downloaded, no node-wide state was changed, and no upstream issue, pull request, or comment was posted or modified.

## Reproduction

Use one assigned MI300X and keep the Triton cache outside the checkout:

```bash
cd /job/sglang
export TRITON_CACHE_DIR=/tmp/sglang-cache-52587/triton
mkdir -p "$TRITON_CACHE_DIR"

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-52587d209a6b/validate_mi300x.py \
  --mode main \
  --output reports/j-52587d209a6b/mi300x-main.json

PYTHONPATH=/tmp/sglang-candidate-32670-52587/python /opt/venv/bin/python \
  reports/j-52587d209a6b/validate_mi300x.py \
  --mode candidate \
  --output reports/j-52587d209a6b/mi300x-candidate-32670.json
```

The candidate worktree was created read-only from the preserved upstream ref:

```bash
cd /job/sglang
git worktree add --detach /tmp/sglang-candidate-32670-52587 \
  refs/remotes/upstream-pr/32670
```

The focused candidate tests were loaded directly from the worktree file because `test.manual` is not an importable package:

```bash
cd /tmp/sglang-candidate-32670-52587
PYTHONPATH=/tmp/sglang-candidate-32670-52587/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-52587/triton \
/opt/venv/bin/python - <<'PY'
import importlib.util, unittest
from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
    gemma_fused_add_rmsnorm as rocm_fused,
    gemma_rmsnorm as rocm_rmsnorm,
)
from sglang.srt.layers import layernorm

def fused_add_cuda_contract(x, residual, weight, eps):
    out, residual_out = rocm_fused(x, residual, weight, eps)
    x.copy_(out)
    residual.copy_(residual_out)
    return x, residual

layernorm.gemma_rmsnorm = rocm_rmsnorm
layernorm.gemma_fused_add_rmsnorm = fused_add_cuda_contract

spec = importlib.util.spec_from_file_location(
    "test_layernorm_candidate_substitute",
    "test/manual/layers/test_layernorm.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.TestLoader().loadTestsFromTestCase(module.TestGemma3RMSNorm)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
```
