# ROCm DWDP startup investigation: j-65d3656d5fae

## Scope and conclusion

- Upstream context read only: sgl-project/sglang issue 31995 and open pull request 31996. No upstream issue, pull request, or comment was created or modified.
- Working clone: `/job/j-65d3656d5fae`, branch `amdpilot/j-65d3656d5fae`, based on mirror `main` at `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`.
- Tested upstream candidate commit: `4154f8a0e02896db248afc218f5739762d118e75` from sgl-project/sglang pull request 31996.
- The unmodified mirror `main` fails `from sglang.srt.layers.moe.dwdp import DwdpManager` on this ROCm node because `cuda.bindings` is absent.
- The current source no longer contains the issue's `vmm.py`; the observed unconditional imports are in `dwdp/transport.py` and `dwdp/page_pool.py`.
- This change guards only those observed imports. It does not emulate CUDA and does not alter the qualified Torch/ROCm stack.
- Enabled DWDP transport still refuses explicitly when `cuda.bindings` is unavailable.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, device ID `0x74a1`, GUID `61795`, GFX version `gfx942`, CUDA capability reported by Torch as `(9, 4)`.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, source `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, native module `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Installed SGLang context: version `0.5.18.dev20260826+g937af8538b`, source `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Tested clone source: `/job/j-65d3656d5fae/python/sglang/__init__.py`.
- `sgl_kernel`: version `0.4.6.post1`, source `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Aiter native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Triton source: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- `cuda` and `cuda.bindings` are unavailable in the qualified environment.

## Validation

DWDP was disabled for the synthetic run. `ServerArgs.dwdp_size` has default value `1`. No model weights were downloaded.

The reduced path imported `DwdpManager` and `sglang.srt.model_executor.model_runner`, then initialized and ran `SiluAndMul` on MI300X:

```bash
PYTHONPATH=/job/j-65d3656d5fae/python /opt/venv/bin/python - <<'PY'
import sys
import torch
import sglang
from sglang.srt.layers.moe.dwdp import DwdpManager
import sglang.srt.model_executor.model_runner as model_runner
from sglang.srt.layers.activation import SiluAndMul

torch.manual_seed(65)
torch.cuda.set_device(0)
x = torch.randn(128, 64, device="cuda", dtype=torch.float32)
layer = SiluAndMul()
actual = layer(x)
expected = layer.forward_native(x)
torch.cuda.synchronize()
print("cuda_bindings_loaded", "cuda.bindings.driver" in sys.modules)
print("max_abs_error", (actual - expected).abs().max().item())
print("allclose_exact", torch.allclose(actual, expected, rtol=0.0, atol=0.0))
PY
```

Observed branch results:

- `DwdpManager` import: success.
- `ModelRunner` module import: success.
- `cuda.bindings.driver` loaded: `False`.
- Output shape and dtype: `(128, 32)`, `torch.float32`.
- Numerical gate: `torch.allclose(..., rtol=0.0, atol=0.0) == True`.
- Maximum absolute error: `0.0`.

Enabled CUDA-only transport refusal probe:

```bash
PYTHONPATH=/job/j-65d3656d5fae/python /opt/venv/bin/python - <<'PY'
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
try:
    DWDPTransport.create(None, None, None, None, 0)
except RuntimeError as exc:
    print(type(exc).__name__, exc)
PY
```

Observed result: `RuntimeError: DWDP transport requires cuda.bindings, which is unavailable on this platform`.

The exact upstream candidate commit was also tested in `/job/candidate-31996`. Its reduced-path result was identical: no `cuda.bindings` import, `max_abs_error=0.0`, and exact allclose against the native reference.

## Raw logs

- `baseline.log`: unmodified mirror `main` import failure.
- `candidate-31996.log`: exact upstream candidate commit validation.
- `validation.log`: branch syntax check, synthetic MI300X run, and explicit refusal.
- `environment.log`: Python, Torch, SGLang, kernel, Aiter, and Triton paths.
- `gpu.log`: `rocm-smi --showproduct --showid` output.

## Commands and limits

- Repository operations used bounded HTTPS clone/fetch retries.
- Validation used `/opt/venv/bin/python`; no package install, upgrade, or framework-stack change was performed.
- No full model download or full ModelRunner/model load was performed.
- No multi-rank DWDP run was attempted because the current transport is CUDA-only and only one MI300X was assigned.
- No CUDA environment was available for positive `cuda.bindings` testing.
- `ruff` was unavailable in the image; syntax was checked with `py_compile`, and `git diff --check` was clean.
