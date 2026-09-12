# MI350X AMD environment qualification

Result: **qualified** for the bounded AITER-backed RMSNorm checks below. No
SGLang source defect was reproduced and no source or toolchain files were
changed.

## Fixed environment

- Checkout: `/job/repo`
- Branch: `amdpilot/j-6f40a8eaec02`
- Revision/base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recorded interpreter: `/tmp/amdpilot-repo-j-6f40a8eaec02/venv/bin/python`
  (resolves to `/usr/bin/python3.12`)
- Python: `3.12.3`; Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- Native tools: `/opt/rocm/bin/hipcc`, `/opt/rocm/bin/amdclang++`,
  `/opt/rocm/bin/rocm-smi`, `/opt/rocm/bin/rocminfo`
- GPU: one `AMD Instinct MI350X`,
  `gfx950:sramecc+:xnack-`, 270566162432 bytes
- Ordinary job identity: `uid=15310(qinwu) gid=15310`
- External caches: `/tmp/amdpilot-repo-j-6f40a8eaec02/cache/{aiter,torch_extensions,triton,xdg}`
- Vendor configuration cache: `/tmp/aiter_configs`

The prepared AITER RMSNorm extension was
`/tmp/amdpilot-repo-j-6f40a8eaec02/cache/aiter/module_rmsnorm_quant.so`.
Relevant `ldd` resolutions included:

```text
libc10.so => /opt/venv/lib/python3.12/site-packages/torch/lib/libc10.so
libtorch_cpu.so => /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7
libhsa-runtime64.so.1 => /opt/rocm/lib/libhsa-runtime64.so.1
libamd_comgr.so => /opt/venv/lib/python3.12/site-packages/torch/lib/libamd_comgr.so
```

`rocminfo` reported:

```text
Name:                    gfx950
Marketing Name:          AMD Instinct MI350X
Name:                    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-
```

## AITER configuration-cache permission

Command (run with the recorded interpreter as UID 15310):

```bash
stat -c '%A %a %U:%G %u:%g %n' /tmp/aiter_configs
/tmp/amdpilot-repo-j-6f40a8eaec02/venv/bin/python - <<'PY'
import os
from pathlib import Path
cache = Path('/tmp/aiter_configs')
lock = cache / 'j-6f40a8eaec02.permission.lock'
config = cache / 'j-6f40a8eaec02.permission.json'
replacement = cache / 'j-6f40a8eaec02.permission.json.new'
for p in (lock, config, replacement):
    try: p.unlink()
    except FileNotFoundError: pass
lock.write_text('lock-created\n')
config.write_text('{"generation": 1}\n')
replacement.write_text('{"generation": 2}\n')
os.replace(replacement, config)
print('uid=', os.getuid(), ' gid=', os.getgid(), sep='')
for p in (lock, config):
    st = p.stat()
    print(f'{p}: mode={oct(st.st_mode & 0o777)} uid={st.st_uid} gid={st.st_gid} content={p.read_text().strip()!r}')
lock.unlink(); config.unlink()
print('cleanup=passed')
PY
```

Raw output:

```text
drwxrwxrwx 777 root:root 0:0 /tmp/aiter_configs
uid=15310 gid=15310
/tmp/aiter_configs/j-6f40a8eaec02.permission.lock: mode=0o644 uid=15310 gid=15310 content='lock-created'
/tmp/aiter_configs/j-6f40a8eaec02.permission.json: mode=0o644 uid=15310 gid=15310 content='{"generation": 2}'
cleanup=passed
```

This demonstrates creation of a lock file, creation and atomic replacement of
a configuration file, readback, and cleanup without elevated privileges.

## Source-driven dependency imports

`test/manual/layers/test_layernorm.py` imports
`sglang.srt.layers.layernorm` and `sglang.test.test_utils`. With
`SGLANG_USE_AITER=1`, `layernorm.py` imports AITER's LayerNorm/RMSNorm symbols;
the MoE collection dependency was checked at its existing
`sglang.srt.layers.moe.topk` path, which imports `aiter.fused_moe`.

Command excerpt:

```bash
SGLANG_USE_AITER=1 /tmp/amdpilot-repo-j-6f40a8eaec02/venv/bin/python - <<'PY'
import importlib
for name in ('aiter', 'aiter.fused_moe', 'sglang.srt.layers.moe.topk',
             'sglang.srt.layers.layernorm', 'sglang.test.test_utils'):
    mod = importlib.import_module(name)
    print(f'IMPORT PASS {name}: {mod.__file__}')
import aiter
for attr in ('layernorm2d_fwd', 'rmsnorm2d_fwd', 'rmsnorm2d_fwd_with_add'):
    print(f'AITER SYMBOL PASS {attr}: {getattr(aiter, attr)!r}')
PY
```

Raw output (addresses omitted only from callable representations):

```text
[aiter] import [module_aiter_core] under /tmp/amdpilot-repo-j-6f40a8eaec02/cache/aiter/module_aiter_core.so
IMPORT PASS aiter: /sgl-workspace/aiter/aiter/__init__.py
IMPORT PASS aiter.fused_moe: /sgl-workspace/aiter/aiter/fused_moe.py
IMPORT PASS sglang.srt.layers.moe.topk: /job/repo/python/sglang/srt/layers/moe/topk.py
IMPORT PASS sglang.srt.layers.layernorm: /job/repo/python/sglang/srt/layers/layernorm.py
IMPORT PASS sglang.test.test_utils: /job/repo/python/sglang/test/test_utils.py
AITER SYMBOL PASS layernorm2d_fwd: <function wrapper_custom>
AITER SYMBOL PASS rmsnorm2d_fwd: <function wrapper_custom>
AITER SYMBOL PASS rmsnorm2d_fwd_with_add: <function wrapper_custom>
```

## Existing test collection

Command, with all cache variables pointing outside `/job/repo`:

```bash
SGLANG_USE_AITER=1 \
TORCH_EXTENSIONS_DIR=/tmp/amdpilot-repo-j-6f40a8eaec02/cache/torch_extensions \
TRITON_CACHE_DIR=/tmp/amdpilot-repo-j-6f40a8eaec02/cache/triton \
XDG_CACHE_HOME=/tmp/amdpilot-repo-j-6f40a8eaec02/cache/xdg \
timeout 180s /tmp/amdpilot-repo-j-6f40a8eaec02/venv/bin/python \
  -m pytest --collect-only -q test/manual/layers/test_layernorm.py
```

Raw result:

```text
manual/layers/test_layernorm.py::TestRMSNorm::test_rms_norm
manual/layers/test_layernorm.py::TestGemmaRMSNorm::test_gemma_rms_norm
manual/layers/test_layernorm.py::TestGemma3RMSNorm::test_gemma3_rms_norm_2d
manual/layers/test_layernorm.py::TestGemma3RMSNorm::test_gemma3_rms_norm_3d
manual/layers/test_layernorm.py::TestGemma3RMSNorm::test_gemma3_rms_norm_3d_unflatten
manual/layers/test_layernorm.py::TestGemma3RMSNorm::test_gemma3_rms_norm_4d
manual/layers/test_layernorm.py::TestGemma3RMSNorm::test_gemma3_rms_norm_4d_unflatten
manual/layers/test_layernorm.py::TestLayerNorm::test_layer_norm
8 tests collected in 11.74s
COLLECTION_EXIT=0
```

The only collection diagnostics were an unrelated unknown `asyncio_mode`
pytest configuration warning and dependency deprecation warnings.

## Bounded MI350X RMSNorm execution

To select a small subset without modifying the existing test, the command
loaded `test/manual/layers/test_layernorm.py` and invoked its existing
`TestRMSNorm._run_rms_norm_test` helper for two representative cases. It was
bounded by `timeout 300s` and used the same external cache variables shown
above.

```bash
timeout 300s /tmp/amdpilot-repo-j-6f40a8eaec02/venv/bin/python - <<'PY'
import importlib.util, time, torch
path = 'test/manual/layers/test_layernorm.py'
spec = importlib.util.spec_from_file_location('qualified_test_layernorm', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = mod.TestRMSNorm(methodName='test_rms_norm')
mod.TestRMSNorm.setUpClass()
for params in ((7, 128, False, torch.float16, 0),
               (7, 128, True, torch.bfloat16, 0)):
    case._run_rms_norm_test(*params)
    torch.cuda.synchronize()
    print('PASS', params)
PY
```

Raw result:

```text
[aiter] import [module_aiter_core] under /tmp/amdpilot-repo-j-6f40a8eaec02/cache/aiter/module_aiter_core.so
[aiter] import [module_rmsnorm_quant] under /tmp/amdpilot-repo-j-6f40a8eaec02/cache/aiter/module_rmsnorm_quant.so
test_source=test/manual/layers/test_layernorm.py
layernorm_source=/job/repo/python/sglang/srt/layers/layernorm.py
_use_aiter=True _has_aiter_layer_norm=True _has_vllm_rms_norm=True
device=AMD Instinct MI350X arch=gfx950:sramecc+:xnack-
PASS TestRMSNorm._run_rms_norm_test(7, 128, False, torch.float16, 0) elapsed_s=0.950
PASS TestRMSNorm._run_rms_norm_test(7, 128, True, torch.bfloat16, 0) elapsed_s=0.002
PASS selected_cases=2 total_elapsed_s=0.952
RMSNORM_EXIT=0
```

AITER also warned that host NUMA balancing is enabled. This did not prevent
imports, collection, native module loading, or the selected numerical checks;
it was not changed because qualification was required as the ordinary UID.
