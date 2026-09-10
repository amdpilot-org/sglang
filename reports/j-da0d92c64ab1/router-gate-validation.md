# Router GEMM and model gate validation

## Scope

This investigation follows sgl-project/sglang issue 38695 without replacing the
model-specific gates with a broad `RouterGate` architecture. Upstream pull
request 38719 already implements the focused six-gate fix, so its head commit
was tested unchanged rather than duplicated:

- Upstream candidate: `664767428904e6f12ce8869a783335fed2e7d439`
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Candidate changes: preserve FP32 correction bias in MiMo-V2, Ernie4, and
  Kimi Linear; preserve router-weight dtype in Bailing MoE, Bailing MoE Linear,
  and LLaDA2.

No production numerical threshold was changed. The new GPU test uses
`atol=1e-6`, `rtol=1e-6`, and a `1e-5` near-tie separation.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Container hostname: `banff-cyxtera-cx57-4`
- GPU: one AMD Instinct MI300X, gfx942, serial `692440003936`, unique ID
  `0x7ecf53b7cc7c10da`, driver `6.19.14.31400000`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang`
- Installed SGLang source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed `sgl_kernel` Python: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Installed `sgl_kernel` native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

The installed source is environment context only. It is a different revision
from the mirror base and is not evidence for checkout changes.

## Installed-source baseline

The first GPU execution used the installed AITER router helper:

```bash
/opt/venv/bin/python /tmp/router_gemm_baseline.py
```

The helper is `sglang.srt.layers.rocm_linear_utils.aiter_dsv3_router_gemm`. It
passes `otype=hidden_states.dtype`, so BF16 inputs return BF16 logits on this
stack. The independent reference was
`hidden_states.float() @ router_weights.float().T`.

| Tokens | Output dtype | Max abs error vs FP32 | CUDA-event mean |
|---:|---|---:|---:|
| 1 | BF16 | 0.4920806884765625 | 40.9403 us |
| 8 | BF16 | 0.5875244140625 | 28.1789 us |
| 16 | BF16 | 0.9566650390625 | 28.0988 us |

The first GPU execution took 0.21930287592113018 seconds. Timing used
`torch.cuda.Event` around 20 calls after 5 warmups. The complete baseline is
in job-level `baseline-first.json`.

The installed JIT `dsv3_router_gemm` test and benchmark skip HIP because the
kernel is SM90+ CUDA-only. gfx942 therefore used the supported AITER/torch
neighboring control above.

## GPU gate comparison

The same synthetic BF16 hidden states and FP32 weights were used for all three
affected wrappers. The FP32 reference rows were `1.0000097751617432` and `1.0`;
BF16 rounds both to `1.0`. The FP32 top-2 order is `[0, 1]`, while the BF16
round-trip order is `[1, 0]`.

```bash
PYTHONPATH=/tmp/sglang-base-j-da0d92c64ab1/python \
  /opt/venv/bin/python /tmp/compare_router_gates.py

PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python /tmp/compare_router_gates.py
```

### Mirror base `0084030179bfba86bfeb6d43f7997d4076329d2c`

| Gate | Output dtype | Max abs error | Top-2 | Matches FP32 |
|---|---|---:|---|---|
| Bailing MoE | BF16 | 9.775161743164062e-06 | `[1, 0]` | No |
| Bailing MoE Linear | BF16 | 9.775161743164062e-06 | `[1, 0]` | No |
| LLaDA2 | BF16 | 9.775161743164062e-06 | `[1, 0]` | No |

### Candidate `664767428904e6f12ce8869a783335fed2e7d439`

| Gate | Output dtype | Max abs error | Top-2 | Matches FP32 |
|---|---|---:|---|---|
| Bailing MoE | FP32 | 0.0 | `[0, 1]` | Yes |
| Bailing MoE Linear | FP32 | 0.0 | `[0, 1]` | Yes |
| LLaDA2 | FP32 | 0.0 | `[0, 1]` | Yes |

## Validation commands

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/models/test_moe_router_fp32_contract.py \
  test/registered/kernel/models/test_moe_router_fp32_contract_gpu.py

/opt/venv/bin/python -m compileall -q \
  python/sglang/srt/models/bailing_moe.py \
  python/sglang/srt/models/bailing_moe_linear.py \
  python/sglang/srt/models/ernie4.py \
  python/sglang/srt/models/kimi_linear.py \
  python/sglang/srt/models/llada2.py \
  python/sglang/srt/models/mimo_v2.py \
  test/registered/unit/models/test_moe_router_fp32_contract.py \
  test/registered/kernel/models/test_moe_router_fp32_contract_gpu.py

PRE_COMMIT_HOME=/tmp/sglang-cache-j-da0d92c64ab1/pre-commit \
XDG_CACHE_HOME=/tmp/sglang-cache-j-da0d92c64ab1/xdg \
pre-commit run --files <changed files>
```

Results:

- Focused CPU and GPU tests: 6 passed, 9 subtests passed.
- Bytecode compilation: passed.
- `git diff --check`: passed.
- Pre-commit on all changed files: passed.

## Limitations

- This validates the model-local gate contracts only; it does not establish a
  shared `RouterGate` architecture or change deterministic dispatch.
- The SM90+ JIT router GEMM is not supported on gfx942.
- No full model weights were downloaded and no broad model accuracy run was
  performed.
- The installed AITER baseline is not proof for the later mirror checkout.
