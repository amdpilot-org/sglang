# gfx942 xAI temperature scaling investigation

## Scope

- Upstream context: sgl-project/sglang issue 32942 and pull request 32943.
- Tested upstream candidate commit: `79dfba5440fc1a1f33088d30ba4d8907f9fd36af`.
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 206,141,652,992 bytes.
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`; Torch `2.9.1+rocm7.2.0.git7e1940d4`; HIP `7.2.26015-fc0010cf6a`; Triton `3.7.0+amd.rocm7.2.0.git89002410`.

## Installed-source baseline

The installed source was `/sgl-workspace/sglang` at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. Its Python module was
`/sgl-workspace/sglang/python/sglang/__init__.py`, and the attention implementation was
`/sgl-workspace/sglang/python/sglang/kernels/ops/attention/extend_attention.py`.
The installed `sgl_kernel` Python module was
`/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`; importing its FA3
attention path failed because `flash_ops` was unavailable. Test initialization loaded
the native Aiter module `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

The first successful GPU baseline was:

```bash
cd /sgl-workspace/sglang
PYTHONPATH=/sgl-workspace/sglang/python \
PYTEST_ADDOPTS='-p no:cacheprovider' \
/opt/venv/bin/python -m pytest -q \
'test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention' -s
```

Result: `1 passed, 3 warnings` in 39 seconds, measured with shell epoch time around
the complete pytest process. The test dispatched Triton `extend_attention_fwd` and
compared against `redundant_attention` using `torch.allclose(rtol=1e-2, atol=1e-3)`.
This installed-source result is not evidence for later checkout changes.

## Synthetic comparison

The focused case used two sequences with prefix lengths `[7, 11]`, extend lengths
`[9, 13]`, 8 query heads, 2 KV heads, head dimension 64, bfloat16 QKV, and
`xai_temperature_len=4`. It dispatched Triton `extend_attention_fwd` for the regular
path and Triton `extend_attention_fwd_unified` for the deterministic path. The
independent reference computed float32 Torch einsum logits and applied
`log2(position) / log2(4)` only when `position > 4`.

Raw maximum absolute differences:

| Comparison | Mirror main | Upstream candidate / fixed working tree |
| --- | ---: | ---: |
| Regular vs Torch reference | 0.00781702995300293 | 0.00781702995300293 |
| Deterministic vs Torch reference | 2.6570067405700684 | 0.00781702995300293 |
| Deterministic vs regular | 2.6572265625 | 0.0078125 |

All three comparisons passed `torch.allclose(rtol=2e-2, atol=2e-2)` after the fix;
before the fix, only the regular path passed. Timing used `time.perf_counter` around
one call plus `torch.cuda.synchronize`; the first call includes Triton JIT. The fixed
probe measured 0.8233433188870549 seconds for regular and
0.004651139490306377 seconds for deterministic. These are bounded one-call
measurements, not performance benchmarks.

## Validation

```bash
PYTHONPATH=/job/sglang/python \
PYTEST_ADDOPTS='-p no:cacheprovider' \
/opt/venv/bin/python -m pytest -q \
'test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_xai_temperature_reference' -s
```

Result: `1 passed, 3 warnings` in 24 seconds after formatting.

```bash
PRE_COMMIT_HOME=/tmp/sglang-cache-j-934041f0d312/pre-commit \
pre-commit run --files \
python/sglang/kernels/ops/attention/extend_attention.py \
test/registered/attention/test_triton_attention_kernels.py
```

Result: all applicable hooks passed after one automatic Ruff format pass.

The unchanged neighboring `test_extend_attention_unified_vs_regular` still fails on
gfx942 in its `B=8, N_CTX=256, H_Q=64, H_KV=8, D=80` subtest with maximum absolute
difference `0.1669921875`. The local environment does not set the AMD CI flag, so the
test uses non-CI tolerance `rtol=0.15, atol=0.15`; the CI-specific tolerance is
`atol=0.17`. This is a pre-existing architecture-specific tolerance limitation, not a
result of the xAI temperature change.
