# gfx942 attention backend and KV-cache dtype admission investigation

## Scope

This is a bounded, one-GPU investigation of the current `amdpilot-org/sglang` source and one upstream candidate. It does not redesign the backend registry, download model weights, or claim results for architectures other than the assigned MI300X.

- Working source: `/job/sglang`, commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Candidate: upstream pull request 32576, commit `0b4e0d5f719aaeed7752e90e168805ca49915d3c`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, compute capability `(9, 4)`, 191.98 GiB HBM
- Image: operator-specified `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`

The preinstalled source at `/sgl-workspace/sglang` was used only for environment context. It is commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` and has local modifications/untracked files; it is not the tested working source.

## Source and native paths

- Tested Python package: `/job/sglang/python/sglang/__init__.py`
- Installed context package: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`, version `0.4.6.post1`
- `sgl_kernel.flash_attn`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/flash_attn.py`
- Aiter Python package: `/sgl-workspace/aiter/aiter/__init__.py`
- Aiter core native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Aiter BF16 attention native module: `/sgl-workspace/aiter/aiter/jit/mha_fwd_bf16_nbias_mask_nlse_ndropout_nqscale.so`
- Aiter FP8 attention native module: `/sgl-workspace/aiter/aiter/jit/mha_fwd_fp8bf16_nbias_mask_nlse_ndropout_pertensor.so`

## Resolver observations

### Ordinary attention backends

For an ordinary MHA model with no explicit backend, `get_default_attn_backend` returns `aiter` on HIP at `python/sglang/srt/arg_groups/model_override_base.py:343`. For MLA, HIP returns `aiter` when the KV head count is 16 or 128 and otherwise `triton` at `python/sglang/srt/arg_groups/model_override_base.py:358`.

`handle_attention_backend_compatibility` is called by the resolution pipeline at `python/sglang/srt/arg_groups/pipeline.py:252`, before model loading. On gfx942:

- Default MHA BF16 and `fp8_e4m3` both resolve to `aiter`.
- Explicit `aiter` BF16 and `fp8_e4m3` are both admitted.
- Explicit `trtllm_mla` BF16 and `fp8_e4m3` are rejected early in `_mla_kv_cache_dtype_checks` at `python/sglang/srt/arg_groups/overrides.py:1299`.
- Explicit `tokenspeed_mla` BF16 and `fp8_e4m3` are rejected early at `python/sglang/srt/arg_groups/overrides.py:1311`.
- Explicit `cutedsl_mla` BF16 and `fp8_e4m3` are rejected early at `python/sglang/srt/arg_groups/overrides.py:1349`.
- Explicit `trtllm_mha` BF16 and `fp8_e4m3` are rejected early at `python/sglang/srt/arg_groups/attention_hook.py:135`.

The early diagnostics are specific. For example, `trtllm_mla` reports:

```text
TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend.
```

### Late `fa3` admission

Explicit `fa3` BF16 and `fp8_e4m3` are admitted by argument resolution on gfx942. The registry check at `python/sglang/srt/layers/attention/attention_registry.py:224` accepts major version 9; it does not reject HIP. MI300X reports `(9, 4)`, so the assertion passes.

The first observed failure in this image is the native import in `FlashAttentionBackend.__init__` at `python/sglang/srt/layers/attention/flashattention_backend.py:275`:

```text
ImportError: Can not import FA3 in sgl_kernel. Please check your installation.
```

This is not an early argument-resolution failure. `ModelRunner.initialize` loads the model at `python/sglang/srt/model_executor/model_runner.py:660` and resolves the KV dtype at `python/sglang/srt/model_executor/model_runner.py:691`. The scheduler then initializes attention backends at `python/sglang/srt/managers/scheduler.py:1106`, before CUDA graph capture at `python/sglang/srt/managers/scheduler.py:1107`. Thus the observed `fa3` failure lands after model loading and before graph capture in this environment.

### DSA split backends

`_dsa_split_backend_resolution` has these gfx942 results:

| KV dtype | Input prefill | Input decode | Resolved prefill | Resolved decode |
|---|---|---|---|---|
| BF16 | unset | unset | `tilelang` | `tilelang` |
| BF16 | `tilelang` | unset | `tilelang` | `fa3` |
| BF16 | `flashmla_sparse` | unset | `flashmla_sparse` | `fa3` |
| BF16 | unset | `tilelang` | `flashmla_sparse` | `tilelang` |
| FP8 E4M3 | unset | unset | `tilelang` | `tilelang` |
| FP8 E4M3 | `tilelang` | unset | `tilelang` | `flashmla_kv` |
| FP8 E4M3 | `flashmla_sparse` | unset | `flashmla_sparse` | `flashmla_kv` |
| FP8 E4M3 | unset | `tilelang` | `flashmla_kv` | `tilelang` |

The BF16 one-sided cases are admitted with decode `fa3`. Source inspection shows a further late-failure risk: on HIP, `python/sglang/srt/layers/attention/dsa_backend.py:147` imports Aiter symbols but does not define `flash_attn_with_kvcache`. `_forward_fa3` references that name at `python/sglang/srt/layers/attention/dsa_backend.py:2526`. After importing the module on gfx942, `flash_attn_with_kvcache_defined` was `false`. Therefore the first DSA `fa3` decode forward would fail with an undefined name before executing a kernel. This path was not exercised with a full DSA model.

No tilelang, `flashmla_sparse`, or `flashmla_kv` kernel claim is made here; only their argument-resolution results were tested.

## Synthetic operator validation

The controls used one MI300X, batch 1, query length 128, key length 128, 8 query heads, 1 KV head, and head dimension 128.

### BF16 control

Operator: `aiter.flash_attn_func`

- Output dtype: `torch.bfloat16`
- Maximum absolute difference from the FP32 reference: `0.0078125`
- Elapsed wall time after JIT build: `18.996201921254396 ms`

### FP8 E4M3 control

Operator: `aiter.flash_attn_fp8_pertensor_func`

- Storage dtype on HIP: `torch.float8_e4m3fnuz`
- Output dtype: `torch.bfloat16`
- Maximum absolute difference from the BF16 control: `0.03515625`
- Existing numerical gate: maximum absolute difference `< 0.055`
- Gate result: passed
- Elapsed wall time after JIT build: `13.200472109019756 ms`

The gate was not changed. An initial exploratory run used `torch.randn`; it produced maximum difference `0.15625` and truthfully failed the unchanged `0.055` gate. The recorded passing control uses the `torch.rand` distribution from Aiter's existing `test_mha_fp8.py`, with the same gate. Both raw results are retained in `logs/`.

## Candidate 32576

Upstream pull request 32576 was tested at preserved commit `0b4e0d5f719aaeed7752e90e168805ca49915d3c` in a detached worktree. Its parent is `8a311d1c889244ab1f857d7df79de7e5f0a6891c`.

Focused candidate tests:

```text
14 passed, 3 warnings in 11.79s
```

Direct gfx942 DSA facts from the candidate:

- BF16, both sides unset: `('tilelang', 'tilelang')`, admitted
- FP8 E4M3, both sides unset: `('tilelang', 'tilelang')`, admitted
- BF16, prefill set and decode unset: decode defaults to `fa3`, admitted
- FP8 E4M3, prefill set and decode unset: decode defaults to `flashmla_kv`, admitted

The candidate also still admits ordinary explicit `fa3` for BF16 and `fp8_e4m3` on gfx942. Therefore this candidate does not fix the observed late `fa3` admission issue, and no such fix claim is made. The candidate was not merged or cherry-picked into this report branch.

## Current-source focused tests

Command:

```sh
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/test_dsa_tilelang_fp8_validation.py \
  test/registered/unit/test_model_overrides.py \
  -k 'dsa_split_backend_resolution or tilelang_fp8'
```

Result:

```text
6 passed, 92 deselected, 3 warnings, 6 subtests passed in 11.57s
```

## Limits and unfinished work

- Only gfx942 was tested; no result is claimed for other architectures.
- No full model weights were downloaded and no end-to-end model load, DSA forward, or graph capture was run.
- The tilelang, `flashmla_sparse`, and `flashmla_kv` kernels were not numerically validated.
- The full repository test suite was not run.
- The image name and local image ID are recorded from the operator specification; hostname was not used as image identity.
- Candidate 32576 is an older upstream tree. Its focused tests pass in its detached worktree, but it was not validated as a clean apply to current `main`.
