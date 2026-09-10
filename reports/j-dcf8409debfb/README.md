# AITER allreduce-fusion log resolution report

Campaign: `repo-e2e-20260909`
Task ID: `j-dcf8409debfb`
Date: 2026-09-10

## Conclusion

The default DeepSeek/GLM AITER allreduce-fusion auto-enable remains intentionally disabled. The final change does not restore it. The enabled log now fires only when the **resolved** `enable_aiter_allreduce_fusion` value is true, covering both an explicit raw input and a model-override declaration.

The one-rank GPU linear-plus-RMSNorm control is bit-identical before and after the change. Multi-GPU fusion performance and correctness were not tested and are outside this task's one-GPU scope.

## Environment

- Assigned image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Assigned local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- The container does not expose Docker/containerd metadata, so the image ID above is the operator-provided local identity. The hostname is not used as image identity.
- Hostname: `banff-cyxtera-cx57-5`
- GPU: one assigned AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, serial `692440004373`, GUID `19304`
- Python: `/opt/venv/bin/python`
- Delivery source: `/job/sglang/python/sglang`
- Preinstalled source context: `/sgl-workspace/sglang/python/sglang`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`, version `2.9.1+rocm7.2.0.git7e1940d4`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- AITER Python package: `/sgl-workspace/aiter/aiter`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- ROCm: `/opt/rocm-7.2.0`, HIP runtime `7.2.26015-fc0010cf6a`
- Job-private cache: `/tmp/sglang-cache-j-dcf8409debfb`

## Context and commits

- Upstream issue read: sgl-project/sglang issue 38710. It had no comments at the time of reading.
- Upstream candidate read and tested: sgl-project/sglang pull request 38733, commit `207c50bece1f126bfb97fc594ac2ea150d8c184e`.
- Delivery base: `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- The tested upstream candidate was preserved as a cherry-pick on the delivery branch. The follow-up changes its raw-field check to the resolved view and adds regression coverage.
- No upstream issue, pull request, or comment was posted or modified.

## Reproduction

The hook exercise used `/opt/venv/bin/python` with `PYTHONPATH=/job/sglang/python` (or the candidate worktree path) and `SGLANG_CACHE_HOME=/tmp/sglang-cache-j-dcf8409debfb`. It constructed `ServerArgs(model_path="dummy")`, supplied a fake HF config for `DeepseekV3ForCausalLM`, `GlmMoeDsaForCausalLM`, or `GptOssForCausalLM`, called `handle_model_specific_adjustments`, captured the model-hook logger, and read `resolved_view(server_args).enable_aiter_allreduce_fusion`.

### Base result

```text
CASE DeepseekV3ForCausalLM explicit=False raw False resolved False enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE DeepseekV3ForCausalLM explicit=True raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE GlmMoeDsaForCausalLM explicit=False raw False resolved False enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE GlmMoeDsaForCausalLM explicit=True raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
```

This reproduces issue 38710: the default case logs enabled while the resolved flag is false.

### Candidate result

Tested upstream commit `207c50bece1f126bfb97fc594ac2ea150d8c184e`:

```text
CASE DeepseekV3ForCausalLM explicit False declared False raw False resolved False enabled_messages []
CASE DeepseekV3ForCausalLM explicit True declared False raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE GlmMoeDsaForCausalLM explicit False declared False raw False resolved False enabled_messages []
CASE GlmMoeDsaForCausalLM explicit True declared False raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE DeepseekV3ForCausalLM explicit False declared True raw False resolved True enabled_messages []
```

The candidate fixes explicit/default observability, but its raw-field check misses a declaration-resolved true value. The final follow-up therefore reads `resolved_view(server_args)`.

### Final result

```text
CASE DeepseekV3ForCausalLM explicit False declared False raw False resolved False enabled_messages []
CASE DeepseekV3ForCausalLM explicit True declared False raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE GlmMoeDsaForCausalLM explicit False declared False raw False resolved False enabled_messages []
CASE GlmMoeDsaForCausalLM explicit True declared False raw True resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
CASE DeepseekV3ForCausalLM explicit False declared True raw False resolved True enabled_messages ['Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM']
```

## Validation

Focused unit command:

```bash
PYTHONPATH=/job/sglang/python SGLANG_CACHE_HOME=/tmp/sglang-cache-j-dcf8409debfb \
  /opt/venv/bin/python -m pytest \
  test/registered/unit/server_args/test_aiter_allreduce_fusion_log.py -q
```

Raw result:

```text
2 passed, 3 warnings, 6 subtests passed in 12.35s
```

The warnings are pre-existing pytest configuration and Cython deprecation warnings; they are unrelated to this change.

### One-rank numerical regression control

Label: `one-rank-linear-rmsnorm-control`

The control used one MI300X, seed `20260910`, a `torch.nn.Linear(128, 128)` in bfloat16, and SGLang `RMSNorm(128)`. It compared the base commit and the fixed working tree. It also compared the AITER-selected norm with a native RMSNorm reference.

```text
label=one-rank-linear-rmsnorm-control
shape (8, 128) dtype torch.bfloat16
output_sha256 f82ac97590142527882690fce6849093a27cf0d00ae02b6acd7a7b34a5c567b4
native_reference_sha256 f82ac97590142527882690fce6849093a27cf0d00ae02b6acd7a7b34a5c567b4
max_abs_diff_vs_native 0.0
allclose_vs_native True
```

Base and fixed output hashes are equal. The native-reference hash and maximum absolute difference are also equal. The numerical gate is unchanged.

## Scope and uncertainty

- The image ID could not be independently queried from inside the container because no Docker/containerd client or metadata mount was available. The operator-provided local image ID is recorded above.
- The GLM hook still uses the legacy message text naming `DeepseekV3ForCausalLM`; this task changes activation observability only and does not alter model routing or execution.
- Multi-GPU allreduce-fusion performance and correctness remain untested and outside the one-GPU scope.
