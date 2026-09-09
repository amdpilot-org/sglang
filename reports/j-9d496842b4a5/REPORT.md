# int4_w4a16 MoE tuner/runtime key investigation

## Status

This is an investigation report, not a production-code fix. The current mirror source reproduces the reported filename mismatch on one assigned AMD Instinct MI300X (`gfx942`). Open upstream change sgl-project/sglang pull request 38275, tested at commit `429350e4d9cdd975d4de246cdd253e2d99603c2d`, makes the tuner and runtime keys agree in the bounded probe. It is not merged here and this report does not claim that the open upstream change is complete or universally correct.

Reference context:

- sgl-project/sglang issue 35252: tuner writes a filename the runtime never reads for `int4_w4a16`.
- amdpilot-org/sglang issue 69: this bounded mirror investigation.
- sgl-project/sglang pull request 38275: removes the extra int4-only halving and adds a CPU filename regression.

## Environment

- Required qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`.
- Operator-provided local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. Container hostname is not image identity.
- GPU: one assigned AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, card model `0x74a1`, node ID 4, GUID 61795.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime `7.2.26015`.
- Triton: `3.7.0`; `sglang-kernel` `0.4.6.post1`.
- Working mirror base: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` (`main`).
- Tested upstream candidate: `429350e4d9cdd975d4de246cdd253e2d99603c2d`.
- Working Python source paths:
  - current: `/job/workdir/sglang/python/sglang/...`
  - candidate: `/job/workdir/sglang-candidate/python/sglang/...`
- Installed environment-context source: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Native modules used by the probe:
  - `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
  - `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

## Fixture and bounds

The probe used a job-private synthetic Mixtral config, not model weights:

- Experts: 8; top-k: 2.
- Hidden size: 128; intermediate size: 128.
- `shard_intermediate_size`: 256; runtime down-projection `N`: 128.
- Activation dtype: `bfloat16`; int4 group size: 128.
- Token count: 3.
- Search space: exactly one Triton config, so this was not a sweep.

The one searched config was:

```json
{
  "BLOCK_SIZE_M": 32,
  "BLOCK_SIZE_N": 16,
  "BLOCK_SIZE_K": 32,
  "GROUP_SIZE_M": 1,
  "num_warps": 1,
  "num_stages": 2,
  "waves_per_eu": 0
}
```

## Current source result

At mirror `main` commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`:

- Tuner-written filename: `E=8,N=64,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json`
- Runtime-read filename: `E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json`
- Agreement: false.

Before the tuner JSON was manually copied to the runtime key, `get_moe_configs(...)` returned no file and the selected default config was:

```json
{
  "BLOCK_SIZE_M": 16,
  "BLOCK_SIZE_N": 32,
  "BLOCK_SIZE_K": 64,
  "GROUP_SIZE_M": 1
}
```

After copying the tuner JSON to the runtime-read key, the selected up-projection and reused down-projection config was the one searched config:

```json
{
  "BLOCK_SIZE_M": 32,
  "BLOCK_SIZE_N": 16,
  "BLOCK_SIZE_K": 32,
  "GROUP_SIZE_M": 1,
  "num_warps": 1,
  "num_stages": 2,
  "waves_per_eu": 0
}
```

GPU output versus an independent local dequantized reference:

- GPU output L2 norm: `0.07435648888349533`
- Reference L2 norm: `0.07439721375703812`
- Maximum absolute error: `6.103515625e-05`
- Mean absolute error: `1.0615835890348535e-05`
- Relative L2 error: `0.00443923150804798`
- Unchanged gate: `torch.testing.assert_close(gpu_output, reference, atol=2e-2, rtol=0)`
- Gate result: passed.

## Candidate result

At upstream candidate commit `429350e4d9cdd975d4de246cdd253e2d99603c2d`:

- Tuner-written filename: `E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json`
- Runtime-read filename: `E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json`
- Agreement: true.
- Selected config after the tuner file was placed at the runtime key: the same one-config JSON shown above.
- GPU/reference metrics were identical to the current-source probe: max absolute error `6.103515625e-05`, relative L2 error `0.00443923150804798`, unchanged gate passed.

Focused candidate regression:

```text
/opt/venv/bin/python -m pytest test/registered/unit/layers/moe/test_fused_moe_triton_config.py -q
3 passed, 1 warning in 8.24s
```

## Commands and environment blocker

The mirror was cloned over HTTPS with up to three bounded attempts. Issue and pull-request context were read with `gh issue view` and `gh pr view`; nothing was posted or changed upstream.

The tuner CLI was attempted with one batch size and the one-config search-space file:

```text
/opt/venv/bin/python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --model /job/workdir/artifacts/synthetic_model \
  --dtype int4_w4a16 --tp-size 1 --batch-size 3 \
  --tune --search-space-file /job/workdir/artifacts/search_space.json
```

It could not start because the qualified image has no `ray` module:

```text
ModuleNotFoundError: No module named 'ray'
```

To avoid installing another framework stack, the probe imported the real tuner module with a minimal Ray stub and exercised its actual `get_config_filename`, `benchmark_config`, and `save_configs` functions. It also called the real runtime `get_config_file_name`, `get_moe_configs`, `try_get_optimal_moe_config`, and `fused_moe` paths. The reproduction command is:

```text
PYTHONPATH=/job/workdir/sglang/python \
SGLANG_MOE_CONFIG_DIR=/job/workdir/moe_configs/current_fresh4 \
TRITON_CACHE_DIR=/job/workdir/.triton_cache \
HF_HUB_OFFLINE=1 \
/opt/venv/bin/python /job/workdir/artifacts/int4_probe.py \
  --repo /job/workdir/sglang \
  --model /job/workdir/artifacts/synthetic_model \
  --server-model /job/workdir/artifacts/server_model \
  --search-space /job/workdir/artifacts/search_space.json \
  --output-dir /job/workdir/artifacts/current_fresh4 \
  --config-dir /job/workdir/moe_configs/current_fresh4 \
  --label current_fresh4
```

The candidate command is identical except for `--repo /job/workdir/sglang-candidate` and fresh output/config directories.

## Interpretation

- The current tuner derives `N = shard_intermediate_size // 2`, then applies an additional `N // 2` only for `int4_w4a16`.
- The runtime derives `E, _, N = w2_shape` and has no int4-only halving.
- For this fixture, that changes the key from `N=128` to `N=64`, exactly reproducing the reported factor-of-two disagreement.
- The candidate removes only the int4-only halving and makes the keys agree; no unrelated format was changed by this investigation.
- Numerical output remains within the pre-existing unchanged gate in both cases, so this is a configuration-discovery issue, not an observed int4 numerical-output regression.

## Left undone / uncertainty

- The full tuner CLI was not run because `ray` is absent; no replacement framework was installed.
- No full tuning sweep, model download, server launch, or full-model end-to-end run was performed.
- Only the upstream fix commit `429350e4d9cdd975d4de246cdd253e2d99603c2d` was tested; the later merge commit listed on the open pull request was not separately rerun.
- The probe uses a synthetic local Mixtral config and direct fused-MoE tensors, not a complete quantized checkpoint.
