# EP scatter kernel investigation: j-5e16009dc20c

## Scope

This records local `gfx942` behavior for `_fwd_kernel_ep_scatter_1`. It is not a
distributed expert-parallel proof, does not exercise DeepEP transport, and does
not use model weights.

Upstream context is sgl-project/sglang issue 31929 and candidate pull request
31930. The candidate head commit is
`3612e7a1ea60324e9425e7a3d41ed9f2414cad78`. No upstream issue, pull request, or
comment was posted or modified.

## Environment

- Required local image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Container marker: `/.dockerenv` exists. No Docker/CRI utility is available inside the job, so the image ID could not be independently resolved from container metadata.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 compute units.
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton: `3.7.0+amd.rocm7.2.0.git89002410`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Job-private Triton caches: `/tmp/sglang-cache-j-5e16009dc20c*`

## Sources

- Installed-source baseline: `/sgl-workspace/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py`
- Installed-source repository commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed package version string: `0.5.18.dev20260826+g937af8538b`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Delivery source: `/job/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py`

The installed-source baseline is evidence only for that installed source. It is
not proof for later checkout changes.

## First GPU baseline

The first valid installed-source case used four experts with padded counts
`[128, 256, 0, 128]`, valid counts `[100, 200, 0, 128]`, `BLOCK_E=128`,
`num_warps=8`, and a 512-entry
`m_indices` buffer.

- Exclusive starts matched `torch.cumsum`.
- Expert IDs and `-1` padding matched the independent reference.
- First successful GPU execution wall time: `0.8750913273543119` seconds.
- Mean CUDA-event time: `0.018389066060384113` ms over 30 launches after three warmups.
- Full record: `/job/baseline-first.json`.

The first attempt launched the GPU kernel successfully but failed while building
the Python reference because `torch.cat` was given a generator instead of a
tuple. The successful neighboring control used `tuple(generator)` and the same GPU
case. The concrete error is recorded in `/job/baseline-first.json`.

## Boundary cases

The test compares:

- exclusive starts from `torch.cumsum`,
- expert IDs written by `torch.Tensor.scatter_`,
- `-1` entries for valid-token tails,
- `-7` guard storage before and after the `m_indices` view.

`expert_start_loc` is initialized to `-1`. This makes an unsafe stale read
deterministic and directs a stale offset into the left guard rather than into an
unrelated allocation. The guard size is the largest padded expert count.

| Case | Padded counts | Valid counts | Unfixed starts | Unfixed indices/guards | Fixed starts | Fixed indices/guards |
|---:|---|---|---|---|---|---|
| 0 | `[128]` | `[0]` | pass | pass | pass | pass |
| 1 | `[128, 128, 128]` | `[0, 128, 127]` | pass | pass | pass | pass |
| 2 | `[128, 256, 0, 128]` | `[100, 200, 0, 128]` | pass | pass | pass | pass |
| 3 | `[0, 128, 128, 0, 128]` | `[0, 128, 128, 0, 127]` | pass | pass | pass | pass |

All comparisons use exact integer equality. Timing uses the mean CUDA-event
elapsed time over 30 launches after three warmup launches and a synchronize.

| Case | Unfixed mean ms | Fixed mean ms |
|---:|---:|---:|
| 0 | 0.03650423288345337 | 0.01840513348579407 |
| 1 | 0.03490589857101441 | 0.018069700400034586 |
| 2 | 0.035671667257944746 | 0.01842116713523865 |
| 3 | 0.03527073462804158 | 0.018434532483418784 |

Raw records are in `/job/installed-unfixed-gfx942-results.json` and
`/job/mirror-fixed-gfx942-results.json`.

## Architecture-specific result and limitation

On this `gfx942` stack, the unfixed installed kernel passed all bounded cases and
did not reproduce the illegal access reported on the NVIDIA system in issue
31929. Nevertheless, generated AMDGCN for the unfixed kernel contains a
`global_store_dword` for `expert_start_loc` followed by a `global_load_dword`
from that same buffer, with no barrier between them. The fixed kernel derives
`cur_expert_start` by reducing `tokens_per_expert` in registers and does not
reload the just-written global value.

Therefore, the local result establishes a latent hazard on `gfx942`, not absence
of a hazard. It does not prove cross-architecture behavior, distributed EP
correctness, DeepEP transport behavior, or long-running production stability.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-5e16009dc20c
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/moe/test_ep_moe_kernels.py
```

The final run reported `4 passed` in `5.97` seconds on the assigned MI300X. The
only warning was pytest's existing `Unknown config option: asyncio_mode`.
