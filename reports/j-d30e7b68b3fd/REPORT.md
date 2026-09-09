# store_cache_4d gfx942 validation report

## Outcome

The `AMDGPUCanonicalizePointers` / `PassManager::run failed` symptom did **not**
reproduce on the assigned MI300X. The current `store_cache_4d` path compiled and
matched an independent Torch advanced-indexing reference for contiguous and
supported outer-strided 4D layouts. No lowering or wrapper correction was needed.

This is an already-fixed/negative result, not a new fix. In particular, the
unmerged pointer-branch workaround in upstream PR 30391 is unnecessary because
the unmodified current source in upstream PR 36753 also compiles and passes.

Current mirror `main` (`ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`) no longer
contains `python/sglang/kernels/ops/kvcache/cache_move.py` or
`test/registered/unit/mem_cache/test_store_cache_4d.py`. Upstream commit
`4bea51d885538466caef09223e8beb1c307b4489` removed both while replacing the
page-major envelope path. Therefore this branch carries only the validation
report and reproduction artifact.

## Bounded candidate tests

- Upstream PR 36753, commit `ecd80e19b0d83ea4edf88bac7553de090c0e6b48`
  (`[AMD] Register test_store_cache_4d for AMD 1-GPU PR CI`): **PASS**, 10/10
  tests in 32.042 seconds. This commit only adds AMD CI registration; its
  source retains the shared `tl.load`/`tl.store` outside the `pid_kv` branches.
- Upstream PR 30391, commit `afd12e0369fdcf4b307b22935177f69148dfc4cc`
  (`Avoid ROCm pointer merge in store_cache_4d`): **PASS**, 12/12 tests in
  15.742 seconds. This closed, unmerged PR moves load/store into each branch.
  It is not adopted because the unmodified current source already passes.

The relevant source and test paths in the PR 36753 worktree were:

- `/job/sglang/.worktrees/pr36753/python/sglang/kernels/ops/kvcache/cache_move.py`
- `/job/sglang/.worktrees/pr36753/test/registered/unit/mem_cache/test_store_cache_4d.py`

The PR 30391 worktree used the pre-migration paths:

- `/job/sglang/.worktrees/pr30391/python/sglang/srt/mem_cache/triton_ops/cache_move.py`
- `/job/sglang/.worktrees/pr30391/test/registered/unit/mem_cache/test_store_cache_4d.py`

## Environment

- Campaign: `repo-e2e-20260909`
- Coordination tracker: `amdpilot-org/amdpilotv2` issue 402
- GPU: one assigned AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- `rocminfo` UUID: `GPU-b5c590cf4c10631d`
- `amd-smi monitor -g 0 --json`: GPU 0, 192.0 GiB VRAM total
- Image: operator-specified `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`,
  local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`,
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Triton: `3.7.0`,
  `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Torch HIP library:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Native dependencies: `/opt/rocm/lib/libamdhip64.so.7`,
  `/opt/rocm/lib/libhsa-runtime64.so.1`
- Triton AMD backend: `/opt/venv/lib/python3.10/site-packages/triton/backends/amd`
- Installed source context without `PYTHONPATH`:
  `/sgl-workspace/sglang/python/sglang/__init__.py`

The container has no Docker/Podman/ctr/crictl image-inspection client, so the
image ID above is the operator-provided local image identity rather than a
runtime-inspected digest.

## Numerical gates

The historical test suite uses `torch.equal` as a byte-identity gate against the
legacy advanced-indexing path. It covers bf16 and fp8_e5m2, int32 and int64
locations, empty input, page sizes 1 and greater than 1, asymmetric K/V head
dimensions, stride assertions, and two `set_kv_buffer` integration cases.

The focused synthetic check additionally requires all of:

- written K slots equal the independent reference (`torch.equal`)
- written V slots equal the independent reference (`torch.equal`)
- unchanged K and V neighboring slots equal the reference (`torch.equal`)
- whole K and V storage equal the reference (`torch.equal`)
- source `cache_k` and `cache_v` remain unchanged
- maximum absolute difference is exactly `0.0`

Synthetic sizes:

- contiguous K: `(17, 8, 16, 128)`, stride `(16384, 2048, 128, 1)`, 557056 bytes
- contiguous V: `(17, 8, 16, 96)`, stride `(12288, 1536, 96, 1)`, 417792 bytes
- strided K view: `(17, 8, 16, 128)`, stride `(65536, 4096, 128, 1)`,
  2228224-byte backing storage
- strided V view: `(17, 8, 16, 96)`, stride `(49152, 3072, 96, 1)`,
  1671168-byte backing storage
- 64 unique int64 locations, bf16 data, page size 8

All focused gates were true and both maximum absolute differences were `0.0`.
Raw outputs are retained under `reports/j-d30e7b68b3fd/raw/`.

## Reproduction

The task clone was created first with a bounded depth-50 clone:

```bash
git clone --depth 50 https://github.com/amdpilot-org/sglang.git /job/sglang
```

Fetch only the two bounded candidate heads from the public upstream repository:

```bash
cd /job/sglang
git fetch --no-tags --depth 50 https://github.com/sgl-project/sglang.git \
  ecd80e19b0d83ea4edf88bac7553de090c0e6b48
git fetch --no-tags --depth 50 https://github.com/sgl-project/sglang.git \
  afd12e0369fdcf4b307b22935177f69148dfc4cc
git worktree add --detach .worktrees/pr36753 \
  ecd80e19b0d83ea4edf88bac7553de090c0e6b48
git worktree add --detach .worktrees/pr30391 \
  afd12e0369fdcf4b307b22935177f69148dfc4cc
```

Run the historical suite:

```bash
cd /job/sglang/.worktrees/pr36753
export PYTHONPATH=/job/sglang/.worktrees/pr36753/python
export TRITON_CACHE_DIR=/job/.cache/triton-pr36753
export XDG_CACHE_HOME=/job/.cache/xdg-pr36753
/opt/venv/bin/python test/registered/unit/mem_cache/test_store_cache_4d.py -v
```

Run the focused synthetic check:

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/.worktrees/pr36753/python
export TRITON_CACHE_DIR=/job/.cache/triton-pr36753
export XDG_CACHE_HOME=/job/.cache/xdg-pr36753
/opt/venv/bin/python reports/j-d30e7b68b3fd/synthetic_store_cache_4d_check.py
```

No model weights were downloaded. Triton and XDG caches were kept under
`/job/.cache`, outside the delivery clone.

## Uncertainty and left undone

- The exact upstream commit or toolchain change that eliminated the original
  compiler failure was not isolated. The issue's 2026-08-27 rerun and both
  candidate heads pass on this ROCm 7.2/Triton 3.7 stack.
- Upstream PR 36753 was still open when checked. Its CI-only registration is
  not duplicated here because current mirror `main` removed the test and source
  path on 2026-08-30.
- No unsupported layout was made to work: the wrapper intentionally accepts only
  trailing-dim-contiguous views. The focused strided case varies outer page and
  token strides while preserving the supported trailing strides.
