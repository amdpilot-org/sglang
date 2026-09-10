# Bounded gfx942 cache-move compiled-path reuse report

## Scope

This follow-up covers the existing `copy_all_layer_kv_cache_tiled` execution
representation on one AMD Instinct MI300X. It checks cold versus warm Triton
compiled-path reuse over a finite supported `num_locs` sequence while preserving
the kernel's documented in-place aliasing, pointer, and dtype contracts.

It does not repeat the host-driven cycle-permutation probe from
amdpilot-org/sglang issue 210 (already covered by commit `8dea37e62`),
integrate a new compactor, or validate the throughput claims in
sgl-project/sglang issue 38357. At investigation time, issue 38357 had no
comments and no linked upstream fix was found.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`, Python 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0`
- GPU: one AMD Instinct MI300X, capability `(9, 4)`
- Actual Triton dispatch target: `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
- Persistent checkout: `/job/sglang`
- PR base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Changed source: `/job/sglang/python/sglang/kernels/ops/kvcache/cache_move.py`
- New test: `/job/sglang/test/registered/kernels/ops/kvcache/test_cache_move_compiled_path_reuse.py`
- Native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Native module SHA-256: `d38b79daf2c5dd193d49837c903798e6ff5440cbf31d899c2fd8ca68fed6aa98`

The installed-source baseline was recorded before checkout changes at
`/job/baseline-first.json`. Its source was `/sgl-workspace/sglang`, commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. That baseline is environment
context only and is not proof for the persistent-checkout changes.

## Method

The reuse case uses two contiguous `16 x 16` bfloat16 GPU buffers. Each buffer
has a 32-byte row stride, sentinel rows at indices 0 and 15, and active rows
1 through 14. The finite sequence is `num_locs = [3, 5, 7, 9]` with
`num_locs_upper = 16`, `bytes_per_tile = 128`, one byte tile, and four warps.

Every launch uses the overlapping source indices `[1, num_locs]` and target
indices `[2, num_locs + 1]`, so the documented in-place aliasing path is
exercised. Before each launch, the GPU buffers are restored from independent
CPU clones. The expected result is computed on CPU with
`index_copy_` and `index_select` from the original CPU reference.

Timing uses `time.perf_counter_ns` around one launch plus
`torch.cuda.synchronize()`. The first measurement includes Triton compilation;
later measurements are warm launches. No timing threshold is asserted because
these are bounded observations, not a performance gate.

The test asserts all rows are bit-exact against the CPU reference, both sentinel
rows remain unchanged, the `data_ptrs` values still equal the two buffer
addresses, the `data_ptrs` tensor address is unchanged, and the Triton compiled
cache grows by exactly one specialization across the four shapes.

The GPU wrapper now rejects unsupported `data_ptrs`, `strides`, `tgt_loc`, and
`src_loc` dtypes with a clear `TypeError` instead of relying on Triton to accept
or reinterpret them. The CPU path remains unchanged.

## Commands

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-471ba7c8c880/jit

/opt/venv/bin/python -m pytest -q -s \
  test/registered/kernels/ops/kvcache/test_cache_move_compiled_path_reuse.py

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hicache.py::test_hicache_transfer_mha
```

## Raw Results

- Focused reuse and dtype test: `5 passed, 1 warning in 4.96s`.
- Cold first launch (`num_locs = 3`): `740.43244 ms`.
- Warm launches:
  - `num_locs = 5`: `0.126728 ms`
  - `num_locs = 7`: `0.065392 ms`
  - `num_locs = 9`: `0.049563 ms`
- Compiled-cache delta across the sequence: `1`.
- Every result was bit-exact against the independent CPU reference.
- Both sentinel rows were unchanged after every launch.
- Unsupported dtype variants failed clearly with `TypeError`.
- Existing consumer regression control: `8 passed, 3 warnings in 64.35s`.

The observed Triton signature was:

```text
[('*u64', 'DS'), ('*i64', 'DS'), ('*i64', 'DS'), ('*i64', 'DS'),
 ('i32', ''), ('constexpr', 16), ('constexpr', 128)]
```

The installed-source baseline used the existing
`sglang.kernels.ops.memory.memcpy_triton.memcpy_triton` control. Its first JIT
call took `1310.218242 ms`; subsequent supported shapes took `0.227209 ms` and
`0.080804 ms`, and a warm repeat took `0.169567 ms`. All baseline calls were
bit-exact and left guard rows unchanged.

## Limitations

- Timing is one bounded pass per shape and is not a throughput benchmark.
- The probe covers one MI300X `gfx942`; CUDA and other ROCm architectures were
  not tested here.
- The test intentionally uses Triton's `device_caches` structure to prove reuse;
  that structure is specific to the installed Triton 3.7 API.
- This change does not implement or integrate the issue's proposed compaction
  algorithm and does not substantiate its Blackwell/H100 performance claims.
- No full model weights, model run, toolchain replacement, or unbounded stress
  test was used.
