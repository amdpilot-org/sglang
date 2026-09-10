# gfx942 FP8 paged MQA logits investigation

## Scope

- Upstream context: `sgl-project/sglang` issue 34718.
- Related merged upstream PR: 34167 (`topk_v2` DSMEM/address-space fix).
- No upstream issue, PR, or comment was modified.
- Persistent mirror base: `amdpilot-org/sglang` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, 206 GB HBM.
- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP: `7.2.26015-fc0010cf6a`
- Persistent checkout: `/job/sglang`
- Installed AITER: `/sgl-workspace/aiter/aiter/__init__.py`
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`

## Baseline

The installed-source baseline used read-only `/sgl-workspace/sglang` at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`; it is not proof for later checkout
changes.

1. Direct DeepGEMM test:

   ```bash
   /opt/venv/bin/python -m pytest \
     /sgl-workspace/sglang/test/registered/kernels/ops/attention/test_deepgemm_paged_mqa_logits.py \
     -q --tb=short
   ```

   Result: `192 skipped`, elapsed `6545 ms`. All cases were skipped because the
   SM90/SM100 DeepGEMM path is architecture-gated and does not support gfx942.

2. Neighboring metadata control:

   ```bash
   /opt/venv/bin/python -m pytest \
     /sgl-workspace/sglang/test/registered/kernels/ops/attention/test_paged_mqa_metadata.py \
     -q --tb=short
   ```

   Result: `54 failed`, elapsed `152544 ms`. JIT compilation failed with:

   ```text
   fatal error: 'cub/block/block_scan.cuh' file not found
   ```

   `/opt/rocm/include/cub` and `/opt/rocm-7.2.0/include/cub` are absent in this
   image, so the control could not execute numerically.

## Real GPU cases

The added test uses AITER's `deepgemm_fp8_paged_mqa_logits` base path with
`KVBlockSize=1`, `Preshuffle=False`, and `ChunkK=128`.

1. Small valid cache:
   - Batch `1`, `next_n=1`, heads `32`, head dim `128`.
   - Context `64`, output width `256`.
   - Independent dequantized dot-product reference.
   - Maximum absolute error: `1.1444091796875e-05`.
   - Elapsed kernel-and-synchronize time: `980.182577855885 ms`.

2. Bounded longer-row/page-map matrix:
   - Batch `4`, `next_n=2`, heads `32`, head dim `128`.
   - Contexts `[65, 129, 193, 257]`, output width `512`.
   - Independent dequantized dot-product reference.
   - Maximum absolute error: `3.0517578125e-05`.
   - Elapsed kernel-and-synchronize time: `1069.3240631371737 ms`.
   - Allocation budget: below `1 MiB`; observed tensor bytes were roughly
     `q=32768`, `kv=82432`, `fused=85008`, `out=4096`, `page_map=2048`,
     `weights=256`.

Sentinel bounds were checked per row:

- Valid positions match the independent reference.
- Positions between the row's valid end and its context edge are `-inf`.
- Positions beyond the context edge retain the `-12345` sentinel.
- Unused page-map entries retain `-1`.

## Raw exploratory errors

- A non-preshuffle exploratory run with `KVBlockSize=64` produced:

  ```text
  HIP error: an illegal memory access was encountered
  ```

- A production-preshuffle exploratory run with `KVBlockSize=64` and
  `ChunkK=128` returned large garbage values rather than matching the
  independent reference. This was observed with the installed JIT AITER path;
  no fix is claimed here.

## Architecture-specific limitation

DeepGEMM's `fp8_paged_mqa_logits` is explicitly gated to SM90 or newer NVIDIA
architectures in the existing test and does not run on gfx942. The passing
gfx942 evidence here is for AITER's base one-token-page path, not the upstream
DeepGEMM SM90 kernel from issue 34718. The production preshuffle and
64-token-block configurations remain unproven by this investigation.

## Reproduction

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  test/registered/amd/test_aiter_paged_mqa_logits.py -q --tb=short
```

Observed result:

```text
2 passed, 3 warnings in 25.72s
```
