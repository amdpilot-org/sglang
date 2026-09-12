# Investigation report: HiCache draft KV direct-backup failure

Upstream issue: https://github.com/sgl-project/sglang/issues/31252

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2269

## Outcome

`candidate_verified`: the prepared `main` checkout already contains the
issue-specific architectural correction merged by upstream PR
https://github.com/sgl-project/sglang/pull/30393 at commit
`8e11feb68e057eb5f8fb8ababa97d4b4d9d40f07` on 2026-08-06. No additional
production-code change is justified.

The original v0.5.15.post1 controller performed two independent writes with the
same indices: first to the target host pool, then to
`mem_pool_host_draft`. The reported exception was raised by that second call.
The current implementation no longer has that separate draft write. For
standard NextN EAGLE models (the category that includes GLM-5.x),
`_can_pack_hicache_mtp` selects `HiCacheDraftMode.PACKED`, attaches the draft
device pool to the target runner, extends the target host pool's layer count,
and sends the target and draft buffers through one all-layer direct transfer.
Load-back reconstructs a draft transfer from the appended host layer.

Relevant current-source locations:

- `python/sglang/srt/speculative/base_spec_worker.py`: `_can_pack_hicache_mtp`
  and `_build_hicache_draft_plan` select and install the packed plan.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`:
  `build_kv_only_group` appends draft layer mappings to the target host pool.
- `python/sglang/srt/mem_cache/pool_host/mla.py`:
  `MLATokenToKVPoolHost.get_size_per_token` includes draft layers and
  `backup_from_device_all_layer` resolves the packed target+draft buffers for
  `transfer_kv_all_layer_direct_lf_pf`.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`:
  `_l2_load_transfers` maps appended host layers back to draft device pools.

## Validation

The CPU tests in `unit_tests.log` exercise asymmetric MHA direct dispatch and
page-first-direct host registration boundaries. Result: 20 tests and 12
subtests passed.

The GPU tests in `gpu_direct_transfer.log` executed the real installed
`sgl_kernel.transfer_kv_all_layer_direct_lf_pf` implementation on the assigned
AMD Instinct MI355X (gfx950), with page size 64, for both MHA and MLA
layer-first-to-page-first copies. Each test compares device/host results to an
independent PyTorch indexed-copy reference. Result: 2 passed.

No native source was changed or rebuilt.

## Limitations

The original deployment cannot be reproduced in this environment: it requires
CUDA's `cudaMemcpyBatchAsync`, GLM-5.2-FP8 and its EAGLE draft weights, eight
NVIDIA GPUs per role, two PD nodes, and Mooncake/RDMA. This job provides one AMD
gfx950 GPU and no reported model weights or multi-node fabric. Therefore the
result is not a claim that the original CUDA/TP8/PD workload was reproduced.
It is implementation-level evidence that current source has removed the exact
failing second-draft-pool call, plus numerical validation of the corresponding
direct-copy primitive on the available accelerator.
