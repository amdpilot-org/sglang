# gfx942 paged allocator representation follow-up

## Conclusion

This is a follow-up to the paged KV allocator work in sgl-project/sglang issue 34399. It does not repeat the original insufficient-capacity trigger: mirror pull request 239 and upstream pull request 34400 already cover that bug and its working ordering fix.

The new experiment tested execution representations for `PagedTokenToKVPoolAllocator.alloc_extend` and `alloc_decode`. The supported representation is a flat, contiguous, 1-D `torch.int64` free-page tensor on the allocator device. A stride-two view is not supported: on unmodified mirror `main`, both operations launch the Triton kernel and silently read skipped backing slots. A packed 2-D `(2, 1)` view and a flat `torch.int32` tensor happen to produce the small control values, but only through accidental address and dtype compatibility; they are outside the allocator's established representation.

This change adds a pre-launch guard for the kernel-facing tensors. Unsupported device, dtype, rank, and stride variants now raise a clear `ValueError` before output allocation or kernel dispatch. The guard does not copy tensors, change aliasing, change static addresses, or alter the supported flat `int64` path.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one AMD Instinct MI300X, GFX `gfx942`, node ID 5, unique ID `0x7ecf53b7cc7c10da`, serial `692440003936`.
- Python: `/opt/venv/bin/python` (3.10).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, Triton `3.7.0`.
- Installed source: commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, `/sgl-workspace/sglang/python`.
- Delivery base: mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Allocator source: `/job/sglang/python/sglang/srt/mem_cache/allocator/paged.py`.
- Native dispatch: `/job/sglang/python/sglang/kernels/ops/memory/allocator.py`, functions `alloc_extend_kernel` and `alloc_decode_kernel`.

## Installed-source baseline

The first GPU control ran before cloning or editing. It used the installed source, a 16-token allocator with page size 4, two logical free pages `[1, 2]`, and one extend request from prefix length 0 to sequence length 4.

The native result was `[4, 5, 6, 7]`, exactly matching the independent `alloc_extend_naive` reference. The operation made one Triton launch and left the sentinel-backed slots beyond the logical free list unchanged.

Timing used `time.perf_counter` around one synchronized call. The first GPU execution completed in 1.4365253569958263 seconds from script start, including imports, allocator construction, ROCm warmup, and JIT. Five post-JIT calls had a median of 166.00731760263443 microseconds. The complete record is `/job/baseline-first.json`; it is installed-source evidence only and is not proof for checkout changes.

## Representation probe

The probe used page size 4, tiny index tensors, and in-bounds backing storage. The stride-two backing was `[1, sentinel, 3, sentinel, ...]`; its logical free-page view was therefore `[1, 3]`, while skipped slots contained the sentinel. The packed control used shape `(2, 1)`, and the dtype control used `torch.int32`.

### Unmodified mirror main

Source commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

- Flat contiguous `int64`: both extend and decode matched their independent references, with one kernel launch each.
- Stride-two `int64`: extend expected `[4, 5, 6, 7, 12, 13, 14, 15]` but returned `[4, 5, 6, 7, -3673094580, -3673094579, -3673094578, -3673094577]`. Decode expected `[4, 12]` but returned `[4, -3673094580]`. Both made one kernel launch and read a skipped sentinel slot.
- Packed 2-D `int64`: both operations matched the reference and made one launch, but this relied on the packed view having the same base address as the flat representation.
- Flat `int32`: both operations matched the reference and made one launch, but this relied on Triton accepting the alternate pointer dtype.

Raw record: `/job/results-representation-main.json`.

### Guarded checkout

The supported flat contiguous `int64` control still matched both references and made one kernel launch. Every unsupported variant failed before launch:

- Stride-two `int64`: `free_pages must be contiguous, got stride [2]`; zero launches.
- Packed 2-D `int64`: `free_pages must be a 1-D tensor, got shape [2, 1]`; zero launches.
- Flat `int32`: `free_pages must have dtype torch.int64, got torch.int32`; zero launches.

All sentinel values remained unchanged. Peak CUDA/ROCm allocated memory in the guarded probe was 35,840 bytes. Raw record: `/job/results-representation-guarded.json`.

The bounded timing matrix recorded one synchronized first call and three post-JIT repetitions for each operation and representation. Median supported controls were 194.35 microseconds for extend and 118.12 microseconds for decode. Median guard-failure paths ranged from 38.53 to 66.31 microseconds. These are focused dispatch measurements, not throughput benchmarks.

## Tests

Focused new regressions:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q test/registered/unit/mem_cache/test_paged_allocator_representation.py
7 passed, 1 warning in 4.68s
```

Adjacent allocator tests:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_paged_allocator_lazy_release.py \
  test/registered/unit/mem_cache/test_paged_free_segment.py \
  test/registered/unit/mem_cache/test_paged_allocator_representation.py
23 passed, 3 warnings in 28.82s
```

GPU probes:

```text
PYTHONPATH=/job/sglang/python \
  REPRESENTATION_OUTPUT=/job/results-representation-main.json \
  /opt/venv/bin/python /job/representation_probe.py

PYTHONPATH=/job/sglang/python \
  REPRESENTATION_LABEL='mirror branch representation probe after guard' \
  REPRESENTATION_OUTPUT=/job/results-representation-guarded.json \
  /opt/venv/bin/python /job/representation_probe.py
```

## Contracts and limitations

- Kernel-facing tensors must be 1-D, contiguous, `torch.int64`, and on the allocator device.
- The guard preserves aliasing by validating views in place; it does not call `.contiguous()` or silently copy unsupported representations.
- The guard preserves static graph-address behavior for supported tensors; no pointer or storage is replaced.
- Unsupported representations fail clearly before output allocation and native dispatch.
- No model weights were downloaded, no toolchain was replaced, physical VRAM was not exhausted, and no unbounded stress loop was run.
- This is a focused allocator representation test on one gfx942 GPU, not an end-to-end serving benchmark or CUDA-graph capture test.
- Upstream pull request 34400 already fixes the original OOM launch ordering. That fix is intentionally not duplicated here.
