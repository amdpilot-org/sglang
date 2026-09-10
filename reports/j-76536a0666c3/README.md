# gfx942 paged allocator OOM investigation

## Conclusion

The mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c` reproduces the paged KV allocator launch-order bug described by sgl-project/sglang issue 34399. Both `alloc_extend` and `alloc_decode` launch their allocation kernels before returning the expected `None` result when the logical free-page list is too short.

The open sgl-project/sglang pull request 34400, tested at commit `1af8b575ed3a46f075e7fd48707f31db06f4d35e`, moves the capacity checks before output allocation and kernel launch. On one AMD Instinct MI300X (gfx942), it preserves successful-path numerical results, returns `None` for both insufficient-capacity cases, launches zero kernels on refusal, and leaves the free-page counters and sentinel backing values unchanged.

No product code is changed here because the working fix already exists in the open upstream pull request; duplicating it would create avoidable review churn.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one AMD Instinct MI300X, GFX version `gfx942`, node ID 4, unique ID `0x78cd0e3e671b4299`, serial `692440003970`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, HIP `7.2.26015-fc0010cf6a`.
- Installed SGLang source: commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, `/sgl-workspace/sglang/python`.
- Installed native SGLang module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Delivery checkout: `/job/sglang`, Python source path `/job/sglang/python`.

## Installed-source baseline

This baseline is from the preinstalled source and is not proof for later checkout changes.

The control used a 16-token allocator with page size 4, one request with prefix length 0 and sequence length 8, and no KV backing tensor. `alloc_extend` returned `[4, 5, 6, 7, 8, 9, 10, 11]`, exactly matching the independent `alloc_extend_naive` result on the same GPU tensors and free-page order.

Timing used `time.perf_counter` around one synchronized `alloc_extend` call. The first GPU execution completed 7.319043455645442 seconds after script start, including imports, allocator construction, ROCm warmup, and the first kernel. After that, 20 bounded iterations had a median of 0.10549603030085564 ms, minimum 0.10261870920658112 ms, and maximum 0.4232339560985565 ms.

The complete raw record is `/job/baseline-first.json`.

## Persistent-checkout results

All cases used page size 4 and tiny index tensors only. The insufficient-capacity pools exposed two logical free pages while requiring three. Their free-page tensor was a view of an in-bounds backing tensor containing `[7, 99, 101]`; only `[7, 99]` was logically free. This makes the over-read deterministic without risking an invalid physical GPU address or exhausting VRAM.

### Mirror main

Source commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

- Sufficient extend: output `[4, 5, 6, 7, 8, 9, 10, 11]`; matched `alloc_extend_naive`; one kernel launch.
- Sufficient decode: output `[4]`; matched the page-arithmetic reference; one kernel launch.
- Insufficient extend: returned `None`; one kernel launch. The captured kernel output was `[28, 29, 30, 31, 396, 397, 398, 399, 404, 405, 406, 407]`, proving that page 101 beyond the logical free list was read.
- Insufficient decode: returned `None`; one kernel launch. The captured output `[28, 396, 404]` again used out-of-list page 101.
- Both refusals left `free_pages=[7, 99]`, free-page count 2, release-page count 0, available size 8, and backing sentinel `[7, 99, 101]` unchanged.
- Peak CUDA/ROCm allocated memory across the run was 36,864 bytes.

Raw record: `/job/results-main.json`.

### Upstream candidate

Source commit: `1af8b575ed3a46f075e7fd48707f31db06f4d35e` from sgl-project/sglang pull request 34400.

- Sufficient extend: output `[4, 5, 6, 7, 8, 9, 10, 11]`; matched `alloc_extend_naive`; one kernel launch.
- Sufficient decode: output `[4]`; matched the page-arithmetic reference; one kernel launch.
- Insufficient extend: returned `None`; zero kernel launches; all counters and sentinels unchanged.
- Insufficient decode: returned `None`; zero kernel launches; all counters and sentinels unchanged.
- Peak CUDA/ROCm allocated memory across the run was 36,864 bytes.

Raw record: `/job/results-pr-34400.json`.

The candidate's focused regression test was also run:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q test/registered/unit/mem_cache/test_paged_allocator_oom.py
6 passed, 1 warning in 5.88s
```

## Reproduction commands

The installed-source baseline was:

```text
PYTHONPATH=/sgl-workspace/sglang/python /opt/venv/bin/python /tmp/sglang_baseline_first.py
```

The persistent-checkout GPU cases were run from `/job/sglang` on each revision:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python /tmp/sglang_gpu_cases.py <label> <output.json>
```

GPU identity was recorded with:

```text
rocm-smi --showproductname --showserial --showuniqueid
```

The raw JSON artifacts remain at `/job/baseline-first.json`, `/job/results-main.json`, and `/job/results-pr-34400.json`.

## Limitations

- This is a focused allocator control on one gfx942 GPU, not an end-to-end serving run.
- The sentinel backing tensor keeps the demonstrated over-read physically in bounds. It proves a logical free-list over-read but does not claim that every allocator layout produces an illegal-address exception.
- No full model weights were downloaded and physical VRAM was never exhausted; the largest measured allocation was only 36,864 bytes.
- Timing is a bounded control measurement, not a throughput benchmark. First-execution time includes import, ROCm warmup, and JIT effects.
- The upstream candidate was tested at its own commit; its parent is upstream commit `a23670ddbf89678a53230ceb2cc5ed75236c4d85`, while the mirror baseline is newer at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- The successful-path checks cover the small extend and decode controls used here; they do not replace the broader upstream test suite.
