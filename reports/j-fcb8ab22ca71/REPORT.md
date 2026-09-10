# gfx942 Qwen4 PLE gather index-property investigation

## Scope and prior evidence

- This is a separate follow-up to `amdpilot-org/sglang` issue 235, not a repeat of its completed PLE bandwidth baseline.
- Read-only upstream context: `sgl-project/sglang` issue 38731, the Qwen3.8-Flash-Next roadmap. It was open with no comments when read.
- Related upstream PR 38701 was open at commit `b254e1fd3a6f9f1c9ee48724c62c13168642efba`.
- Existing mirror PR 325 was open at commit `27e063d041e1ef321d643d0f9834969b9edfed2d`. It already fixes the fused hash signed-remainder mismatch and includes a synthetic FP8/BF16 gather control; that fulfilled scope was not duplicated or changed here.
- The new property tested the unchanged pinned-host row gather: permuted indices, repeated valid indices, explicit out-of-range IDs, TP shard boundaries, and bounded table-scale changes.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, serial `692412003101`, node 9, GUID 19304.
- Image requested by the operator: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. The container has no Docker/Podman socket, so the supplied local image ID could not be independently queried.
- Interpreter: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; ROCm/HIP `7.2.26015-fc0010cf6a`.
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`; native library directory `/opt/venv/lib/python3.10/site-packages/torch/lib`.
- Installed SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`, revision `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`, with native example `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Mirror checkout: `/job/sglang`, PR base `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Relevant Qwen4 source history: commit `52fecfdf0908dca24f4c6799ff5967125cc4110e` (`support qwen 3.8 flash next (#37500)`).
- Job-private JIT cache: `/tmp/sglang-cache-j-fcb8ab22ca71`; its ngram native module was `/tmp/sglang-cache-j-fcb8ab22ca71/gfx942/sgl_kernel_jit_ngram_embedding/build-4bc510426ba26483/deps-439295d6e91b6419/sgl_kernel_jit_ngram_embedding.so`.

## Installed-source baseline

The exact prior Qwen4 PLE test was unavailable in the installed source:

```text
/sgl-workspace/sglang/test/registered/kernel/embeddings/test_qwen4_ple_offload.py: No such file or directory
/sgl-workspace/sglang/python/sglang/kernels/ops/qwen4_ple.py: No such file or directory
```

The installed source predates that follow-up. A meaningful supported neighboring control was therefore used before cloning:

```bash
PYTHONPATH=/sgl-workspace/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-fcb8ab22ca71 \
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_ngram_embedding.py
```

Result: 10 passed in 32.98 seconds; total first successful GPU execution elapsed time was 34.759766 seconds. The test compares decode and general ngram gather/hash outputs with `torch.testing.assert_close(atol=0, rtol=0)`.

Bounded timing used the existing benchmark:

```bash
PYTHONPATH=/sgl-workspace/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-fcb8ab22ca71 \
SGLANG_IS_IN_CI=true \
/opt/venv/bin/python \
  /sgl-workspace/sglang/test/registered/kernels/benchmark/speculative/bench_ngram_compute_decode.py
```

It uses `triton.testing.do_bench` through `run_benchmark_no_cudagraph` and the CI range `[32, 1024]`. Raw results were 8.720/8.579 microseconds for batch 32 and 16.518001/9.101 microseconds for general/decode at batch 1024. This installed-source baseline is evidence for the environment only, not for later checkout changes.

## Mirror validation

The current mirror focused test was run before the property probe:

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-fcb8ab22ca71 \
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /job/sglang/test/registered/kernel/embeddings/test_qwen4_ple_offload.py
```

Result: 10 passed in 18.38 seconds; total elapsed time was 21.525426504 seconds.

The property probe used `Qwen4ExpPinnedHostEmbedding.gather` with deterministic finite CPU-generated index matrices. Every TP1 matrix contained all valid row IDs in permuted order, at least 26 repeated valid IDs, and five explicit invalid IDs (`-1`, `rows`, `rows + 1`, `2^31`, and `2^63 - 1`). The TP2 matrix covered global IDs 0 through 7, the `[4, 8)` shard, repeated IDs, and the same invalid boundary values. Table scales were bounded to `(8, 7)`, `(33, 64)`, and `(257, 257)` rows by embedding dimension, plus a `(8, 13)` TP2 shard case. Both BF16 and FP8-e4m3fn tables were tested; the operation contract still emitted BF16 and zeroed out-of-shard rows.

The independent reference was CPU shard-local row indexing after converting the pinned table to BF16, with out-of-shard or invalid rows replaced by zero. The numerical gate remained unchanged:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

All eight cases passed with maximum absolute error 0 and mismatch count 0. Timing used two warmup calls followed by the mean of ten real gather calls bracketed by CUDA events. Raw values are in `gpu_property_results.json`.

## Result and boundaries

- No gather mismatch was demonstrated, so no kernel or production code was changed.
- The property held for the tested permutations, duplicated valid indices, invalid boundaries, TP shard-local indexing, BF16/FP8 tables, and bounded table scales.
- This is single-GPU `gfx942` evidence only; no multi-GPU TP communication or end-to-end model run was performed.
- No model weights or checkpoints were downloaded.
- The property probe did not use CUDA Graph replay; its small-table timings include launch overhead and are not bandwidth-only claims.
- The installed source and mirror checkout are different revisions, as recorded above.
- No upstream issue, PR, comment, or review was posted or modified.
