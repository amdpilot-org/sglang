# gfx942 embedding/gather qualification report

## Scope and conclusion

This report covers generic vocabulary-parallel embedding/gather behavior requested by the unified-kernel namespace work in sgl-project/sglang issue 29630. It is intentionally separate from the GLM NextN TP-shard issue.

The persistent mirror base already contains the relevant merged upstream change, sgl-project/sglang PR 30948, including the fused Triton kernel, its layout gate, invalid-index clamping, and focused GPU tests. No duplicate code fix is needed. This report records the installed-source baseline, the persistent-checkout rerun, direct Torch-indexing comparisons, actual dispatch, and invalid-input diagnostics.

## Environment

- Campaign: `repo-e2e-20260909`; coordination tracker: amdpilot-org/amdpilotv2 issue 402.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one assigned AMD Instinct MI300X, gfx942, serial `692412003101`, node 9.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; Triton: `3.7.0`.
- Installed SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`; installed source head `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` (dirty environment context).
- Persistent checkout import: `/job/sglang/python/sglang/__init__.py`; tested head `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Public op: `/job/sglang/python/sglang/kernels/ops/embeddings/vocab_parallel_embedding.py`.
- Public layer: `/job/sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`.
- The embedding path is Triton JIT; no separate native embedding `.so` is dispatched. `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py` was recorded as the installed AOT package path, but it is not this embedding dispatch.

## Installed-source first GPU baseline

Command:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/embeddings/test_vocab_parallel_embedding.py
```

Timing used one cold process around `subprocess.run` with `time.monotonic()`; there was no burn-in or repeated work. The first successful GPU execution completed in `28.782759` seconds; pytest reported `22 passed` in `24.89` seconds. The full record is in `/job/baseline-first.json` on the job host.

The installed source was a different, dirty revision, so this baseline is labeled installed-source evidence only and is not proof for later checkout changes.

## Persistent checkout rerun

Command:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
HOME=/tmp/sglang-cache-j-9441809f3aaf/home \
/opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/kernels/ops/embeddings/test_vocab_parallel_embedding.py
```

Result: `22 passed`, return code `0`, wall time `25.014917` seconds. The unchanged numerical gate is exact: `torch.testing.assert_close(..., rtol=0, atol=0)`.

The first checkout attempt failed before GPU collection because a minimal environment omitted `HOME`, causing Torch to try creating `/home/amd`. Rerunning with the job-private `HOME` above fixed only that environment issue.

## Direct Torch-indexing comparison

The synthetic shard configuration was:

```text
org_vocab_start_index=10
org_vocab_end_index=18
num_org_vocab_padding=4
added_vocab_start_index=100
added_vocab_end_index=103
```

The independent reference computed valid local IDs and used Torch advanced indexing `weight[local_ids]`, then zeroed invalid rows with `torch.where`. It did not call SGLang's `get_masked_input_and_mask` or `F.embedding`.

All four real GPU cases matched exactly with `rtol=0, atol=0`:

- contiguous weight, int32 IDs;
- strided weight (`stride=(300, 1)`), int32 IDs;
- contiguous weight, int64 IDs;
- strided weight (`stride=(300, 1)`), int64 IDs.

Each input was shape `(2, 8)` and included valid original and added-vocabulary IDs, padding-adjacent boundaries, negative IDs, zero, out-of-shard IDs, and int32 boundary values. Every invalid row was all zero. Each output was shape `(2, 8, 257)` with contiguous strides `(2056, 257, 1)`.

Actual dispatch was recorded by wrapping the Triton kernel lookup. All four cases dispatched `_vocab_parallel_embedding_kernel` from `sglang.kernels.ops.embeddings.vocab_parallel_embedding` with grid `(16, 1)`. The public layer gate selected the fused path for TP2 unquantized CUDA int64 input and fell back for TP1, int16 input, and non-unit final-dimension weight stride.

Unsupported direct-op inputs raised `AssertionError` before launch for int16 IDs, noncontiguous input, weight `stride(1) != 1`, weight `ndim != 2`, and CPU weight.

## Async invalid-input diagnostic

With `SGLANG_ENABLE_ASYNC_ASSERT=1`, the public layer's `maybe_detect_oob` path was exercised directly:

- valid IDs `[0, 15]` against bounds `[0, 16)` synchronized without error;
- invalid IDs `[-1, 200]` surfaced `AcceleratorError` at synchronization, with HIP reporting an unspecified launch failure/device-side assertion from `_assert_async_cuda_kernel`.

This diagnostic was run in a fresh process after all numerical comparisons because a device-side assert intentionally poisons the CUDA context.

## Reproduction

The direct comparison and diagnostics are summarized in `results.json` in this directory. The job-private cache/log path was `/tmp/sglang-cache-j-9441809f3aaf`; no full model weights were downloaded, no node-wide state was changed, and no upstream issue, PR, or comment was posted or modified.

Nothing was left unresolved for this bounded scope. No workload code change was made because the relevant upstream fix and tests are already present at the tested mirror head.
