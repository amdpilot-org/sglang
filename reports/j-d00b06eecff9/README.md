# gfx942 Triton decode benchmark report

## Scope

This report records a bounded, single-GPU investigation of the Triton decode path referenced by upstream issue 2271. It contains findings only; no kernel or runtime code was changed.

- GPU: one AMD Instinct MI300X, architecture `gfx942` (`(9, 4)`), one visible device.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP: `7.2.26015-fc0010cf6a`.
- Current-main source: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Current-main import path: `/job/sglang/python/sglang/__init__.py`.
- Torch native path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- No full model weights were downloaded, no node-wide state was modified, and no upstream issue, PR, or comment was posted or changed.

## Timing method

All benchmark scripts separate compile time from steady-state kernel timing:

- Compile: one cold call plus synchronize, reported as wall time and as cold wall time minus a one-call CUDA event.
- Kernel: CUDA events around 20 calls after one cold call and a five-call warmup.
- Numerical references use independent float32 PyTorch attention; no reference tolerance was changed.
- Triton caches were kept job-private under `/tmp/sglang-cache-j-d00b06eecff9`.

## Installed-source first baseline

The first GPU execution objective was completed before cloning or editing. The installed source was environment context only and is not proof for later checkout changes.

- Source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` (dirty in the image).
- Import path: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Existing layout-parity test: 1 passed in 9.14 seconds.
- First useful GPU execution elapsed time: 18 seconds.
- Shape: batch 1, 16 heads, head dimension 128, context 32768, float16, four KV splits.
- Independent float32 SDPA comparison: max absolute error `0.05877685546875`, mean absolute error `0.00986655056476593`; a generic `rtol=2e-2, atol=2e-2` gate failed.
- Cold first-call wall time: `1.1520139649510384` seconds.
- Steady-state 20-call mean: `1.1673280715942382` ms.

The raw early artifact is saved outside the repository at `/job/baseline-first.json` and copied here as `installed-baseline-first.json`.

## Current-main split-cap matrix

The current-main benchmark used batch sizes 1 and 4, contexts 8192 and 32768, 16 heads, head dimension 128, bfloat16, and six existing split-cap candidates: 1, 2, 4, 8, 16, and 32. All 24 cases passed the unchanged existing `rtol=1e-2, atol=1e-2` reference gate.

Steady-state kernel time in milliseconds:

| Batch | Context | 1 | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 1.3515 | 0.6800 | 0.3455 | 0.1969 | 0.1255 | 0.1189 |
| 1 | 32768 | 5.3949 | 2.7018 | 1.3573 | 0.6852 | 0.3523 | 0.2131 |
| 4 | 8192 | 1.3772 | 0.6994 | 0.3574 | 0.2090 | 0.1521 | 0.1351 |
| 4 | 32768 | 6.0262 | 3.6753 | 1.5307 | 0.9053 | 0.6864 | 0.5494 |

For this bounded matrix, 32 splits was fastest in every shape. The current AMD platform hook caps the default at 16; the evidence supports 32 for this tested non-MLA GQA geometry, but it does not justify a global AMD default change because MLA and other AMD architectures were not covered.

## Current-main GQA control

The direct GQA control used batch 1, context 32768, 32 query heads, 8 KV heads, head dimension 128, float16, and caps 8, 16, 32, and 64. All cases passed the unchanged AMD grouped-decode `rtol=5e-2, atol=5e-2` gate.

| Cap | Actual splits | Kernel ms |
|---:|---:|---:|
| 8 | 8 | 0.2065 |
| 16 | 16 | 0.1363 |
| 32 | 32 | 0.1342 |
| 64 | 64 | 0.1312 |

The 32- and 64-split results are close; 64 is not a clear win and increases split-buffer footprint.

## Existing upstream candidates

### PR 35801

- URL: https://github.com/sgl-project/sglang/pull/35801
- Tested commit: `84a51946708edd03ddfc93a3a5b373d0408e7486`.
- Its 26 dynamic-split tests passed in 13.83 seconds on gfx942.
- At context 32768, its dynamic heuristic selected the same actual split count as the legacy heuristic at both cap 8 and cap 32.
- Cap 8: legacy `0.2048` ms, dynamic `0.1882` ms.
- Cap 32: legacy `0.0751` ms, dynamic `0.0745` ms.
- All cases passed the unchanged PR test tolerance `rtol=3e-2, atol=3e-2`.
- Conclusion: the tested dynamic heuristic does not improve this long-context shape on gfx942; the cap dominates.

### PR 27786

- URL: https://github.com/sgl-project/sglang/pull/27786
- Tested commit: `e6dca69353c1409b4fbb0346c954e897a89290e2`.
- Its context-gating test and existing grouped-decode test passed on gfx942.
- At context 32768, all tested caps selected the full cap. Kernel time:
  - 8 splits: `0.2060` ms.
  - 16 splits: `0.1084` ms.
  - 32 splits: `0.0667` ms.
  - 64 splits: `0.2850` ms.
- All cases passed the unchanged AMD grouped-decode `rtol=5e-2, atol=5e-2` gate.
- Conclusion: on gfx942, 64 splits regresses badly and the PR’s AMD-specific 16-split cap is slower than 32 for this shape.

Cross-branch timings are not a direct current-main regression claim because these PR branches are older than current main and contain code drift.

## Current-main Lean attention control

Current main already contains the AMD Work-Centric Lean Attention path. The supported dispatch was benchmarked with Lean forced on and off at batch 1, 32 query heads, 8 KV heads, head dimension 128, float16, and a 32-split standard control.

| Context | Standard ms | Lean ms | Result |
|---:|---:|---:|---|
| 65536 | 0.1451 | 0.1627 | Lean slower |
| 98304 | 0.2088 | 0.2110 | Approximately equal |
| 131072 | 0.2654 | 0.2509 | Lean faster |

All cases passed the unchanged AMD grouped-decode `rtol=5e-2, atol=5e-2` gate. The existing Lean attention test suite passed: 11 tests and 20 subtests in 50.06 seconds.

For this GQA geometry, the measured Lean crossover is between 96K and 128K, not at the current 64K gate. This is architecture- and shape-specific evidence only.

## Reproduction

From a clean current-main checkout:

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d00b06eecff9/triton

/opt/venv/bin/python -m pytest \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention -q

/opt/venv/bin/python reports/j-d00b06eecff9/benchmark_triton_decode.py \
  --output reports/j-d00b06eecff9/benchmark-results.json

/opt/venv/bin/python reports/j-d00b06eecff9/benchmark_current_gqa.py
/opt/venv/bin/python reports/j-d00b06eecff9/benchmark_current_lean.py
```

The PR-specific scripts record their exact tested commits and must be run from those checked-out commits:

```bash
git checkout 84a51946708edd03ddfc93a3a5b373d0408e7486
/opt/venv/bin/python reports/j-d00b06eecff9/benchmark_pr35801.py

git checkout e6dca69353c1409b4fbb0346c954e897a89290e2
/opt/venv/bin/python reports/j-d00b06eecff9/benchmark_pr27786.py
```

## Limitations

- Only one assigned MI300X (`gfx942`) was used; no multi-GPU result is claimed.
- These are direct kernel/dispatch microbenchmarks, not end-to-end serving benchmarks.
- The installed-source baseline used a dirty image revision and is context only.
- PR 35801 and PR 27786 were tested at their preserved commits, but their branches are older than current main.
- Lean was forced on in the labeled Lean cases; production activation remains shape-dependent.
- The 32-split result is specific to the tested non-MLA GQA geometry and does not cover MLA, other batch/head geometries, or other AMD architectures.
- No full model weights were downloaded.
