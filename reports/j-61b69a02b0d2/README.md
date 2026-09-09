# PR 55 FP16 KV-cache verification follow-up

## Result

This report continues the unresolved FP16 verification for [PR 55](https://github.com/amdpilot-org/sglang/pull/55), which addresses [sgl-project/sglang issue 30815](https://github.com/sgl-project/sglang/issues/30815). It does not duplicate or merge PR 55's runtime change into current `main`.

At the exact PR head `817319afcdf93e6830bc1e3ac8d63b9dd6b0df73`, the admitted FP16 fused path is byte-exact for finite values, scalar/device scales, changed graph-replay slots and scales, asymmetric rows, no-scale fallback, noncontiguous fallback, and row-alignment fallback. Two admitted adversarial cases failed exact byte comparison: direct NaN input and zero scale producing NaN. The existing GPU fallback and independent CPU cast encode this NaN as `0x80`; the fused Triton cast encoded it as `0xff`.

The smallest kernel correction is preserved in [pr55-fp16-nan-payload.patch](pr55-fp16-nan-payload.patch). Its exact base is `817319afcdf93e6830bc1e3ac8d63b9dd6b0df73`; it is not applied to this current-main report branch. The correction detects NaN before `tl.clamp`, canonicalizes the FP8 payload to `0x80`, and adds a zero-scale FP16 regression test. With that patch, all focused FP16 cases pass byte-exactly and the neighboring BF16 test file passes (`8 passed`). No FP16 case remains unresolved after the candidate patch.

## Final handoff

The current-main mirror PR is [PR 63](https://github.com/amdpilot-org/sglang/pull/63), opened as a draft and left unmerged. Its report-only branch is `amdpilot/j-61b69a02b0d2`, cut from `main` at `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`. The exact-base candidate patch remains clearly identified and unmerged; it is not silently claimed to be part of current `main`.

## Environment and checkouts

- Validation control: `/job/sglang-validation-817319`, detached at `817319afcdf93e6830bc1e3ac8d63b9dd6b0df73`, clean.
- Candidate: `/job/sglang-candidate-817319`, branch `validation/pr55-fp16-nan-payload` based on the same commit, dirty with only the identified patch.
- Delivery: `/job/sglang-main-delivery`, branch `amdpilot/j-61b69a02b0d2` from current `main` `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`.
- Imported control sources: `python/sglang/srt/mem_cache/memory_pool.py` and `python/sglang/kernels/ops/kvcache/triton_store_cache.py`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime: `7.2.26015-fc0010cf6a`; Triton: `3.7.0`.
- `hipcc --version`: HIP `7.2.26015-fc0010cf6a`, AMD clang `22.0.0git` (`roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5`). A bare `clang` executable was not on `PATH`.
- GPU: exactly 1 visible GPU, AMD Instinct MI300X, `gfx942` / CUDA capability `(9, 4)`.
- Native caches were kept under `/job/native-cache`, outside all checkouts.

## Reproduction

The harness uses the actual `MHATokenToKVPool.set_kv_buffer` entrypoint. It independently performs float32 division, explicit `float8_e4m3fnuz` clamp/cast on CPU, and slot scatter, then compares every cache byte (including untouched slots) and checks input nonmutation bitwise for FP16/BF16.

```bash
export TRITON_CACHE_DIR=/job/native-cache/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/job/native-cache/inductor-cache

# Baseline at exact PR head.
cd /job/sglang-validation-817319
SGLANG_VALIDATION_OUTPUT=/job/native-cache/pr55_fp16_validation.json \
  python /job/sglang-main-delivery/reports/j-61b69a02b0d2/validate_pr55_fp16.py

# Candidate from the same exact head.
cd /job/sglang-candidate-817319
git apply /job/sglang-main-delivery/reports/j-61b69a02b0d2/pr55-fp16-nan-payload.patch
SGLANG_VALIDATION_CHECKOUT=/job/sglang-candidate-817319 \
  SGLANG_VALIDATION_OUTPUT=/job/native-cache/pr55_fp16_candidate_validation.json \
  python /job/sglang-main-delivery/reports/j-61b69a02b0d2/validate_pr55_fp16.py

PYTHONPATH=/job/sglang-candidate-817319/python \
  python -m pytest -q test/registered/unit/mem_cache/test_mha_fp8_kv_write_hip.py
```

The synchronized latency harness records CUDA-event elapsed time after warmup and synchronization. Run it once per checkout with `SGLANG_BENCHMARK_CHECKOUT` set to that checkout and `SGLANG_BENCHMARK_OUTPUT` set to a JSON path.

## Raw evidence

- Baseline matrix: [pr55_fp16_baseline_validation.json](pr55_fp16_baseline_validation.json)
- Candidate matrix: [pr55_fp16_candidate_validation.json](pr55_fp16_candidate_validation.json)
- Candidate neighboring test log: [pr55_candidate_existing_and_nan_test.log](pr55_candidate_existing_and_nan_test.log)
- Latency rounds: [pr55_baseline_latency_round1.json](pr55_baseline_latency_round1.json), [pr55_baseline_latency_round2.json](pr55_baseline_latency_round2.json), [pr55_candidate_latency_round1.json](pr55_candidate_latency_round1.json), [pr55_candidate_latency_round2.json](pr55_candidate_latency_round2.json)

### Validation matrix

| Case | PR-head baseline | Candidate |
|---|---:|---:|
| FP16 default scales | PASS | PASS |
| FP16 scalar scales | PASS | PASS |
| FP16 negative and integer scalar scales | PASS | PASS |
| FP16 device scales | PASS | PASS |
| FP16 no-scale existing fallback | PASS | PASS |
| FP16 asymmetric rows | PASS | PASS |
| FP16 saturation/underflow/sign boundaries | PASS for finite values; FAIL for NaN payload | PASS |
| FP16 zero scale (publicly admitted) | FAIL: NaN payload `0xff` vs expected `0x80` | PASS |
| FP16 noncontiguous input fallback | PASS | PASS |
| FP16 unsupported device-scale dtype fallback | PASS | PASS |
| FP16 row-alignment fallback | PASS | PASS |
| BF16 scalar-scale control | PASS | PASS |
| BF16 device-scale control | PASS | PASS |
| FP16 CUDA graph with changed slots/device scales | PASS | PASS |

All passing comparisons had zero mismatched K and V cache bytes. The baseline failures were isolated to NaN payload bytes; finite saturation, underflow, sign, zero, infinity, and negative-zero values matched exactly.

### Paired latency

The runtime candidate changes the fused kernel, so synchronized paired baseline/candidate latency was collected. Each value is microseconds per call, averaged over 200 calls after 30 warmups. Two alternating rounds were run baseline/candidate/baseline/candidate.

| Configuration | Baseline rounds | Candidate rounds | Baseline mean | Candidate mean | Delta |
|---|---:|---:|---:|---:|---:|
| 128 tokens, 8 heads, 128/128, scalar scales | 38.898, 36.128 | 37.128, 36.720 | 37.513 | 36.924 | -1.57% |
| 128 tokens, 8 heads, 128/128, device scales | 38.079, 35.391 | 36.281, 36.586 | 36.735 | 36.434 | -0.82% |
| 64 tokens, 8 heads, 192/128, scalar scales | 37.350, 34.795 | 36.201, 35.409 | 36.072 | 35.805 | -0.74% |

These are microbenchmarks of the cache-write entrypoint only. No full model or overall inference speedup is claimed.
