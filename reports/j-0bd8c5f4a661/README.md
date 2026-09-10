# gfx942 reduced-block prefill study

This is a bounded, one-GPU numerical and throughput study. It makes no model-quality claim.

## Scope

- Campaign: `repo-e2e-20260909`
- Coordination: `amdpilot-org/amdpilotv2` issue 402
- Upstream context: `sgl-project/sglang` issue 31783
- Checked candidate: `sgl-project/sglang` PR 32576 is a resolution-time DSA compatibility table and has no GPU kernel change; this study does not duplicate it.
- Delivery commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, capability `(9, 4)`, 304 CUs
- Torch/ROCm: Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`

## Installed-source baseline

The first relevant installed-source test was:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/manual/attention/test_flashattn_backend.py::TestFlashAttentionBackend::test_forward_extend_cp
```

It is unsupported at installed source commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` because the mock lacks the new `token_to_kv_pool_allocator` hook:

```text
AttributeError: 'MockModelRunner' object has no attribute 'token_to_kv_pool_allocator'
```

The first GPU execution attempt took `35.22488278802484` seconds. A supported neighboring control then ran `TorchNativeAttnBackend.forward_extend` for BF16 KV, explicit values, and an independent SDPA reference. It passed `torch.allclose(atol=2e-2, rtol=2e-2)` with zero measured error, dispatched `aten::_flash_attention_forward`, and recorded a warm median of `0.0007292891386896372` seconds over ten real forwards. The complete baseline is in `/job/baseline-first.json`; it is installed-source evidence only and is not evidence for this checkout.

## Method

The checkout study uses the dummy-model direct AITER linear prefill path:

```text
aiter.ops.mha.mha_batch_prefill_func
linear 3D NHD K/V, page_size=1, causal=True
```

Each case uses explicit locally generated Q/K/V values from seed `31783`. BF16 uses the values directly. FP8 uses `float8_e4m3fnuz` with scale `0.02` and per-tensor Q/K/V descales. The independent reference dequantizes FP8 Q/K/V to FP32 and uses SDPA with GQA; BF16 uses the original values directly. Outputs are compared in BF16.

The unchanged numerical gate for every case and both modes is:

```text
max_abs_error <= 0.1 and mean_abs_error <= 0.01
```

This is a per-case output comparison only, not a model-quality gate.

Timing uses `time.perf_counter` around synchronized real forwards. The first BF16 and FP8 calls include JIT compilation (`50.2125` and `41.2249` seconds respectively). Each case-mode then records five warm real forwards and one additional profiler forward for dispatch capture. There is no synthetic burn or unbounded loop.

## Cases

All cases use head dimension `128` and `32768` total query tokens.

| Case | Batch | Sequence | Q heads | KV heads |
|---|---:|---:|---:|---:|
| `b8_s4096_h32_kv8_d128` | 8 | 4096 | 32 | 8 |
| `b16_s2048_h32_kv8_d128` | 16 | 2048 | 32 | 8 |
| `b32_s1024_h32_kv8_d128` | 32 | 1024 | 32 | 8 |
| `b8_s4096_h64_kv16_d128` | 8 | 4096 | 64 | 16 |
| `b16_s2048_h64_kv16_d128` | 16 | 2048 | 64 | 16 |
| `b32_s1024_h64_kv16_d128` | 32 | 1024 | 64 | 16 |

## Raw results

The complete dimensions, dtypes, scales, timings, allocations, profiler events, and errors are in `results.json`.

| Case | Mode | Warm median | Max error | Mean error | Query tokens/s | Approx. attention TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `b8_s4096_h32_kv8_d128` | BF16 | 3.257 ms | 0.031250 | 0.000233 | 10,060,246 | 337.57 |
| `b8_s4096_h32_kv8_d128` | FP8 | 2.383 ms | 0.064453 | 0.000922 | 13,751,639 | 461.43 |
| `b16_s2048_h32_kv8_d128` | BF16 | 2.155 ms | 0.031250 | 0.000319 | 15,205,730 | 255.11 |
| `b16_s2048_h32_kv8_d128` | FP8 | 1.412 ms | 0.064453 | 0.001247 | 23,203,378 | 389.29 |
| `b32_s1024_h32_kv8_d128` | BF16 | 1.296 ms | 0.031250 | 0.000433 | 25,285,375 | 212.11 |
| `b32_s1024_h32_kv8_d128` | FP8 | 0.956 ms | 0.078125 | 0.001669 | 34,276,541 | 287.53 |
| `b8_s4096_h64_kv16_d128` | BF16 | 7.268 ms | 0.031250 | 0.000233 | 4,508,294 | 302.55 |
| `b8_s4096_h64_kv16_d128` | FP8 | 5.004 ms | 0.064453 | 0.000923 | 6,548,606 | 439.47 |
| `b16_s2048_h64_kv16_d128` | BF16 | 4.136 ms | 0.031250 | 0.000319 | 7,922,134 | 265.82 |
| `b16_s2048_h64_kv16_d128` | FP8 | 2.970 ms | 0.070312 | 0.001248 | 11,033,458 | 370.22 |
| `b32_s1024_h64_kv16_d128` | BF16 | 2.678 ms | 0.031250 | 0.000433 | 12,234,059 | 205.25 |
| `b32_s1024_h64_kv16_d128` | FP8 | 1.930 ms | 0.078125 | 0.001670 | 16,974,534 | 284.79 |

All 12 case-mode results pass the unchanged numerical gate. The largest observed peak Torch allocation is `12,763,661,824` bytes (`11.89 GiB`), below the `48 GiB` live-allocation limit. The largest generated tensor set is `1,342,177,280` bytes (`1.25 GiB`), below the `4 GiB` limit. No model weights were downloaded.

Profiler dispatch records `aiter::mha_batch_prefill` followed by the corresponding CK tile prefill kernel for BF16 and FP8. Native modules used by the final run include:

```text
/tmp/sglang-cache-j-0bd8c5f4a661/aiter-jit-final/module_aiter_core.so
/tmp/sglang-cache-j-0bd8c5f4a661/aiter-jit-final/mha_batch_prefill_bf16_nlogits_nbias_mask_nlse_ndropout_nqscale_nsink.so
/tmp/sglang-cache-j-0bd8c5f4a661/aiter-jit-final/mha_batch_prefill_fp8bf16_nlogits_nbias_mask_nlse_ndropout_pertensor_nsink.so
```

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export AITER_JIT_DIR=/tmp/sglang-cache-j-0bd8c5f4a661/aiter-jit-final
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-0bd8c5f4a661/triton-final
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/sglang-cache-j-0bd8c5f4a661/flydsl-final

/opt/venv/bin/python \
  /job/sglang/reports/j-0bd8c5f4a661/reduced_block_prefill_study.py \
  --output /job/sglang/reports/j-0bd8c5f4a661/results.json \
  --repository /job/sglang
```

The cache directories are intentionally outside `JOB_WORKDIR`. The wall limit is 7200 seconds, the case count is six, and each case-mode is bounded to one cold call, five warm calls, and one profiler call.
