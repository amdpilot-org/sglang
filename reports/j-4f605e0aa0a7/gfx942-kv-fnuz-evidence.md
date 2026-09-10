# gfx942 fused FP8 KV-cache evidence

## Scope

This records cache encoding and dequantization roundtrip fidelity only. It does not claim coverage for dynamic activation quantization, kpool dispatch, or scaled GEMM.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, 206,141,652,992 bytes HBM.
- Image: local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12).
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, version `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Installed-source context: `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Mirror base: `/job/sglang`, commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- JIT cache: `/tmp/sglang-cache-j-4f605e0aa0a7` (outside `JOB_WORKDIR`).
- No model weights were downloaded.

## Source and native paths

- Python wrapper: `python/sglang/kernels/ops/kvcache/fused_fp8_qkv_kv_cache.py`.
- JIT source: `python/sglang/kernels/jit/csrc/attention/fused_fp8_qkv_kv_cache.cuh`.
- Focused test: `test/registered/kernels/ops/kvcache/test_fused_fp8_qkv_kv_cache.py`.
- Native bf16 module: `/tmp/sglang-cache-j-4f605e0aa0a7/gfx942/sgl_kernel_jit_fused_fp8_qkv_kv_cache_bf16_t_false/build-ebefcf10bee0564d/deps-7ff4f3e9804b31ae/sgl_kernel_jit_fused_fp8_qkv_kv_cache_bf16_t_false.so`.
- Native fp16 module: `/tmp/sglang-cache-j-4f605e0aa0a7/gfx942/sgl_kernel_jit_fused_fp8_qkv_kv_cache_fp16_t_false/build-6d7a4b3b1a0a888c/deps-7ff4f3e9804b31ae/sgl_kernel_jit_fused_fp8_qkv_kv_cache_fp16_t_false.so`.

## Raw results

Timing is wall clock from GNU `date +%s%N` immediately around each bounded pytest process. Elapsed time includes imports, JIT compilation, allocations, GPU execution, and assertions.

1. Installed-source control passed:
   `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4f605e0aa0a7 /opt/venv/bin/python -m pytest -q -s 'test/registered/kernels/ops/kvcache/test_store_cache.py::test_store_cache[1-64]'`
   - Result: `1 passed`; 26.481304329 s.
   - Gate: exact bf16 K/V cache bytes at written locations.

2. Installed fused FP8 target failed before this change:
   `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4f605e0aa0a7 PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s 'test/registered/kernels/ops/kvcache/test_fused_fp8_qkv_kv_cache.py::test_fused_fp8_qkv_kv_cache[True-False-None-1-8-1-128-dtype0]'`
   - Result: `1 failed`; 10.694812375 s.
   - Error: native matcher expected `uint8` on HIP while Python supplied `float8_e4m3fn`.

3. Supported neighboring FNUZ control passed:
   `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4f605e0aa0a7 PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py::test_set_mla_kv_buffer_triton_fp8_quant_reserved_skip_index`
   - Result: `1 passed`; 6.931633246 s.
   - Gates: exact FNUZ bytes at a written slot and unchanged reserved-slot bytes.

4. Focused fidelity test passed after the fix:
   `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4f605e0aa0a7 PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s test/registered/kernels/ops/kvcache/test_fused_fp8_qkv_kv_cache.py::test_fused_fp8_qkv_kv_cache_fnuz_fidelity`
   - Result: `2 passed`; final post-portability-check run 7.456300859 s.

5. Original Q+KV case and no-Q control passed after the fix:
   `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4f605e0aa0a7 PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s 'test/registered/kernels/ops/kvcache/test_fused_fp8_qkv_kv_cache.py::test_fused_fp8_qkv_kv_cache[True-False-None-1-8-1-128-dtype0]' 'test/registered/kernels/ops/kvcache/test_fused_fp8_qkv_kv_cache.py::test_fused_fp8_qkv_kv_cache[False-False-None-1-8-1-128-dtype0]'`
   - Result: `2 passed`; 6.251219108 s.

Intermediate failures are preserved: the first post-change run exposed `q=None` being viewed as bytes; the second exposed a missing native argument; the third exposed an independent-reference subnormal shift error. Each was fixed and the same focused cases rerun.

## Numerical gates

The focused test uses an independent bit-level E4M3FNUZ encoder implementing round-to-nearest-even and finite saturation, not the kernel or Torch FP8 cast as the encoder. It checks:

- Separate K and V scales (`1.7` and `0.9`) and separate source row strides.
- Exact encoded K and V bytes at all written locations.
- Zero-valued source rows encode to all-zero bytes.
- Positive and negative outliers saturate exactly.
- Writes land on page-boundary and page-boundary-adjacent slots with a 65-element tail dimension.
- Every unused K and V cache byte remains exactly unchanged.
- Independent reference dequantization matches the cache dtype's dequantized values exactly.

Upstream PR 31652 was inspected as context. It adds a CUDA-only `store_cache_quant` path and skips HIP, so this change does not duplicate it.
