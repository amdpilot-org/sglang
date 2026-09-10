# gfx942 reshape-and-cache evidence

## Installed-source baseline

- GPU: one AMD Instinct MI300X, gfx942, Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Public import: `sglang.kernels.ops.kvcache:reshape_and_cache_flash`.
- Python and implementation paths: `/opt/venv/bin/python`; `/sgl-workspace/sglang/python/sglang/kernels/ops/kvcache/cache_ops.py`.
- Installed source revision: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` (preinstalled tree was dirty).
- Native artifacts: two `reshape_and_cache_flash.hsaco` files under `/tmp/sglang-cache-j-49c855f39783`.
- Image context: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1` (operator supplied).
- Cases: 7 tokens, 3 heads, head size 17; 5-slot pages; slots `[0, 4, 5, 9, 14, 19, 24]`; unused cache slots filled with `-12345`.
- Independent reference: Python indexed each slot as `(slot // page_size, slot % page_size)` and copied the corresponding token.
- FP16 and BF16: exact K/V equality and exact unused-slot sentinel equality.
- Timing: `torch.cuda.Event.elapsed_time` around 10 calls after 3 warmups and synchronization; FP16 `0.3114370107650757 ms` total, BF16 `0.39386799931526184 ms` total.
- First execution after process start: FP16 `0.8011023094877601 s`, BF16 `0.07077514939010143 s`.
- Raw baseline: `/job/baseline-first.json`. This installed-source baseline is not evidence for later checkout changes.

## Demonstrated wrapper mismatch

The public wrapper exposes `k_scale` and `v_scale` as independent optional tensors. The launcher used one `USE_SCALE` constexpr and always loaded both scale pointers when either was present. It also used `key` as the value-scale dummy pointer. On gfx942:

- both scales passed;
- key-only scaling corrupted V;
- value-only scaling corrupted K.

The fix splits the kernel specialization into `USE_K_SCALE` and `USE_V_SCALE`, and uses `value` as the value-scale dummy pointer.

## Checkout validation

- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Command: `PYTHONPATH=/job/sglang/python TRITON_CACHE_DIR=/tmp/sglang-cache-j-49c855f39783 /opt/venv/bin/python -m pytest test/registered/kernels/ops/kvcache/test_reshape_and_cache_flash.py -q`
- Result: `8 passed` (FP16/BF16 × no-scale/both/key-only/value-only), with exact page layout and sentinel checks.
- Adjacent gate: `PYTHONPATH=/job/sglang/python TRITON_CACHE_DIR=/tmp/sglang-cache-j-49c855f39783 /opt/venv/bin/python -m pytest test/registered/attention/test_fused_fp8_kv_write.py -q`
- Result: `7 passed`, covering the unchanged production both-scales FP8 path.
