# Structured BF16/FP8 KV write evidence

## Scope

This record covers a bounded, real-GPU check of the supported fused
BF16→FP8 paged KV-cache write path. It does not make any model-quality,
accuracy, throughput, or production-readiness claim.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, device ID `0x74a1`, GUID `19304`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm: `7.2.26015-fc0010cf6a`.
- Checkout commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Kernel source: `python/sglang/kernels/ops/kvcache/cache_ops.py`.
- Attention re-export: `python/sglang/kernels/ops/attention/utils.py`.
- FP8 dtype helper: `python/sglang/kernels/ops/quantization/fp8_kernel.py`.
- Test source: `test/registered/attention/test_fused_fp8_kv_write.py`.

## Installed-source baseline

The first GPU execution used the preinstalled source at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` and the existing
`fused_fp8_qkv_kv_cache` unit test. It reached the generated C++/TVM FFI
tensor-match boundary but failed before numerical assertions because the
wrapper required `uint8` cache tensors while the test allocated
`float8_e4m3fn`:

```text
RuntimeError: Dtype value [float8_e4m3fn] not in the allowed options: [uint8]
```

The bounded first execution elapsed `26.849387 s` and did not reach the
test’s independent reference comparison. A meaningful supported neighboring
control, `test_mla_kv_pack_quantize_fp8.py`, passed in `13.066523 s` on the
same GPU. Full details are in `/job/baseline-first.json`; this baseline is
installed-source evidence only and is not proof for later checkout changes.

## Structured checkout test

The persistent mirror checkout adds
`TestFusedFp8KvWrite::test_structured_inputs_match_dequantized_references`.
It uses five fixed, identical-shape cases:

1. `zeros`
2. `tiny` finite values (`1e-6`)
3. `mixed` magnitudes (`1e-3`, `0.1`, `1.0`, `10.0`, `50.0`)
4. `cancel` (linearly spaced positive/negative values)
5. `skew` (sparse `50.0` and `-25.0` state)

All cases use:

- `num_tokens=32`
- `num_heads=4`
- `head_dim=128`
- `total_slots=64`
- `page_size=1`
- explicit per-tensor `k_scale=v_scale=0.5`

The test allocates complete caches of shape
`(64, 1, 4, 128)`:

- BF16 cache: `32768` bytes per K or V cache.
- FP8 cache: `16384` bytes per K or V cache.

Each case records two real dispatches through
`launch_reshape_and_cache_flash`:

1. BF16 write with no scale.
2. FP8 write with explicit `k_scale` and `v_scale`.

The BF16 complete-cache reference is an independent scatter of the input.
The FP8 complete-cache reference is independent of the fused kernel:
`float32 divide → FP8 cast → scatter → dequantize`. The test compares both
raw FP8 bytes and dequantized FP8 values, and checks every cache slot,
including untouched slots.

## Raw results

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/attention/test_fused_fp8_kv_write.py::TestFusedFp8KvWrite::test_structured_inputs_match_dequantized_references
```

Result:

```text
1 passed, 5 subtests passed in 6.20s
```

Bounded wall time around the process: `7.753316 s`.

Full focused file command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/attention/test_fused_fp8_kv_write.py
```

Result:

```text
8 passed, 9 warnings, 5 subtests passed in 15.82s
```

Bounded wall time around the process: `18.503928 s`.

## Boundaries and notes

- No checkpoint or model weights were downloaded.
- No model-quality claim is made.
- No upstream issue, pull request, or comment was posted or changed.
- The installed-source `fused_fp8_qkv_kv_cache` failure is preserved as a
  concrete environment boundary, not hidden or replaced by a fabricated result.
