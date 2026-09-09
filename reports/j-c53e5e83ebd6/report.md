# MLA reserved padding slot zero validation

## Result

**Already fixed / cannot reproduce** on the assigned MI300X. No runtime patch was
made because no overwrite was demonstrated.

The working clone at `amdpilot-org/sglang` main commit
`ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` already contains the
`reserved_skip_index=0` guards from upstream pull request 36003. Ten synthetic
sentinel cases, including padded maps and HIP graph replay, left slot zero
byte-for-byte unchanged and preserved every valid write.

## Investigation

- Read-only issue: `sgl-project/sglang` issue 36207, still open.
- Candidate read: upstream pull request 36003, merged on 2026-08-26. Its head
  was `01950f8813aecb012fb5a0a1e386bcaabfa89b24`; the implementation commit was
  `15f1e8969cedff1d1e8f830ac767214d104ab334`. I did not check out or modify the
  upstream repository.
- Tested mirror: `amdpilot-org/sglang` main commit
  `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`.
- SGLang-owned writer operations inspected and exercised:
  - `set_mla_kv_buffer_triton`
  - `set_mla_kv_buffer_dcp_sharded_triton`
  - `set_mla_kv_buffer_triton_fp8_quant`
  - `set_mla_kv_scale_buffer_triton`
- The CUDA TMA/JIT operation `set_mla_kv_buffer` is not supported on this HIP
  stack: `is_arch_support_pdl()` is false and the attempted native build fails
  because `cuda/ptx` is unavailable under `hipcc` for gfx942. Its source guard
  was inspected, but the native path was not run.
- The fused AITER operation `fused_qk_rope_cat_and_cache_mla` was not exercised
  because the SGLang integration `_fused_rope_cat_and_cache` is gated to gfx95.
  The assigned GPU is gfx942. Issue 36207 and pull request 36003 both identify
  this fused path as a separate API-safe follow-up; the installed AITER wrapper
  also has no `pad_slot_id` parameter.

## Synthetic validation

The reproducer is `validate_mla_slot_zero.py`. It:

1. Creates a device cache and a separate host cache-content reference.
2. Uses the padded location map `[0, 0, 2, 3, 4, 5]`.
3. Puts NaN sentinels in both source rows mapped to slot zero.
4. Computes the complete expected cache independently on the host, skipping
   only destination zero.
5. Runs eager writers and captures/replays writers with `torch.cuda.CUDAGraph`.
6. Restores the original cache sentinel before each graph replay.
7. Requires exact equality for the complete cache, slot zero, and all valid
   neighboring slots.

All ten cases reported `slot_zero_unchanged=true`,
`cache_exactly_matches_reference=true`, and `mismatch_count=0`. Raw JSON is in
`validation.log`.

The focused existing test command also passed:

```text
5 passed, 3 skipped, 55 deselected
```

The three skips are the CUDA-only TMA tests; this is an HIP gfx942 run.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-supplied local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, gfx942, capability `(9, 4)`
- GPU serial: `692440004395`
- GPU unique ID: `0x6e8448eb6f49db1d`
- PCI bus: `0000:0C:00.0`
- Python: `/opt/venv/bin/python` 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Working Python source: `/job/sglang/python`
- Triton writer source: `/job/sglang/python/sglang/kernels/ops/kvcache/mla_buffer.py`
- TMA wrapper source: `/job/sglang/python/sglang/kernels/ops/kvcache/set_mla_kv_buffer.py`
- TMA native source: `/job/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_buffer.cuh`
- Preinstalled source context: `/sgl-workspace/sglang/python`
- Installed AITER fused source context:
  `/sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py`

No model weights were downloaded. Build and Triton caches stayed outside the
working clone under `/job/.cache/sglang` and `/tmp`.

## Scope and limitations

- No runtime source or native file was changed.
- The gfx942 run cannot clear the known gfx95-only fused AITER follow-up.
- The CUDA TMA/JIT writer could not be executed on this HIP image for the
  native-toolchain reason recorded above.
- Numerical gates were unchanged: validation used exact tensor equality, not
  tolerance-based comparisons.
