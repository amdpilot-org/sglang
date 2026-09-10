# gfx942 shared-FP8 / routed-FP4 MoE prerequisite

## Result

The numerical mapping prerequisite passes on one AMD Instinct MI300X (gfx942),
but no available native mixed-precision MoE kernel supports the required
separate FP8 shared expert and FP4 routed experts on gfx942. This is a bounded
investigation and report; it does not implement a MegaMoE backend.

The reproducible composition uses:

- a native gfx942 AITER FP8 block-scaled MoE for the separately owned shared
  expert;
- AITER's Torch FP4 dequantizing MoE stages for the routed experts;
- an appended shared slot (`top_k=2 -> 3`, shared expert ID `8`, weight `1.0`);
- an independent FP32 Torch reference that dequantizes both representations.

The shared weights remain `torch.float8_e4m3fnuz` with their own FP32
128x128-block scales. They are not requantized to FP4, and their data pointers
are unchanged after the check.

## Raw numerical result

| Metric | Result | Gate |
|---|---:|---:|
| Combined relative L2 | 0.03514507785439491 | <= 0.05 |
| Routed contribution relative L2 | 0.002315653720870614 | recorded |
| Shared contribution relative L2 | 0.043077096343040466 | recorded |
| Combined cosine distance | 0.0006177783345998611 | <= 0.001 |
| Routed weight probe relative L2 | 0.002315653720870614 | <= 0.005 |
| Shared weight probe relative L2 | 0.043077096343040466 | <= 0.05 |
| Maximum absolute error | 0.016566067934036255 | recorded |

The FP8 shared contribution error is consistent with the native kernel's BF16
output rounding and the small synthetic output magnitude. The AITER FP8 baseline
itself reports `logits_diff=3.90867e-06` while its strict elementwise
`checkAllclose` reports 5.8% differing elements.

## Unsupported kernel evidence

Two direct probes were preserved:

- Native gfx942 routed FP4 `fused_moe` aborts with:
  `fused_dynamic_mx_quant_moe_sort_hip: not support output type: fp4x2`.
- AITER's heterogeneous shared-FP8/routed-FP4 API raises:
  `Heterogeneous MXFP4/FP8 experts currently require gfx950`.

The installed AITER source explicitly gates heterogeneous MXFP4/FP8 experts to
gfx950. Its advertised gfx942 A16W4 test also fails during FlyDSL compilation
because the generated assembly uses `v_cvt_scalef32_pk_bf16_fp4`, which the
gfx942 assembler reports as unsupported. All 12 test cases fail for this
reason.

No `deep_gemm` Python module is installed in the qualified environment. This
report therefore makes no claim about DeepGEMM CUDA support.

## Environment and provenance

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local
  image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one AMD Instinct MI300X, gfx942, 304 CUs, unique ID
  `0xa09b4a46354a21d9`, serial `692412003101`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP
  `7.2.26015-fc0010cf6a`, module
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- AITER: commit `c16d44b93a528b2a4bfd6d8d3409116d465872a9`, module
  `/sgl-workspace/aiter/aiter/__init__.py`.
- Native AITER modules observed:
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`,
  `/sgl-workspace/aiter/aiter/jit/module_moe_sorting_opus.so`,
  `/sgl-workspace/aiter/aiter/jit/module_quant.so`, and
  `/sgl-workspace/aiter/aiter/jit/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so`.
- Delivery source: `/job/j-12bc43398f03/sglang`, branch
  `amdpilot/j-12bc43398f03`, base `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Preinstalled source context: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`; the delivery clone is separate.
- Upstream `sgl-project/sglang` main read for context:
  `3700c4ee26a1df3fd27e10a4a83d40d991d87d6b`.
- Job-private caches: `/tmp/sglang-cache-j-12bc43398f03/{aiter,triton,inductor}`.

## Upstream context

Issue 38700 requests shared-to-sparse fusion for DSV4 MegaMoE. Merged PR 27349
appends the shared expert as an additional physical expert and requantizes FP8
shared weights to FP4 for the fused FP4 path. The issue explicitly notes that
this does not preserve the original DSV4 checkpoint's separate blockwise-FP8
shared expert and MXFP4 routed expert precision.

The installed AITER revision contains a newer heterogeneous API with separate
`shared_w1`, `shared_w2`, and shared scale arguments. Its contract requires
gfx950, so it was tested as a candidate and rejected on gfx942 rather than
duplicated or silently lowered to FP4. No open upstream PR referencing issue
38700 was found at investigation time.

## Reproduction

From the delivery clone:

```bash
export AITER_CACHE_DIR=/tmp/sglang-cache-j-12bc43398f03/aiter
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-12bc43398f03/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-12bc43398f03/inductor
/opt/venv/bin/python reports/j-12bc43398f03/reproduce_gfx942.py \
  --output reports/j-12bc43398f03/gfx942-results.json
```

The script asserts one visible gfx942 GPU, runs the numerical gates, probes the
native FP4 path in a child process, and records the heterogeneous API rejection.
It does not download model weights or modify node-wide state.

## Limitations

- The routed FP4 path is AITER's Torch dequantizing reference, not a native
  gfx942 FP4 MoE kernel, because no such kernel is available in this stack.
- The shared expert uses FP32 128x128-block scales in this numerical check.
  The gfx950-only AITER heterogeneous candidate instead requires E8M0 per-32
  scale storage; that exact scale conversion was not numerically validated.
- This check covers routing weights, the appended shared slot, output
  combination, and dtype/scale ownership. It does not establish performance or
  end-to-end DSV4 correctness.
