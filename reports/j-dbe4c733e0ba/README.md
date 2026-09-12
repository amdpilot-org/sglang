# INT8 KV-cache feature intake report

## Outcome

`candidate_rejected`

The requested feature is not implemented by this change. Current `main` rejects
`--kv-cache-dtype int8`, and the issue does not define enough of the storage and
quantization contract to choose one interoperable integer representation. The
only directly related implementation found, upstream PR #22004, was closed
without merge and explicitly implemented only a phase-1 subset. Applying that
proposal to the recorded base also fails because the relevant runtime
architecture has since changed.

This report is deliberately not presented as a fix. It records the reproduced
behavior, related work, exact missing contract, and unavailable validation paths.

## Issue provenance

- Upstream issue: https://github.com/sgl-project/sglang/issues/37710
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2675
- Related closed proposal: https://github.com/sgl-project/sglang/pull/22004

## Reproduction on the prepared checkout

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Interpreter: `/tmp/amdpilot-repo-j-dbe4c733e0ba/venv/bin/python`

Command:

```bash
/tmp/amdpilot-repo-j-dbe4c733e0ba/venv/bin/python \
  -m sglang.launch_server --model-path /nonexistent --kv-cache-dtype int8
```

Exit code: `2`

Measured result:

```text
sglang serve: error: argument --kv-cache-dtype: invalid choice: 'int8'
(choose from 'auto', 'fp8_e5m2', 'fp8_e4m3', 'mxfp8', 'bf16',
'bfloat16', 'nvfp4', 'fp4_mx_block16', 'fp4_e2m1')
```

The failure occurs at argument parsing, before model loading. This reproduces
the absence of integer KV-cache support in the actual checkout without requiring
model weights.

## Existing related work

Upstream PR #22004 (`feat(srt): add phase-1 INT8 KV cache for Triton MHA`) was
closed unmerged on 2026-07-08. It selected per-token/per-head asymmetric INT8:

```text
zp = (min + max) / 2
scale = (max - min) / 255
q = round((x - zp) / scale)
x_hat = q * scale + zp
```

Its stated supported scope was MHA, the Triton backend, `kv_cache_dtype=auto`,
and page size 1. It explicitly excluded MLA, hybrid SWA pools, double sparsity,
PD disaggregation, hierarchical cache, deterministic inference, and Ascend.
It also introduced a separate `--int8-kv-cache` boolean instead of fulfilling
the requested `dtype:int`/GGUF-facing dtype contract.

The proposal cannot be applied to this base unchanged. `git apply --check`
fails because, among other changes:

- `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` moved to
  `python/sglang/kernels/ops/attention/decode_attention.py`;
- `model_runner_kv_cache_mixin.py` was removed;
- server arguments moved to typed argument-group fields and resolution hooks;
- the quantized KV-cache documentation path was removed;
- the Triton backend and pool configuration were substantially reworked.

The complete apply-check output is retained at runtime path
`/tmp/amdpilot-repo-j-dbe4c733e0ba/evidence/pr22004_apply_check.log`.

## Missing feature contract

The issue says only `dtype:int` and mentions GGUF. Implementing that literally
requires decisions that affect checkpoint compatibility, memory layout, kernels,
and numerical behavior:

| Contract area | Required decision |
| --- | --- |
| Integer format | INT8 vs UINT8 vs INT4/packed formats |
| Quantization | symmetric vs asymmetric; rounding and saturation rules |
| Granularity | per tensor, layer, token, head, channel, or block |
| Metadata | scale/zero-point dtype, shape, layout, persistence, and ownership |
| GGUF mapping | which GGUF KV/cache metadata identifies the runtime format; GGUF weight types alone do not define activation KV quantization |
| Backend scope | Triton MHA only or FlashInfer/FA/MLA/CPU/NPU and vendor backends |
| Cache features | page sizes, radix reuse, SWA/hybrid pools, HiCache, PD disaggregation, DCP, speculative decoding, and CUDA/HIP graphs |
| Accuracy | accepted error/KL thresholds and reference workloads |
| Serialization | compatibility with cache transfer/offload consumers |

Choosing the abandoned PR's particular asymmetric INT8 codec would be a new
design decision, not a mechanical implementation of the issue's stated contract.

## Validation performed

1. Current-main CLI reproduction above: exit 2; integer KV dtype rejected.
2. Existing quantized KV-cache regression tests:

   ```bash
   /tmp/amdpilot-repo-j-dbe4c733e0ba/venv/bin/python -m pytest -q \
     test/registered/unit/layers/quantization/test_compressed_tensors_kv_cache.py \
     test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py
   ```

   Result: `22 passed, 1 skipped` (exit 0). This measures preservation of the
   existing compressed-tensors/FP4 dtype behavior only; it is not INT8 evidence.
3. GPU availability: one AMD Instinct MI350X (`gfx950`), Torch
   `2.11.0+rocm7.2`, HIP `7.2.26015`; a basic tensor operation succeeded. No
   INT8 KV-cache GPU execution occurred because no accepted implementation is
   present. Consequently there is no independent attention numerical comparison.

Raw test output is retained under
`/tmp/amdpilot-repo-j-dbe4c733e0ba/evidence/`.

## Remaining work

All implementation work remains. Before implementation, the proposal needs at
least the format, quantization granularity, metadata/layout, GGUF mapping,
supported backend matrix, and accuracy acceptance criteria above. Once fixed,
validation must include failing-before/passing-after argument and pool tests,
GPU quantize/dequantize comparison against an independent PyTorch reference,
prefill/decode attention comparisons over adversarial ranges (constant tensors,
zero range, saturation, non-power-of-two dimensions, GQA, long contexts), cache
reuse/eviction tests, and serving-path accuracy on a qualifying model.

