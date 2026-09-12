# Investigation result: fix already present

Upstream issue: https://github.com/sgl-project/sglang/issues/34895

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1476

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Outcome: `fixed`

The reported defect is already fixed in this checkout by upstream PR
https://github.com/sgl-project/sglang/pull/35228, merged as
`5375babbac9977cdb8f061cec77b6efd0987a1fd` on 2026-08-19. The source issue
was opened before that merge. No production source change is justified on the
prepared base.

## Source evidence

At the parent of the fix (`c7e2c08d14da7b1e3df9af4b7b637f7d683d41b7`),
`CompressedTensorsConfig.get_quant_method()` had no `ParallelLMHead` branch.
PR #35228 added the branch, `get_lm_head_scheme()`, and
`test_compressed_tensors_lm_head.py`. On the prepared base:

- a name/regex-targeted `ParallelLMHead` resolves to
  `CompressedTensorsLinearMethod` with `CompressedTensorsW8A8Fp8`;
- `create_weights()` registers `weight_scale` as a channel scale parameter;
- `LogitsProcessor._compute_lm_head()` calls the quant method's `apply()`;
- unmentioned, ignored, and module-type-only heads remain unquantized;
- dotted-prefix targets resolve, while unsupported block-scaled heads fail
  explicitly rather than silently dropping scales.

The regression added with the fix is a failing-before/passing-after test: its
quantized-head assertion receives `None` on the pre-fix dispatch and a
`CompressedTensorsLinearMethod` on this base. All nine current regression and
boundary tests pass; raw output is retained in `unit_tests.txt`.

## GPU evidence

`gpu_fp8_lm_head_check.py` instantiated the real current `ParallelLMHead` and
compressed-tensors W8A8-FP8 method on the assigned AMD Instinct MI350X
(`gfx950`). It populated nonuniform per-channel scales spanning 100x, ran the
actual FP8 apply path, and compared its logits with an independent PyTorch
dequantize-then-matmul reference. The results were cosine 0.9996408 and top-1
agreement 17/17; omitting the scales reduced cosine to 0.6587629. Raw output is
retained in `gpu_fp8_lm_head_check.txt`.

## Limitations

The reported `unsloth/Qwen3.8-27B-NVFP4` weights and two RTX 5090 GPUs were not
available. Therefore this investigation does not claim a full-model, semantic,
SM120, TP=2, or HTTP serving reproduction. The available single-gfx950 fixture
qualifies the dispatch, scale retention, and numerical FP8 head execution only.
No native C++ or FlyDSL source changed, so no native rebuild was applicable.
