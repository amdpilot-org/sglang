# Investigation result: upstream issue 31236

Source issue: https://github.com/sgl-project/sglang/issues/31236

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2275

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

The reported defect is already fixed in the prepared source. No additional runtime source change is justified.

The issue's later reproduction identified the important detail behind the misleading exception: online ModelOpt calibration failed, then `ModelOptModelLoader` continued with a Hugging Face `transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration`. That object does not implement SGLang's `get_embed_and_head` speculative-worker interface. Adding another method to SGLang's Gemma4 class would therefore not fix the failing object.

Merged upstream commit `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63` (PR https://github.com/sgl-project/sglang/pull/33115) changed ordinary, non-serialized `modelopt_fp4` loading to use `DefaultModelLoader`. This preserves SGLang model construction and performs online FP4 conversion through its per-layer weight loaders. Explicit ModelOpt restore/save/export workflows deliberately retain `ModelOptModelLoader`.

The prepared base contains that correction. Its Gemma4 multimodal class also implements `get_embed_and_head`, and the current loader-selection probe returns `DefaultModelLoader` for the issue's online `modelopt_fp4` mode.

## Regression evidence

The fix commit added `TestModelOptFp4LoaderSelection.test_unquantized_modelopt_fp4_preserves_modelopt_workflows`. The pre-fix implementation selected `ModelOptModelLoader` for any `modelopt_fp4` request, so the regression's primary assertion fails before the fix. On the prepared base, it passes and independently checks three boundary cases: explicit checkpoint restore, checkpoint save, and export continue to select `ModelOptModelLoader`.

The companion draft test covers explicit excluded MTP weights, explicit serialized MTP weights, and inherited target quantization. Together the focused class produced `2 passed, 6 subtests passed`.

Raw evidence:

- `raw/upstream_issue.json`: issue discussion and reproduced HF-class diagnostic.
- `raw/upstream_pr_33115.json`: merged fix description and changed files.
- `raw/fix_loader_regression.patch`: exact failing-before/passing-after loader and test diff.
- `raw/test_loader_selection.log`: focused regression output.
- `raw/issue_specific_probe.log`: actual current loader, Gemma4 interface, legacy predicate, and workflow boundaries.
- `raw/gpu_inventory.log`: assigned device inventory.

## Limitations

The original command requires the Google Gemma 4 26B target and assistant weights and NVIDIA ModelOpt/NVFP4 on CUDA (the report used GB10/CUDA 13). Those weights are not present, and the assigned device is one AMD Instinct MI350X (`gfx950`) with ROCm 7.2. Consequently, this investigation did not claim a full model startup, ModelOpt calibration, generation-quality result, or CUDA NVFP4 kernel execution. The deterministic evidence qualifies the corrected loader/interface path only.
