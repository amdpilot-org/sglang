# Independent review of PR 1200

Reviewed exact candidate commit `e0cf917cd8902b7684bb8c2f26899f41f51cd542` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/36599.

Recommendation: **accept**. The candidate fixes the narrow issue-36599 contract for the reported GLM-5.3-Flash-NVFP4 metadata while preserving DeepSeek's existing BF16 NextN behavior.

## Evidence and reasoning

On the recorded base, the actual `DeepseekModelNextN` constructor receives an explicit `modelopt_fp4` config and replaces it with `None` before constructing `DeepseekV2DecoderLayer`. The reproduction imports source from `/job/repo/python`, not an installed SGLang wheel.

At the exact candidate, the blanket override is removed from the shared constructor. DeepSeek's resolver now returns `None` for ModelOpt FP4, retaining its prior behavior, while GLM returns the ModelOpt config unless its checkpoint metadata declares the whole NextN layer unquantized.

The published `LibertAIDAI/GLM-5.3-Flash-NVFP4` config was fetched independently. It has 45 text layers, NVFP4 ModelOpt metadata, no whole-layer NextN exclusion, and `*.eh_proj` among its per-module exclusions. Constructing the real `ModelOptFp4Config` from that metadata showed:

- GLM's resolver retains ModelOpt FP4.
- DeepSeek's resolver returns BF16 (`None`).
- `model.eh_proj` is excluded by the real ModelOpt matcher.
- `model.decoder.mlp.experts` is not excluded, so the packed draft experts are constructed on the FP4 path rather than the BF16 path that caused the reported 4096/2048 mismatch.

The candidate's 5 focused tests and 2 adjacent loader tests (6 subtests) pass. There are no native changes, so no native rebuild was required.

## Scope and limitations

This verifies the source-level configuration routing for the checkpoint named in the issue. It is not a full-model validation: weights were unavailable, and the assigned single AMD Instinct MI350X (`gfx950`, ROCm 7.2) cannot run the reported NVIDIA ModelOpt FP4/Marlin SM121 workload or its TP2/EP2 topology. No GPU inference is claimed.

The whole-layer BF16 guard is spelling-sensitive. It recognizes `model.layers.45.*` but not `model.language_model.layers.45.*`, `model.layers.45`, or `model.layers.45*`. Those are counterexamples for other checkpoint metadata and should be hardened separately; they do not occur in the published config from the original report.

The issue also references separate exclusion-list matching work needed for the broader GLM bring-up. This review does not claim PR 1200 fixes those companion issues.
