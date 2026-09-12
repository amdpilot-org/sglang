# GLM-5.3-Flash JPEG data URL investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/741

## Finding

The prepared base contains the `Glm5NextForConditionalGeneration` model and
configuration, but Transformers 5.12.1 does not provide the corresponding
multimodal processor. On the unmodified base, SGLang's `get_processor` returned
a tokenizer-only `TokenizersBackend` for `zai-org/GLM-5.3-Flash`; accessing its
`image_processor` failed with `AttributeError`. Thus the reported JPEG data URL
could not reach a GLM-5.3 vision processor through the normal serving setup.

Upstream PR https://github.com/sgl-project/sglang/pull/36833 independently
identified the same missing processor compatibility. This change ports its
narrow image-processor compatibility layer, adapted there from Hugging Face
Transformers commit `eb4d9e2a64a013bec12289288b85d0b1210ba0aa`, and registers it
for `glm5_next`.

After the change, the identical processor-loading command returns
`Glm5NextProcessor` with `Glm5NextImageProcessor`. A regression decodes a real
JPEG data URL through SGLang and verifies that the GLM processor produces a
finite `(80, 1176)` patch tensor and the expected `1 x 8 x 10` grid. Independent
tests cover minimum-size upscaling, non-square alignment, maximum-token
downscaling, and rejection of an impossible token budget.

## Limitations

The report's private `GLM-5.3-Flash-sglang0907-128k-0907t3` weights and original
Statue of Liberty JPEG were not available. The assigned hardware is one AMD
MI355X (gfx950), not eight NVIDIA H20 GPUs. Consequently, this investigation
does not claim to reproduce the bird output, verify semantic accuracy, or
qualify the original distributed deployment. The deterministic tiny Llama
fixture is text-only and would validate only transport/engine execution, so it
was not substituted for the missing GLM vision model.

Raw commands and outputs are retained in `raw/`.
