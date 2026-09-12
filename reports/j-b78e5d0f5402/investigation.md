# Correction investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2224

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2102 at exact commit `6f9ba912ed5a5a9ff2434f7505e86326bff8c20d`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2188

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Candidate reproduction

The exact candidate was applied without modification before testing. The focused reproduction in `raw/candidate_counterexamples.log` confirmed both review counterexamples:

- `MossVLForConditionalGeneration` had no model-specific dimension hook. Both `linear_fc1` and `linear_fc2` fell through to the generic helper with the top-level multimodal config and raised `AttributeError` before a usable LoRA buffer shape was returned.
- `get_layer_id("model.notdeepstack_merger_list.4.linear_fc2.weight")` returned `4`.

The MOSS-VL implementation independently confirms the required architecture-specific geometry. `MossVLVisionPatchMerger` concatenates the final vision features with every deep-stack feature, so its input width is `hidden_size * spatial_merge_size**2 * (1 + len(deepstack_visual_indexes))`.

## Correction

- Preserve the candidate's Qwen3-VL target normalization, row-parallel classification, deep-stack indexing, and Qwen-specific dimensions.
- Add the corresponding MOSS-VL model hook using its concatenating-merger geometry and delegate ordinary text modules to the shared default helper.
- Require `layers` and `deepstack_merger_list` to begin at a path-segment boundary.
- Add regressions for MOSS-VL dimensions, fallback behavior, the reported false positive, the analogous `otherlayers` boundary, and a valid layer name at the start of a path.

## Limitations

The Qwen3-VL-8B/EditScore and MOSS-VL model/adapter weights were not available, so no full server/model-load or semantic multimodal run was performed. The deterministic tiny Llama fixture cannot qualify either architecture and was not substituted. The GPU check only validates the candidate's Qwen projection arithmetic at the recorded adapter shapes. No native source changed, so a native rebuild was not applicable.
