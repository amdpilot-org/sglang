# Investigation of SGLang issue #37548

Source issue: https://github.com/sgl-project/sglang/issues/37548

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2678

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The prepared `main` checkout already contains the issue-specific fix, merged as
part of upstream PR https://github.com/sgl-project/sglang/pull/36507 in commit
`cdfc224b0e5bd0fb5d10334d787c9bd3865ba1d9`. No additional production-code
change is justified.

The issue's later root-cause report corrects the original TP-shard hypothesis.
The invalid IDs are multimodal placeholders (`MM_PAD_SHIFT_VALUE + hash`), not
ordinary global vocabulary IDs. During draft prefill, the target has already
created `ForwardBatch.mm_input_embeds` for those placeholders. Current
`DeepseekModelNextN.forward` uses those embeddings and invokes
`embed_tokens` only for each request's appended final token. Current
`EagleWorkerV2._draft_extend_for_prefill` also rotates the multimodal embeddings
with the input IDs so they remain aligned.

## Before/after regression evidence

The checked-in regression is
`test/registered/unit/models/test_deepseek_nextn_mm_embed.py`. I temporarily
replaced only the fixed embedding-selection block with the reported old direct
`self.embed_tokens(input_ids)` behavior and ran the test. It failed because ID
`1000006` reached an embedding with vocabulary size `154880`. After restoring
the checked-out implementation, both the multimodal regression and the
independent no-multimodal fallback case passed.

Raw logs:

- `raw/regression_before_fix.log` (exit 1; one failure, one pass)
- `raw/regression_after_restore.log` (exit 0; two passes)
- `raw/current_regression.log` (exit 0; initial confirmation)

## GPU boundary evidence

`gpu_embedding_boundary.py` ran on the assigned AMD Instinct MI350X
(`gfx950:sramecc+:xnack-`) with ROCm 7.2. It used two request segments containing
interior sentinel IDs `1000003` and `1000009`, selected final positions 2 and 5,
and embedded only valid IDs 13 and 19. The GPU results matched an independent
CPU weight-table gather exactly (`rtol=0`, `atol=0`). See
`raw/gpu_embedding_boundary.log` and `raw/gpu_inventory.log`.

This is targeted GPU evidence for the corrected embedding boundary, not a claim
of full GLM-5.3-Flash serving reproduction.

## Limitations

The reported model weights and the original 8x NVIDIA H20 TP8/DP8/EP8 topology
were unavailable. Therefore I did not reproduce a full model request, CUDA graph
execution, distributed scheduling, semantic output, or the original NVIDIA
device-side assertion. The available single gfx950 cannot qualify those aspects.
No native C++ or FlyDSL source changed, so no native rebuild was applicable.
