# GLM-5.3 NextN ModelOpt FP4 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/36653

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1129

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The reported `4096` versus `2048` mismatch is not evidence of a TP sharding
disagreement. `_load_w13` shards the output dimension, while the traceback
reports dimension 1. For ModelOpt NVFP4, that dimension is the packed hidden
dimension (two FP4 values per byte): an unquantized draft parameter expects
hidden size 4096 while the serialized FP4 tensor is packed to 2048.

The checked-out implementation reproduced the configuration loss directly.
`Glm5NextForConditionalGenerationNextN._resolve_nextn_quant_config` retained
`modelopt_fp4` for a checkpoint whose layer 45 is quantized, but
`DeepseekModelNextN.__init__` then unconditionally replaced that configuration
with `None` before constructing `DeepseekV2DecoderLayer`. The failing-before
test captured the decoder receiving `None`.

The correction moves the legacy DeepSeek ModelOpt-FP4 exception to
`DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config`. This leaves the
existing DeepSeek behavior unchanged, while the GLM-5.3 subclass can retain
the quantization configuration when its NextN checkpoint layer is quantized.
The existing GLM ignore-list behavior remains intact for checkpoints that
declare `model.layers.45.*` unquantized.

## Related work checked

Upstream PR https://github.com/sgl-project/sglang/pull/37322 is open and covers
the same central configuration loss, plus substantially broader exclusion-name
remapping for DeepSeek and Bailing checkpoints. It is not present in this base.
The narrow change here does not duplicate those unrelated remapping changes.

## Validation and limitations

Raw command output is retained in `raw/`. The focused regression failed before
the source change (two failures: quantized GLM decoder configuration was lost,
and the intended legacy DeepSeek resolution boundary was not centralized) and
passes after it. The related multimodal NextN embedding tests also pass.

The assigned device is AMD `gfx950`, while the report is for NVIDIA ModelOpt
NVFP4 on two `sm_121` nodes. The 320B checkpoint weights and two-node NVIDIA
topology were not available. Therefore no full checkpoint load, TP=2 server,
GPU numerical execution, architecture qualification, or semantic generation
claim is made. This is a source-level candidate verified by deterministic unit
coverage, not a full model reproduction.
