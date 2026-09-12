# Investigation of sglang#37755

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/916

The reported 2-node, 64-device Ascend 910B deployment and the private
`MiMo-V2.5-Pro-W8A8` checkpoint were not available in this job. Therefore the
reported response-accuracy failure could not be reproduced end to end.

The prepared source already contains MiMo-V2.5 Pro-specific handling for the
weight format implicated by the report:

- `attention_projection_layout="fused_qkv"` derives the checkpoint's expected
  attention TP from `num_key_value_heads` (8 for the published Pro config).
- startup rejects an incompatible effective attention TP;
- `load_mimo_v2_qkv_proj_weight` maps the checkpoint's TP-interleaved fused QKV
  rows onto a smaller compatible runtime attention TP;
- block-quantized `weight_scale_inv` tensors are deferred, dequantized per
  checkpoint shard, de-interleaved to `[all Q; all K; all V]`, and requantized
  for the runtime shard.

The added regression covers those existing paths. Its numerical scale test
also demonstrates that the old/plain concatenation ordering differs from the
required QKV ordering. It passes on CPU and the numerical transformation was
independently run on the assigned gfx950 GPU. This establishes that the current
source has a plausible, internally consistent correction for the reported
format problem, but it does not establish MiMo response accuracy on Ascend.

Raw command output is retained under `raw/`. The initial pytest run is also
retained: its sole failure was an overly strict test tolerance for FP8 e4m3
rounding, corrected from 2% to 3%; it did not reveal a product defect.
