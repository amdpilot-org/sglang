# Independent review of PR 1145

Candidate reviewed: `803c48fc6fa1441b4e79ccd83de0bb207361947e`

Upstream issue: https://github.com/sgl-project/sglang/issues/36822

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1181

Recommendation: **request changes**.

The recorded base reproduced four focused failures: BF16 scale parameters were
allocated as FP16, the TP=2 act-order/non-act-order shapes were reversed, and
the loader lacked a full-table mode.  The candidate's submitted regression
passes 5/5 and fixes allocation dtype and shapes.

However, an independent test through `FusedMoE._weight_loader_impl` shows the
full-table metadata is not forwarded for GPTQ group scales.  The candidate
passes `load_full_w2` only in the ModelOpt-specific `"weight"` branch. GPTQ
`w2_scales` enters the later generic `GROUP` scale branch, which calls
`_load_model_weight_or_group_weight_scale` without the flag. On TP rank 1 the
test loaded rows 2-3 into rows 0-1 and left rows 2-3 uninitialized instead of
copying the full table. Thus the second correctness issue remains in the real
dispatch path, while the direct `_load_w2(..., load_full=True)` unit test does
not cover it.

No native source changed, so a native rebuild was not applicable. Imports were
confirmed from `/job/repo/python/sglang`. The host has one AMD Instinct MI350X
(`gfx950`) with PyTorch 2.11.0+rocm7.2. The reported CUDA Marlin kernel, RTX
4090 architecture, two-rank execution, model weights, serving failure, and NaN
logits could not be exercised. No GPU kernel execution is claimed.

Raw evidence was preserved outside revision switches under
`/tmp/amdpilot-repo-j-a579d508a308/review-evidence/`.
