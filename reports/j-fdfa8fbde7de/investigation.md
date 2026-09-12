# Investigation notes

- Upstream issue: https://github.com/sgl-project/sglang/issues/31475
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2234
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- The base retained full-strength `shared_output` additions in both
  `DeepseekV2MoE.forward_normal` and `forward_normal_dual_stream` after the
  local all-reduce skip decision.
- Existing upstream PR https://github.com/sgl-project/sglang/pull/31476 proposes
  the same narrow pre-scaling correction. Broader upstream PR
  https://github.com/sgl-project/sglang/pull/35535 hoists shared-expert work
  around additional CP/DP paths; neither was merged into the prepared base.
- Raw test and GPU outputs are retained beside this file.
