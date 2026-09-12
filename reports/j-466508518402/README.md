# Consolidated correction for DP-attention request broadcast

- Upstream issue: https://github.com/sgl-project/sglang/issues/37590
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1120
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/990 at `d184ccfe3578f7cb75647e364570107d971030d9`
- Independent review PR: https://github.com/amdpilot-org/sglang/pull/1085
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The candidate's CP-first ordering was reproduced failing on global rank 1, which
is the unseeded source of CP group `[1,3]`. The consolidated fix retains the
useful CP-before-TP direction but only lets the attention-TP rank-zero column
participate in CP. After that column is seeded from the combined leader, every
CP row can safely broadcast along TP.

The committed four-process Gloo reproducer passes and shows the request payload
on all ranks. The original eight-GPU DeepSeek-V4 serving command remains
unverified because this job had one gfx950 GPU and no model weights.
