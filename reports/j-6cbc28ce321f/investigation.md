# DSpark context-boundary investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33454

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1898

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared checkout reproduced the source-level overflow exactly: a request at
sequence length 1,048,574 with a six-token verify window generated positions
1,048,574 through 1,048,579, four of which are outside a 1,048,576-row RoPE
table. The trace is retained in `raw/before_reproduction.txt`.

Current-related-change review found upstream PR
https://github.com/sgl-project/sglang/pull/33963, an open, unreviewed attempt at
the same issue. The prepared base did not contain that change. That proposal
uses `req_to_token_pool.max_context_len` as its positional bound, but the
prepared implementation deliberately allocates that table with speculative
headroom beyond `model_config.context_len`; it therefore is not the RoPE-table
boundary. This correction instead uses the model context length and is limited
to the DSpark planner and worker in the current checkout.

The worker now intersects the configured/planned verify length with both the
remaining context and remaining generation budget. When clipping is necessary,
it uses a compact ragged target-verification layout so clipped tokens are not
verified or accepted. The fixed-width draft pass remains shape-compatible; its
unused tail repeats the final legal position so no RoPE lookup can cross the
context boundary.

The regression was run before implementation and failed because the clamp was
absent (`raw/regression_before_fix.txt`). After implementation, it and the
existing DSpark scheduler/ragged tests pass. Two gfx950 checks exercised the
tensor clamp and the actual ragged-layout prefix-sum kernel.

The DeepSeek-V4-Flash-0731 weights and the two NVIDIA B300 GPUs from the report
were not available. Therefore this does not claim a full-model, TP2, CUDA crash
reproduction or semantic-accuracy validation. The assigned single AMD MI350X
(gfx950) was used only for the boundary arithmetic and ragged-layout kernel.
