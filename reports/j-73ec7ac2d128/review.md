# Correction generation 2 result

Candidate: https://github.com/amdpilot-org/sglang/pull/896 at `a007eace504e630a2bf4666fa32a78b30db0b282`

Independent review: https://github.com/amdpilot-org/sglang/pull/976

The review's concrete claims reproduce. Before applying the candidate, the actual layout builder produced the reported 42-token/eight-slot failure geometry. The 41, 43, and 47 token layouts are also genuinely ragged. On the assigned GPU, the original DSV4 metadata copy raised the reported overlapping-storage exception.

The candidate's valid fixes are preserved. Capture slots now follow request width, so 42 tokens at width six uses seven equal rows, and DSV4 raw verify/decode metadata copies safely handle overlapping views. The focused suite passes, and both copy paths pass on the real assigned GPU.

The candidate does not resolve the Engram contract for the reviewed 41/43/47 tiers. They remain unequal layouts by design. Fixing that correctly requires the Engram consumer to use `RaggedVerifyLayout` request boundaries when constructing hash context rather than infer equal blocks from token count and request count.

That correction cannot be implemented honestly in this checkout: there is no Engram implementation, model integration, or independent executable reference. Inventing an Engram API or rounding capture tiers to equal blocks would be speculative; the latter would also violate the required compact-ragged semantics. The full DeepSeek-V4.1-Flash first-forward path is unavailable for the same architecture/weights/hardware reason. Raw evidence is retained beside this report.
