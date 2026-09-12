# Kimi-VL packed-attention investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/7433

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2513

The prepared base already contains the issue-specific production fix merged in
https://github.com/sgl-project/sglang/pull/32118. That change removed Kimi-VL's
local SDPA implementation, which allocated a boolean
`[1, total_packed_tokens, total_packed_tokens]` mask, and routed MoonViT through
SGLang's packed `VisionAttention` implementation. The upstream PR reports that
the old path caused a 40.43 GiB single allocation during the Kimi-VL MMMU
nightly test.

This PR does not duplicate that production change. It adds a regression for
the single-image and uneven multi-image boundaries and retains a GPU probe that
compares the actual gfx950 Triton backend with independent per-sequence PyTorch
SDPA. See `gpu_probe.log` and `result.json` for measurements and limitations.
