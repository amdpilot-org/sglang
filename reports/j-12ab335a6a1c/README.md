# Independent review of PR 1027

Candidate: `91e812d9014eed3e8d68ce05b1a87c8bc477ed95`

Upstream issue: https://github.com/sgl-project/sglang/issues/37342

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1059

Recommendation: **request changes**.

The candidate is useful test-only hardening for a host-dependent model-override
unit test. On the prepared ROCm base the old test fails because its nominal
SM100 mock leaves the real HIP flag enabled; the candidate explicitly mocks
`is_hip=False`, adds SM90/SM120 positive cases, and passes.

That is not a regression for the original failure. The issue reports an
`AssertionError: Hidden size mismatch` while loading routed-expert weights
through an already-selected `flashinfer_mxfp4` path. The candidate changes no
loader, quantization, MoE runner, kernel, or native source and never constructs
the reported expert tensors or reaches the failing assertion. Its attribution
to commit `60ff1e33` establishes automatic backend selection, not successful
weight loading after selection, and therefore does not prove the issue fixed.

The exact DeepSeek-V4-Flash-0731 weights and four H800/SM90 GPUs were not
available. The assigned device is one AMD Instinct MI350X (`gfx950`); the SM90
FlashInfer test skipped. Full checkpoint loading, TP=4, DSPARK, serving, and the
relevant NVIDIA kernel remain unverified.

Raw command output is retained under `evidence/`.
