# DeepSeek-V4 MXFP4 candidate correction generation 2

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1217 at
`74ec83a71f7217b9c4af342c749ca37e9c477a63`

Independent review: https://github.com/amdpilot-org/sglang/pull/1312

Upstream issue: https://github.com/sgl-project/sglang/issues/37342

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1357

## Result

The review's direct-loader counterexamples reproduce. The candidate fixture
constructed `FusedMoE` with hand-sized destination tensors, so the generic
shard copier accepted `(2048, 1024)/(2048, 64)` and `(2048, 2048)/(2048, 127)`
just as it accepted the reported `(2048, 2048)/(2048, 128)` pair.

The consolidated test retains the candidate's host-independent backend
selection fix, but replaces the hand-sized loader fixture with buffers made by
the production DeepSeek-V4 FlashInfer CUTLASS MXFP4 quantization method. For
`hidden_size=4096`, production creation yields packed weight width 2048 and
native E8M0 scale width 128. It loads and checks the reported tensors for all
four TP ranks. Boundary cases show that the review's widths do not match that
4096-hidden contract, while `1024/64` is a valid relationship for a different
aligned model with `hidden_size=2048`.

This is a test correction, not evidence that the original full checkpoint
failure has been reproduced or cleared. No production source change was
justified from the available environment.

## Evidence

- `candidate-counterexamples.txt`: the exact candidate accepts all three
  hand-sized pairs.
- `corrected-tests.txt`: eight tests pass, including four TP ranks, two
  counterexample boundaries, another valid hidden size, and backend selection.
- `sm90-suite.txt`: the real FlashInfer SM90 preprocessing/kernel suite skips
  because this host is AMD gfx950, not NVIDIA SM90.
- `gpu.txt`: a real gfx950 float32 matmul agrees with its CPU reference. This
  only establishes GPU execution and does not qualify the Hopper path.

## Remaining limitations

DeepSeek-V4-Flash-0731 weights and four H800/SM90 GPUs are unavailable. Full
checkpoint loading, actual FlashInfer Hopper preprocessing/kernel execution,
distributed TP=4, DSPARK, serving, and semantic accuracy remain unverified.
