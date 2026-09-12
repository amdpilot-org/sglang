# DeepSeek-V4 MXFP4 candidate correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/37342

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1155

Candidate: https://github.com/amdpilot-org/sglang/pull/1027 at
`91e812d9014eed3e8d68ce05b1a87c8bc477ed95`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1116.

## Result

The candidate's backend-selection test is valid and is retained.  Its parent
fails because the nominal SM100 case leaks the physical ROCm platform; the
candidate passes for mocked SM90, SM100, and SM120.

The candidate did not exercise routed-expert loading.  This correction adds a
CPU-deterministic test of the reported single-expert weight shape
`[2048, 2048]`, scale shape `[2048, 128]`, and every TP=4 rank.  It verifies
that the intermediate dimension is sharded to 512 rows, the packed hidden
dimension and its 128 scale groups remain intact, and the two projections land
in FlashInfer's required `[up; gate]` W13 order.  All four ranks pass against
the candidate's unchanged production loader.

Consequently, the review's missing-fixture claim is confirmed, but the stronger
claim that this concrete loader contract still fails is not reproduced.  No
production change is justified from the available hardware and weights.

## Evidence

- `evidence/selection-before.txt`: candidate-parent selection regression fails.
- `evidence/selection-after.txt`: retained candidate selection regression passes.
- `evidence/loader-contract-after.txt`: four exact-shape TP rank cases pass.
- `evidence/sm90-test.txt`: Hopper test skips because the assigned GPU is AMD.
- `evidence/gpu-check.txt`: a real gfx950 matmul matches its CPU reference; this
  proves GPU availability only, not FlashInfer/Hopper correctness.
- `evidence/history.txt`: exact candidate parent/diff and relevant release-to-main
  history.

## Remaining limitations

The DeepSeek-V4-Flash-0731 checkpoint and four H800/SM90 GPUs were unavailable.
Full checkpoint loading, FlashInfer SM90 preprocessing/kernel execution, TP=4
distributed execution, DSPARK, serving, and semantic accuracy remain
unverified.  The deterministic fixture validates tensor loading and sharding,
not CUDA kernel compatibility.  Missing architecture or weights were not used
to justify a speculative source change.
