# Investigation report: GLM-5.3-Flash repeated `!` output

Upstream issue: https://github.com/sgl-project/sglang/issues/36669

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1125

## Outcome

`candidate_verified`: the prepared base already contains the only current
issue-specific maintainer candidate, merged upstream as PR #36143. No duplicate
source change was made.

The reporter used revision `d6ab04bdf157d80aff9e850535921c58adace116`.
At that revision, both TensorRT-LLM fused all-reduce call paths selected
rank-by-rank BF16 accumulation (`fp32_acc=False`). The prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365` instead defaults the fused
all-reduce/residual/RMSNorm path to FP32 accumulation and forces FP32
accumulation for the allreduce-only path when the selected backend is TRT-LLM.
The MNNVL boundary remains independent and does not receive the TRT-only
argument.

This is issue-specific because the upstream maintainer explicitly asked the
reporters to test PR #36143 for the SM90 failure, and that PR describes the
TP=8 numerical divergence caused by seven intermediate BF16 roundings in the
TRT-LLM backend. Raw upstream metadata and the exact source patch are retained
under `raw/`.

## Validation

- The current repository regression file ran 21/21 tests successfully on the
  assigned AMD Instinct MI350X (`gfx950`). It verifies that TRT-LLM receives
  `fp32_acc=True`, MNNVL does not receive the backend-specific argument, and
  includes workspace/shape/dtype/group boundary cases.
- An independent GPU calculation used eight cancellation-sensitive BF16
  contributions. Rank-by-rank BF16 rounding returned `1` versus the FP64
  reference `4`; FP32 accumulation followed by a BF16 cast returned `4`.
  This validates the numerical mechanism only, not the NVIDIA kernel or model.
- The first unittest invocation used a non-package module path and exited 1
  before running tests. It is retained in the test claims; the corrected direct
  file invocation exited 0.

## Limitations

The assigned system has one AMD gfx950 GPU, not 8 NVIDIA H20 (SM90) GPUs, and
the `zai-org/GLM-5.3-Flash` native FP8 weights are not available. Therefore the
original TP=8/EP=8 serving request, TRT-LLM CUDA kernel, and repeated-`!` model
output were not reproduced here. The deterministic tiny Llama fixture would
only test transport/engine execution and cannot qualify GLM-5.3 architecture,
FP8 numerics, SM90 collectives, or semantic output, so it was not substituted
for the reported workload.

## Evidence

- `raw/reporter_revision_fp32_search.log`: old revision source locations and
  `fp32_acc=False` values.
- `raw/upstream_pr36143_files.json`: exact merged candidate file patch from
  GitHub's API.
- `raw/upstream_pr36143_metadata.json`: candidate rationale and merge metadata.
- `raw/focused_flashinfer_tests.log`: passing current regression and boundaries.
- `raw/gpu_numeric_evidence.log`: independent gfx950 numerical reference.
