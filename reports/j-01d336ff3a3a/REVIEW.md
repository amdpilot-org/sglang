# Correction generation 1: DeepSeek V3.2 refactor

Upstream issue: https://github.com/sgl-project/sglang/issues/16255

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3077

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2959

Independent review PR: https://github.com/amdpilot-org/sglang/pull/3045

## Result

The exact candidate commit `642e52a8e98ede176a3c4942bd18f926b40b3434`
was inspected and its valid V3.2 indexer/top-k-policy extraction was retained. The
review's concrete source counterexample was reproduced: the candidate had no
`deepseek_common.hardware_backend` package, and NPU MHA, MLA, and DSA imports and
dispatch remained in `deepseek_v2.py`.

This correction adds a dedicated NPU attention mixin, moves the six NPU
prepare/core entry points behind it, and imports the NPU implementation lazily so
non-NPU hosts do not acquire an NPU runtime dependency. Mock-based tests cover
MHA, MLA, and DSA dispatch, including the DSA-only previous-top-k argument.

## Evidence

- `evidence/candidate-npu-mixin-before.log`: exact candidate import fails because
  the proposed hardware-backend package does not exist.
- `evidence/focused-after.log`: 22 tests passed, 8 subtests passed.
- `evidence/pre-commit-after.log`: all applicable repository hooks passed.
- `evidence/gpu-device-after.log`: on one AMD Instinct MI350X, the existing V3.2
  empty top-k helper matched an independent Torch allocation for device, shape,
  and dtype.

## Remaining limitations

No NPU was available, so the NPU tests validate routing and call signatures with
mocks, not kernels or runtime compatibility. No DeepSeek V3.2 checkpoint or
weights were available; real Indexer/IndexerKPool execution, checkpoint
compatibility, logits, and semantic accuracy remain unverified. No multi-GPU or
pipeline-parallel stage boundary was run. NVIDIA, CPU-only, MUSA, and other
architecture-specific paths were not executed. Dedicated AMD and CPU mixins from
the original illustrative folder proposal were not added speculatively: current
main already has separate ROCm and CPU attention-forward mixins, and this
environment did not establish a concrete behavioral defect requiring wrapper
classes or permit qualification of those architectures. No native source changed.
