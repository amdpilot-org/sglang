# Independent review of amdpilot-org/sglang PR 1612

Upstream issue: https://github.com/sgl-project/sglang/issues/34448

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1648

Candidate: https://github.com/amdpilot-org/sglang/pull/1612 at exact commit
`2a6bd87dc7b815d722007836a68ecfa7efc7a386`.

## Recommendation

Accept the candidate as an issue-specific source fix, with the architecture and
full-model qualification limits below. The recorded base reproduces the original
mechanism: after MXFP4 Triton post-processing, `w13_weight` is absent from the
module state and therefore cannot be exported. The candidate retains all four
runtime weights/scales as registered Parameters, makes non-contiguous save
snapshots contiguous on CPU, and prevents CPU offload from replacing storage held
by the external Triton wrappers.

The candidate regression passed. An independent test on the assigned MI355X
verified GPU pointer aliasing, non-contiguous safetensors serialization, data
visibility after checkpoint copy, incomplete-key detection, and mixed offload
behavior. No remaining source-level counterexample was found within the tested
contract.

`fully_resolves_original` is nevertheless recorded as false because the original
GH200/NVIDIA `triton_kernels`, gpt-oss-20b, TP=2 export/reload was not executable
here. This is a qualification limitation, not evidence of a known residual bug.
The available gfx950 run validates storage and transport, not NVIDIA swizzle or
kernel execution, model semantics, checkpoint-size parity, or distributed load.

## Evidence

- `raw/base_environment.log`: prepared interpreter, repository import paths, ROCm
  version, and assigned GPU architecture.
- `raw/base_regression.log`: failing-before run at recorded base `358c163...`;
  `w13_weight` is absent and the offload guard is absent.
- `raw/candidate_regression.log`: candidate's two registered regressions pass at
  exact commit `2a6bd87...`.
- `raw/adversarial.log`: independent real-gfx950 storage/serialization/offload
  boundary cases pass.
- `raw/upstream-related-pr.json`: related upstream PR 34558 was still open during
  review and changes the same four source/test files.

No native source changed, so no native rebuild was applicable. All imports used
`/job/repo/python`; the interpreter was
`/tmp/amdpilot-repo-j-1ba09882bc14/venv/bin/python`.
