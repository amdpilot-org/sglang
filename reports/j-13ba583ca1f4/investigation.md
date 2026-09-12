# Investigation: WSL2 multimodal CUDA IPC auto-selection

Upstream issue: https://github.com/sgl-project/sglang/issues/35385

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1389

## Finding

The reported behavior is not present at the prepared base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. No source correction was made.

`handle_multimodal_feature_transport` now keeps CUDA IPC opt-in. For an
otherwise automatic, single-node, multimodal NVIDIA CUDA configuration it
resolves `mm_feature_transport` to `cpu`. Consequently, a WSL2 launch matching
the report but not explicitly requesting CUDA IPC cannot enter the reported
CUDA IPC transport path.

This policy came from merged upstream PR
https://github.com/sgl-project/sglang/pull/34662 on 2026-08-13. Its stated
change was to restore CPU as the single-node default while retaining
`--mm-feature-transport=cuda_ipc` as an opt-in. The issue was later updated on
2026-08-25 with a comment independently observing that current main no longer
reproduced the auto-selection claim.

The later WSL-specific upstream PR https://github.com/sgl-project/sglang/pull/36212
was closed without merge. It addressed explicitly requested CUDA IPC, which is
a different case from this issue's automatic-selection reproduction. Adding
that change here would therefore broaden the task without evidence that the
reported default path remains defective.

## Regression and boundaries

The checked-out source already includes the direct regression
`test_default_transport_is_cpu_for_multimodal_model`. The focused
`TestMultimodalFeatureTransport` suite also independently covers:

- explicit `cuda_ipc` selection on a mocked NVIDIA CUDA platform;
- legacy flag and environment opt-in behavior;
- explicit CPU overriding the legacy environment;
- rejection of CUDA IPC on non-NVIDIA platforms and multi-node deployments;
- CPU fallbacks for text-only, language-only, unsupported multi-node, and
  non-MNNVL configurations; and
- CUDA VMM auto-selection only for the supported multi-node MNNVL/IMEX case.

All 18 focused tests passed. Raw output is retained in
`raw/pytest_multimodal_feature_transport.txt`, and the exact inspected source
and tests are retained in `raw/source_and_test_snapshot.txt`.

## Hardware boundary

The assigned device is one AMD Instinct MI355X (`gfx950`) under ROCm 7.2, not
an NVIDIA GPU under WSL2. A deterministic 2x2 GPU matrix multiplication matched
the independent exact reference, confirming execution on the assigned GPU and
recording its architecture in `raw/gpu_environment_and_numeric.txt`. This does
not exercise CUDA IPC and is not used as evidence that WSL2 CUDA IPC works.

An end-to-end reproduction with the reported RTX 5090, WSL2 kernel, container,
and `RadixArk/Qwen3.8-27B-NVFP4` weights was not possible in this environment.
The tiny Llama serving fixture was not used because it cannot validate WSL2
CUDA IPC or the reported multimodal architecture, and the policy-level defect
was already directly disproved by the checked-out implementation and focused
regression.
