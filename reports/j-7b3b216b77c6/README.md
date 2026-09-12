# Independent review of recursive model overrides

Upstream issue: https://github.com/sgl-project/sglang/issues/33505

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2095

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2064

Exact candidate commit: `232bc80929e268025dc5a34f8f0f9b414bbfe78b`

## Verdict

Recommendation: **accept**. The candidate fully resolves the configuration-merge
contract that can be evaluated without the unavailable model and NVIDIA serving
environment. It changes the prior one-level merge into a recursive merge across
existing `PretrainedConfig` and dictionary nodes. No native source is changed.

The recorded base already includes the earlier partial repair from PR 1886. On
that base, the complete issue-shaped nested override passes, but overriding only
`text_config.rope_parameters.rope_theta` fails with `KeyError: 'rope_type'` because
the complete `rope_parameters` dictionary is replaced. The exact candidate makes
that focused regression pass, as well as independent text-only and multimodal
cases that preserve siblings through multiple levels and replace scalars, lists,
and previously absent dictionaries at their proper boundaries.

The candidate's committed raw logs contain trailing whitespace, so its report's
claim that `git diff --check` passed is not reproducible. This is evidence/report
hygiene and is not a functional counterexample to the original merge contract.

## Evidence

- `raw/base_partial_rope.txt`: the approved remaining counterexample fails on
  recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- `raw/base_full_issue_shape.txt`: the earlier complete issue-shaped override
  already passes on the prepared base, demonstrating that base is a partial fix.
- `raw/candidate_pytest.txt`: candidate regression suite passes: 6 tests and 2
  subtests.
- `raw/candidate_adversarial.txt`: independent recursive merge cases pass.
- `raw/candidate_import_paths.txt`: confirms SGLang and the changed loader were
  imported from the exact checked-out repository; it also records Torch/ROCm and
  the visible AMD GPU.
- `raw/gpu_architecture.txt`: records the assigned gfx950 / MI350X architecture.

## Limitations

The original `nvidia/Qwen3.6-27B-NVFP4` weights, CUDA 13/NVFP4 stack, two RTX 5090
GPUs, and TP=2 environment were unavailable. The assigned system has one AMD
gfx950 (MI350X) GPU with ROCm 7.2. The reproduced defect and correction are in
CPU-only configuration loading, so no GPU execution was necessary or claimed.
Full server startup, model loading, NVFP4 execution, distributed behavior, and
semantic accuracy remain unverified. No C++ or other native source changed, and
the prepared environment declares no native artifact, so no native rebuild was
applicable.
