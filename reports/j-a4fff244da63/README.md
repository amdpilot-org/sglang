# Recursive model override correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33505

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2017

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1886

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1976

The review counterexample was reproduced against the candidate's unchanged loader
implementation: overriding only `text_config.rope_parameters.rope_theta` raised
`KeyError: 'rope_type'` because `PretrainedConfig.update()` replaced the complete
`rope_parameters` dictionary.

The correction recursively merges overrides into existing `PretrainedConfig` and
plain-dictionary nodes. Scalars and dictionaries for previously absent fields still
use replacement semantics. The candidate's issue-shaped complete rope override and
its independent replacement boundaries are retained in the expanded regression.

## Evidence

- `raw/pytest_before.txt`: focused regression fails on the recorded base/candidate
  implementation with `KeyError: 'rope_type'`.
- `raw/pytest_after.txt`: complete focused loader suite passes (6 tests and 2
  processor subtests).
- `raw/gpu_environment.txt`: records the assigned gfx950 hardware. The tests are
  configuration-only and did not execute on the GPU.

The original Qwen weights, NVIDIA NVFP4 stack, and two RTX 5090 devices were not
available. No claim is made about full serving startup, NVFP4 execution, TP=2, or
model semantics; those are unnecessary for and independent of the reproduced
CPU-side dictionary merge defect.
