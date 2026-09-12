# Independent review of PR 3498

Candidate: https://github.com/amdpilot-org/sglang/pull/3498

Exact candidate commit: `0e3cc83943b4d0accea9d7a846eefbec3be7f63f`

Upstream issue: https://github.com/sgl-project/sglang/issues/37379

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3488

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3499

## Recommendation

`request_changes`. The candidate is a substantive partial fix, not merely test
hardening: it adds the missing methods to the actual Qwen3.5 causal classes,
delegates from both conditional-generation wrappers, validates rank-specific
non-contiguous attention-layer maps, and fills the tensor and Python-float scale
representations used downstream. Those behaviors passed the candidate tests and
an independent harness.

It does not fully resolve the original issue as written. The exact flat JSON
fixture in the issue (`model_type` plus top-level `scaling_factor`) is rejected
by `QuantParamSchema`, which requires the values under `kv_cache`. The loader
catches the validation failure, logs a generic error, and leaves all attention
scales unset, thereby defaulting to 1.0. The candidate test silently substitutes
a different, nested schema and never exercises the issue's exact fixture.

The candidate's retained `failing-before.log` is also not proof of the original
failure: it records a test-collection `ImportError` for a helper name, rather
than the reported loader `RuntimeError`. This review independently reproduced
the real failure through `load_model_utils.load_kv_cache_scales` on the recorded
base.

## Evidence

- `base-reproduction.log`: at base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`, all four Qwen3.5 classes lack
  `load_kv_cache_scales`, and the real loader utility raises the reported
  `RuntimeError` for `Qwen3_5MoeForConditionalGeneration`.
- `candidate-import-paths.log`: the pinned interpreter imported all changed
  modules from `/job/repo/python`, not an installed SGLang wheel. Torch was
  `2.11.0+rocm7.2`; the assigned GPU was one AMD Instinct MI355X (`gfx950`).
- `candidate-tests.log`: candidate regression plus existing Qwen3.5 pipeline
  tests passed: 10 tests and 6 subtests.
- `independent-adversarial.log`: independently passed dense/MoE causal loading,
  both conditional wrappers through the real loader utility, TP-rank 1
  selection, non-contiguous hybrid attention selection, all four scale fields,
  Triton descale selection, and rejection of wrong model type, missing/extra
  layers, missing TP rank, and wrong dtype. It also preserves the remaining
  counterexample: the issue's exact flat JSON loads no scales.
- `amd-fp8-check.log`: on the assigned MI355X, FP8 E4M3 quantize/dequantize with
  scale 0.5 exactly matched an independent CPU reference. This establishes the
  available device's FP8 arithmetic only; it is not a Qwen3.5 serving run.

## Source and native paths

Reviewed source paths at the exact candidate revision:

- `/job/repo/python/sglang/srt/models/qwen3_5.py`
- `/job/repo/python/sglang/srt/model_loader/weight_utils.py`
- `/job/repo/python/sglang/srt/layers/attention/triton_backend.py`
- `/job/repo/test/registered/unit/models/test_qwen3_5_kv_cache_scales.py`

No C++, FlyDSL, or other native source changed. `repository-environment.json`
records no native build target, so no native rebuild was applicable.

## Environment limitations

The reported Qwen/Qwen3.5-122B-A10B weights, eight NVIDIA L20 GPUs, CUDA, and a
TP=8 reservation were unavailable. Consequently, exact server startup,
distributed rank behavior on L20, and semantic generation remain unverified.
The available host had one AMD Instinct MI355X and ROCm 7.2. The tiny Llama
fixture was not used because it cannot qualify Qwen3.5 architecture or this
model-specific calibration contract.

The prepared checkout exactly matched the requested failing-before commit. At
review time, `origin/main` had advanced to
`bd45cd50ca900dd821f829ca9adfbf9aa3336bda`; the review branch was intentionally
returned to its prepared base before this report was committed.
