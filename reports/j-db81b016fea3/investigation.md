# Null optional config correction

Upstream issue: https://github.com/sgl-project/sglang/issues/37846

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3249

Candidate PR: https://github.com/amdpilot-org/sglang/pull/964

Independent review PR: https://github.com/amdpilot-org/sglang/pull/3247

The candidate at `40c96033f4d6c04788d573231d7ffaa259d49da9` was applied unchanged
to the prepared base and tested before any source correction. Both review
counterexamples reproduced: Muse Glimmer called `.items()` on a null
`text_config`, and ModelOpt FP8 called `.get()` on a null `quantization`
section. The raw failing run is in `raw/candidate_failing_before.log`.

The correction treats a null Muse Glimmer text section like an absent section
while retaining all unrelated top-level values. It also routes a null ModelOpt
quantization section through the same descriptive `ValueError` used for an
absent section. Regression coverage compares null and missing inputs and checks
the valid nested ModelOpt format remains accepted.

No GPU execution or native rebuild was applicable: these paths only parse
Python dictionaries. Model weights were unavailable, so no full-model serving
claim is made.
