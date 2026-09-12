# Independent review of PR 1193

Reviewed exact candidate `28c3f842ad97600b92c8fb01c5df6593e3a49202` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: accept. The candidate fully resolves the matcher contract described by the original issue, subject to the end-to-end hardware/model limitations below.

## Findings

On the recorded base, an independent harness reproduced both defects: a GLM fused runtime name did not inherit a checkpoint constituent's exclusion, and `visual...` did not match a `model.visual...` exclusion. The base also reached the generic mixed-shard validator instead of selecting the unquantized linear method.

At the exact candidate, the same harness passed for exact and globbed fused constituents, `model.` normalization, combined `language_model.` plus `model.` normalization, unrelated shard names, similar prefixes, distinct layer indices, and actual `UnquantizedLinearMethod` selection. The candidate's four focused tests and the four existing `is_layer_skipped` tests also passed.

The ordering change is necessary for the reported mixed-precision representation: one excluded checkpoint constituent means the fused runtime layer must be constructed unquantized, rather than rejected by the generic same-precision shard validator. The expansion uses the actual `Glm5NextForConditionalGeneration.packed_modules_mapping`, which includes `fused_qkvbfg_a_proj` and its six checkpoint constituents.

No remaining matcher counterexample tied to the reported contract was found. This is a source fix with regression coverage, not test-only hardening.

## Source, native, and environment evidence

Tests imported `sglang` from `/job/repo/python/sglang/__init__.py` using the required interpreter `/tmp/amdpilot-repo-j-a8ec9549e606/venv/bin/python`. The candidate changes one Python source file and one Python test file (plus its prior report); it changes no C++ or native build input. `repository-environment.json` records `native: null` and `wheel: null`, so a native rebuild was not applicable.

The environment has PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD Instinct MI350X (`gfx950`). The original report used two NVIDIA DGX Spark GB10 devices (`SM121`) with TP2/EP2. The GLM-5.3-Flash-NVFP4 weights were not available. Accordingly, this review verifies the deterministic matcher and construction-method contract but does not claim a 120-shard model load, NVIDIA NVFP4 execution, multi-node serving, or performance.

Raw evidence was preserved outside the checkout under `/tmp/amdpilot-repo-j-a8ec9549e606/review-evidence/` before revision switches. The prepared review branch was restored before this report was added.
