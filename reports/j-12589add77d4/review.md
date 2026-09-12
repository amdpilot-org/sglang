# Independent review of candidate PR 580

Reviewed exact candidate commit `1cccaca71af91d8a88bd2afe4d7dfa7b072e7635` against the activation-semantics contract in:

- Upstream issue: https://github.com/sgl-project/sglang/issues/37609
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/582
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/580

Recommendation: **accept**. The candidate fully resolves the original issue as scoped.

The prepared base reproducibly constructs `Spark2_5MLP` with `approximate="tanh"`. It differs from the Transformers exact-GELU reference by `0.0004737377166748047` on the issue's CPU grid and by up to `0.0005218982696533203` through the real ROCm fused path. The candidate's exact regression fails twice on the base and passes twice at the reviewed commit.

Independent candidate checks exercised float32, float16, and bfloat16 GPU tensors, batched shapes, signs, zeros, large magnitudes, varied multipliers, and NaN/Inf values. Candidate output matched Transformers exactly in those finite dtype cases and agreed with a separately evaluated float64 `erf` formula within dtype rounding. The source change is the direct, platform-independent selection `GeluAndMul("none")` in the actual model constructor.

No native source changed, so rebuilding FlyDSL or sgl-kernel was not applicable. The loaded model and activation modules came from the candidate checkout; the unchanged fused kernel package came from the prepared environment. Testing used one AMD Instinct MI355X (`gfx950`) under ROCm 7.2. CUDA, NPU, XPU, CPU AMX, and full-checkpoint logit parity were not tested. These are architecture or scope limitations, not remaining counterexamples to the reported activation contract.

Raw evidence is retained outside the revision-switching checkout under `/job/review-evidence-j-12589add77d4/`.
