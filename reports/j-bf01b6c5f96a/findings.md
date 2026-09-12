# Correction-generation investigation for PR 1076

Upstream issue: https://github.com/sgl-project/sglang/issues/37111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1187

Candidate: https://github.com/amdpilot-org/sglang/pull/1076 at exact commit `3b1f061a7a8c650318803e6ee46f6332557f56f7`

Independent review: https://github.com/amdpilot-org/sglang/pull/1152

## Outcome

`candidate_rejected` as a resolution of the original silent-corruption issue, while preserving its valid narrow fix.

I independently reproduced the draft-extend metadata exception on prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`: a real `EagleDraftExtendInput(num_tokens_per_req=4)` has no `draft_token_num`, so `_speculative_max_row_length` raises `AttributeError`. At the exact candidate, the canonical `num_tokens_per_req` correction passes focused regressions and independent boundary cases. The full QSA file and three GPU reference comparisons also pass.

The review's remaining counterexamples cannot be converted into a justified source correction in this environment. The candidate has no result for either the 1,024-token punctuation-loop gate or the approximately 25K-token ordered-marker gate, and it adds no fail-closed guard. This checkout has one AMD Instinct MI350X (gfx950), while the report requires 2x NVIDIA GB10/SM121, TP2 over RoCEv2, CUDA decode graphs, and Qwen3.8-Flash-Next-NVFP4 weights. CUDA is absent and the SM121-only test is skipped.

A blanket QSA/NEXTN graph rejection would be speculative: the available evidence neither isolates the corrupting condition nor establishes the safe scope of such a guard. The tiny Llama fixture was not substituted because it cannot validate this model architecture, model semantics, CUDA graph implementation, or distributed topology. Therefore this correction generation preserves the proven prerequisite-crash fix and explicitly leaves both model-level semantic counterexamples unverified.

Raw concise evidence is in `evidence/reproduction.txt`.
