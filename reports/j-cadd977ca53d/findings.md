# Independent review of PR 1378

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1378 at exact commit `485c25c09cd8300e7e11db595ebe1edbe475398b`.

Original issue: https://github.com/sgl-project/sglang/issues/37111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1412

## Finding

The candidate is a valid narrow correction, but it does not fully resolve the original issue. On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real `EagleDraftExtendInput(num_tokens_per_req=4)` causes `QwenSparseAttnBackend._speculative_max_row_length` to raise `AttributeError` because the base reads nonexistent `draft_token_num`. The candidate changes that read to the canonical `num_tokens_per_req`; its focused tests and independent width boundaries pass.

That failure is a metadata-path exception, while the original contract is eager/decode-graph semantic equivalence for Qwen3.8-Flash-Next-NVFP4, including rejection of punctuation loops and preservation of four ordered markers at approximately 25K tokens. The candidate contains no model-level A/B regression and no fail-closed guard for those outcomes. Its passing unit tests therefore establish a partial fix only, not the reported silent-corruption fix.

## Verification

- Recorded-base reproduction: fails as expected with `AttributeError` using the real `EagleDraftExtendInput` class imported from the prepared checkout.
- Candidate focused regression: 5 passed, 37 deselected.
- Independent adversarial widths: real width 4, canonical-versus-legacy conflict, zero width, width 1024, and no-speculation cases all passed.
- Full QSA test file: 41 passed, 1 SM121-only test skipped.
- GPU numerical/kernel-reference subset: 3 passed on the assigned AMD Instinct MI355X (`gfx950`).
- Source imports resolved to `/job/repo/python/sglang/...`; Torch resolved to the prepared ROCm installation.
- Native rebuild was not performed because the candidate changes only Python and report files; there is no native diff to rebuild.

## Remaining counterexamples and limitations

This environment has one AMD Instinct MI355X with ROCm 7.2. It has no NVIDIA SM121 runtime, second GB10, TP2/RoCEv2 topology, or RadixArk/Qwen3.8-Flash-Next-NVFP4 weights. Consequently, the 1,024-token eager/graph comparison, punctuation-loop rejection, approximately 25K-token four-marker comparison, and exact dual-GB10 configuration remain unverified. The deterministic tiny Llama fixture cannot qualify a different model architecture, CUDA graph implementation, semantic accuracy, or distributed topology and was not used as substitute proof.

Recommendation: `request_changes` if PR 1378 is presented as resolving issue 37111. The narrow metadata correction itself is supported, but the PR must not claim full original-issue resolution without the missing model-level evidence or a justified fail-closed guard.
