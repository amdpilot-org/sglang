# Independent review of amdpilot-org/sglang PR 1235

Reviewed exact candidate `ddf07cc615936af999def11b6d5417853b6f2da4` against upstream issue https://github.com/sgl-project/sglang/issues/37111 and mirror issue https://github.com/amdpilot-org/sglang/issues/1268, using recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Verdict

Request changes. The candidate is a valid partial correction for a real QSA/NEXTN metadata exception, but it does not fully resolve the original silent semantic-corruption report. It provides neither a passing model-level eager/decode-graph comparison nor a fail-closed guard for the affected Qwen3.8-Flash-Next-NVFP4 QSA + NEXTN graph profile.

On the prepared base, a real `EagleDraftExtendInput(num_tokens_per_req=4)` raises `AttributeError` because `_speculative_max_row_length` reads the nonexistent `draft_token_num`. The candidate's one-line runtime change correctly uses the canonical `num_tokens_per_req`; its focused tests, full QSA test file, independent boundary cases, and three gfx950 GPU kernel/reference comparisons pass.

That exception is not the original issue's observed failure mode. The issue reports HTTP-successful but corrupt 1,024-token punctuation-loop output and loss of ordered markers in an approximately 25K-token prompt on 2x NVIDIA GB10/SM121 with TP2 over RoCEv2. No test in the candidate generates either response, compares eager against graph semantics, or disables the unsafe profile. Source inspection confirms the candidate adds no such guard.

## Environment and architecture limits

The assigned machine has one AMD Instinct MI355X (gfx950), ROCm 7.2, PyTorch 2.11.0+rocm7.2, and no CUDA runtime. It has no second GPU/node, NVIDIA GB10/SM121 execution, TP2/RoCEv2 topology, or Qwen3.8-Flash-Next-NVFP4 weights. The SM121-only QSA test skipped. Therefore the original semantic A/B cannot be reproduced or cleared here. The tiny Llama fixture would only qualify transport/engine execution and was not substituted for the missing model and architecture.

No native source changed in the candidate, `repository-environment.json` specifies no native artifact, and no native rebuild was applicable. The prepared interpreter imported the candidate source directly from `/job/repo/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`.

## Classification

- Canonical speculative-row-width exception: fixed and regression-tested.
- Original 1,024-token punctuation-loop corruption: unverified and unresolved.
- Original approximately 25K ordered-marker corruption: unverified and unresolved.
- Fail-closed behavior for the reported unsafe graph profile: absent.
- Overall: partial fix, not a full original-issue fix and not merely test-only hardening.
