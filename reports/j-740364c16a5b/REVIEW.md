# Independent review of PR 3289

Candidate: https://github.com/amdpilot-org/sglang/pull/3289 at `43c2db023064dfbfe823e0147be6b47d3fc06f2b`

Upstream issue: https://github.com/sgl-project/sglang/issues/38580

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3293

Parent candidate: https://github.com/amdpilot-org/sglang/pull/664 at `183f57d6899bbb1df7a273383fa779ad94bffae7`

Prior independent review: https://github.com/amdpilot-org/sglang/pull/3252

## Recommendation

Request changes. The candidate is useful fail-closed and test hardening, and it fixes the previously reported synthetic stale-field acceptance. It does not demonstrate a usable speculative DSA execution after post-plan padding and therefore does not fully resolve the original issue.

The candidate's positive unit fixture fabricates four-row `dsa_seqlens_expanded`, `token_to_batch_idx`, and `indexer_k_start_end`. The actual `DRAFT_EXTEND_V2` planner does not produce that state: it pads `dsa_cache_seqlens_int32` from three to four rows, leaves `dsa_seqlens_expanded` at three rows, and `_cal_indexer_k_start_end()` returns `(None, None)` because draft-extend-v2 is not `is_extend_without_speculative()`. The new validator consequently rejects the actual planner-shaped metadata. This prevents unsafe attention/KV execution, but turns the affected eager fallback into a runtime error rather than establishing either a matched physical plan or an explicitly compact execution contract.

## Evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate's five new focused cases failed. The GPU padding case measured 3 planned rows where the physical attention-TP-aligned extent is 4.
- At exact candidate `43c2db023064dfbfe823e0147be6b47d3fc06f2b`, its focused suite passed: 14 tests.
- Imports resolved to `/job/repo/python/sglang`, not an installed SGLang wheel.
- On AMD Instinct MI350X with ROCm 7.2.26015, the independent actual-planner-shape case measured `padded_dsa_rows=4`, `expanded_rows=3`, `actual_draft_indexer_ranges=None`, and `actual_token_to_batch=None`; the candidate rejected that state.

Raw outputs and the adversarial script are retained in this report directory.

## Limitations

No DeepSeek DSA model weights or qualified end-to-end speculative DSA fixture were available. The tiny Llama fixture cannot qualify this architecture, so it was not used as substitute evidence. Full attention/indexer/DeepGEMM schedule/KV side-effect execution remains unverified. CUDA-only and distributed variants were not exercised. No C++ or FlyDSL source changed, so no native rebuild was applicable; the prepared AITER native module was loaded from the private runtime cache.
