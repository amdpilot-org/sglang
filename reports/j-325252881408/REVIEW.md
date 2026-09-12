# Independent review of candidate PR 3457

Recommendation: **accept**. Exact candidate commit `44845eb704d9f9eb67a1d81672c51955ae46d344` fully resolves the original issue as scoped.

The prepared checkout was exactly the required failing base, `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that revision, the real `DetokenizerManager._decode_batch_token_id_output` produced chunks `['A 世', '', '世ab']` for the reported incomplete-UTF-8 sequence; independent client concatenation was therefore `A 世世ab` rather than `A 世ab`.

The candidate changes the recovery-only assignment to preserve the greatest already-emitted offset. This matches the actual contract: `decoded_text_len` is fixed during an incomplete recovery run, while clean commits continue to reset `sent_offset` to the newly committed length. The candidate does not blindly clamp clean, finish, or non-streaming transitions.

At the exact candidate revision, imports resolved to source files under `/job/repo/python/sglang`, not an installed wheel. The candidate regression passed (4 tests), and an independent manager-level suite passed the original sequence plus repeated recovery after a committed prefix, per-item and grouped batched decode paths, unaffected clean requests, newline recovery, completion, and non-streaming materialization. Adjacent stop-trimming coverage passed (8 tests). No remaining counterexample was found.

The source change is Python-only. No C/C++, HIP, CUDA, Rust, FlyDSL, or other native source changed, so there was no candidate native artifact to rebuild. The prepared environment is ROCm (`torch 2.11.0+rocm7.2`, HIP 7.2), but no GPU execution was used: this defect is deterministic CPU-side detokenizer bookkeeping. No HTTP/model-weight smoke was treated as proof. Consequently, model-specific tokenizer behavior, HTTP transport, distributed serving, non-AMD architectures, and semantic model output remain outside this review's direct execution evidence, although none is required to establish the reported offset-state contract.

Evidence is retained in `raw/`, including the failing-base output, exact-candidate import paths and changed paths, candidate regression output, independent adversarial output, and adjacent tests.

Upstream issue: https://github.com/sgl-project/sglang/issues/31598

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3442

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3460
