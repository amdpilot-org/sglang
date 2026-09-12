# QSA + NEXTN decode-graph correction generation 2

Candidate https://github.com/amdpilot-org/sglang/pull/1235 was inspected at exact commit `ddf07cc615936af999def11b6d5417853b6f2da4`, together with independent review https://github.com/amdpilot-org/sglang/pull/1294.

## Reproduced correction

On prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real `EagleDraftExtendInput(num_tokens_per_req=4)` raises `AttributeError` in `QwenSparseAttnBackend._speculative_max_row_length` because the implementation reads nonexistent `draft_token_num`. The candidate's one-line use of canonical `num_tokens_per_req` fixes that concrete defect. The failing traceback is retained in `evidence/failing-before.txt`; the focused regression passes after the correction.

The correction and its three boundary tests are preserved. Focused tests report 5 passed, the full QSA file reports 41 passed and 1 NVIDIA-only skip, and three real gfx950 numerical/reference checks pass.

## Remaining review counterexamples

The 1,024-token eager/graph semantic comparison, punctuation-loop rejection, approximately 25K-token four-marker retrieval comparison, and exact 2x NVIDIA GB10 SM121 TP2/RoCEv2 configuration could not be run. This host exposes one AMD Instinct MI350X (`gfx950`) through ROCm 7.2, has no NVIDIA runtime, and has no Qwen3.8-Flash-Next-NVFP4 weights in the prepared private runtime.

The deterministic tiny Llama fixture was not substituted because it can validate transport and engine execution only; it cannot qualify Qwen3.8-Flash-Next semantics, the NVIDIA CUDA graph path, or a distributed TP2/RoCEv2 workload. No broader source guard is added because the unavailable architecture and weights provide no issue-specific evidence defining a correct fail-closed scope.

Accordingly, this candidate remains rejected as a complete resolution of the original silent semantic-corruption report, while its independently reproduced metadata correction is retained.
