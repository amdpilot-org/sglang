# Independent review of PR 2975

Candidate reviewed: `e5c2189055820624395769913d2b1b94a7c09c56`

Upstream issue: https://github.com/sgl-project/sglang/issues/13809

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2911

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3015

## Recommendation

Request changes. The candidate correctly implements entry-rank-only grammar compilation, entry-rank masking, and token broadcast for ordinary TP constrained decoding when speculative decoding and context parallelism are both disabled. It does not fully resolve the original, unqualified TP feature request: both speculative TP and context-parallel TP explicitly retain per-rank compilation and masking.

## Findings

1. The recorded base reproduces the reported duplicate work. A simulated non-entry TP rank calls `get_cached_or_future_value` and receives its own compile `Future`.
2. At the exact candidate commit, the same ordinary-TP non-entry path performs no backend lookup and carries `PlaceholderGrammarObject`. A real two-process Gloo call through `Sampler._sync_token_ids_across_tp` broadcasts rank 0's token successfully.
3. The candidate's focused regression suite passes (97 tests), including the no-grammar/no-collective and legacy MIN-all-reduce fallback assertions.
4. An independent adversarial case demonstrates the remaining direct counterexample: with `tp_grammar_entry_only` disabled, as the candidate does whenever speculative decoding or context parallelism is active, a non-entry rank again invokes `get_cached_or_future_value` and receives a compile `Future`. Thus those TP configurations retain the original `n x` grammar compilation overhead.
5. The candidate changes only Python sources. Imports resolved to `/job/repo/python/sglang/...`; there was no native source change or native artifact to rebuild.

## Architecture and environment limits

The assigned node exposed one AMD Instinct MI355X (`gfx950`) through ROCm 7.2 and Torch 2.11.0+rocm7.2. A GPU masking/reference calculation executed and matched, but one GPU cannot qualify true TP=2/TP=4 serving, context-parallel TP, speculative TP, DP-attention distributed serving, or an end-to-end TP CPU-overhead benchmark. The two-rank collective validation used Gloo processes and validates collective selection/data movement, not a multi-GPU serving workload. The tiny Llama fixture cannot create the missing multi-GPU architecture and was therefore not used as substitute evidence.
