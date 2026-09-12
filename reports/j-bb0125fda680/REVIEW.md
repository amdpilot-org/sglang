# Independent review of PR 3167

Candidate: https://github.com/amdpilot-org/sglang/pull/3167 at exact commit `efa4ba5a75b21b2867e55582e7b1b98f7cf2b81f`

Upstream issue: https://github.com/sgl-project/sglang/issues/13809

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3171

## Recommendation

Request changes. The candidate is a valid partial fix for ordinary tensor-parallel constrained decoding, but it does not fully resolve the original unqualified TP feature request.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduces the original failure: a simulated non-entry TP worker calls `get_cached_or_future_value` once and receives a compile `Future`.

At the exact candidate commit, the same ordinary-TP probe makes zero backend lookups and carries `PlaceholderGrammarObject`. A real two-process Gloo call through `Sampler._sync_token_ids_across_tp` also broadcasts divergent tokens `[3, 9]` to `[3, 3]`. The candidate's focused suite passes 97 tests.

The remaining counterexamples also reproduce against the exact candidate implementation:

```text
candidate_speculative_tp {'queued': True, 'compile_lookup_calls': 1, 'grammar_type': 'Future'}
candidate_context_parallel_tp {'queued': True, 'compile_lookup_calls': 1, 'grammar_type': 'Future'}
```

These are direct consequences of `tp_grammar_entry_only` being disabled whenever speculative decoding is active or the attention context-parallel group has world size greater than one. In those configurations, every applicable rank retains the old compilation and masking path. The original issue does not exclude either mode, so the candidate is partial rather than complete.

This review does not recommend simply deleting the guards. Speculative verification builds grammar masks over draft-tree positions from live FSM state, while the candidate's normal sampler broadcast only synchronizes final token IDs. Context-parallel request ownership spans a topology that is not covered by the candidate's attention-TP-only token collective. A complete correction needs architecture-specific ownership and synchronization, plus distributed regression coverage.

## Environment and limitations

The prepared interpreter resolved SGLang from `/job/repo/python/sglang`. Torch was `2.11.0+rocm7.2`, HIP was `7.2.26015`, and the node exposed one AMD Instinct MI350X (`gfx950`). Only one GPU was assigned, so true TP=2/TP=4 GPU serving, speculative TP, and context-parallel TP could not be executed. The two-process Gloo test validates the candidate's ordinary-TP collective call but is not a substitute for distributed GPU serving. No native sources changed, so no native rebuild was applicable. The deterministic tiny Llama fixture cannot create the missing distributed architectures and was not used as substitute evidence.

Raw commands and outputs are retained under `reports/j-bb0125fda680/raw/`.
