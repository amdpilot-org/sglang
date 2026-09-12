# DeepSeek V4 Transformers 4.57+ compatibility correction

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/2771 at
`192a5253d3268bf9199d6854344eb01b10577510`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2839.

## Reproduction

All four reported counterexamples were reproduced against the exact candidate.
An installed Transformers 5.12.1 `DeepseekV4Config` serialized a dictionary
`compress_rates`, parallel `layer_types`, `hash_moe` entries, and nested
`rope_parameters.main` / `rope_parameters.compress`. The candidate forced that
file through its V3-derived parser and failed with
`StrictDataclassClassValidationError` on `hash_moe`.

An independent direct case showed that the candidate returned the rates
dictionary unchanged as `compress_ratios`, integer indexing raised `KeyError`,
`get_num_indexer_layers` returned zero, and the complete nested RoPE object was
returned as though it were one flat scaling configuration. Raw outputs are in
`/job/evidence-j-411a310890d3/candidate/`.

## Correction

The correction leaves the installed Transformers parser in control and
normalizes the two schemas at SGLang's runtime boundary. Dictionary rates are
expanded in `layer_types` order, legacy list fields remain supported, lengths
and conflicting aliases are validated, and the legacy per-layer runtime alias
is populated. Nested RoPE remains intact while the runtime selects `main` for
uncompressed layers and `compress` for compressed layers, including their
separate theta values.

The generated-config after case completed `ModelConfig` construction with
ratios `[128, 128, 128, 4]`, one C4 indexer layer, and the active compressed
RoPE subsection. Focused tests and pre-commit both pass; raw logs are under
`/job/evidence-j-411a310890d3/`.

## Limitations

No DeepSeek V4 weights were available, so weight loading, full engine startup,
GPU numerical comparison, semantic accuracy, and distributed serving remain
unverified. The deterministic failures and fixes exercised here are CPU-side
configuration and runtime wiring. The pinned stack uses Transformers 5.12.1,
whose generated schema implements the 4.57+ contract. No native source changed,
so a native rebuild was not applicable.
