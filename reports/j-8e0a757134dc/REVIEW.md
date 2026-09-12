# DeepSeek V4 modern config correction generation 2

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2950 at exact
commit `8ae54e54289174deeb7f13f3f306126f430ca108`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3009.

## Reproduction

All three concrete review counterexamples reproduced against the exact
candidate before source changes. The `deepseek_v4` registry entry and
`get_config` both resolved to SGLang's `_DeepseekV4ConfigAlias`, whose parent is
Transformers' `DeepseekV3Config`. A valid Transformers-created config with
`layer_types` containing `heavily_compressed_attention` but a `compress_rates`
mapping containing only `compressed_sparse_attention` normalized to `[0, 4]`.
Nested `rope_parameters` containing `main` but no `compress` normalized to an
empty dictionary. Raw output is retained in
`/job/evidence-j-8e0a757134dc/candidate/`.

## Correction

The correction preserves the candidate's modern/legacy normalization and
per-layer RoPE selection, while selecting Transformers' native
`DeepseekV4Config` when that class is available. Older Transformers releases
without the native class retain the prior V3-derived compatibility fallback.
Dictionary compression rates must now cover every referenced layer type, and a
nested RoPE configuration must contain a non-empty active `compress` section.

The focused suite passes with 12 tests and 4 subtests. An independent
passing-after probe confirmed the native class is registered and loaded, both
malformed configurations raise `ValueError`, and a valid generated config
still normalizes to `[128, 128, 128, 4]` with one C4 indexer layer. Raw output
is retained in `/job/evidence-j-8e0a757134dc/fixed/`.

## Limitations

No DeepSeek V4 weights were available, so weight loading, full engine startup,
GPU numerical comparison, semantic accuracy, and distributed serving remain
unverified. The deterministic defects corrected here are CPU-side parsing and
validation behavior. The pinned environment uses Transformers 5.12.1, which
provides the modern native V4 schema. No native source changed, so a native
rebuild was not applicable.
