# prompt_cache_key investigation

The prepared base silently discarded `prompt_cache_key`. The failing-before
schema output is in `raw/failing-before-schema.txt`. The implementation now
declares the OpenAI field and maps it to the existing engine `cache_salt` when
the SGLang-native value is absent.

Focused CPU tests pass with schema, forwarding, precedence, absent-key, and
empty-key coverage. The final GPU probe used the deterministic fixture from
amdpilot-org/sglang PR 649 at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Generated weights and helper
scripts remained in `/tmp/amdpilot-repo-j-743166142ea1/`; the source checkout
was `/job/repo`. Because that tokenizer has no chat template, the serving run
used a private minimal Jinja chat template without changing the weights.

In `raw/gpu-http-final`, four identical 44-token `/v1/responses` inputs with
keys A/A/B/B returned HTTP 200 and reported cached-token counts 0/43/0/43.
This is transport and engine/cache evidence on one MI355X (`gfx950`), not a
Qwen3.8 hybrid, semantic-quality, large-prompt latency, or distributed result.

Original issue: https://github.com/sgl-project/sglang/issues/37263

Mirror issue: https://github.com/amdpilot-org/sglang/issues/995
