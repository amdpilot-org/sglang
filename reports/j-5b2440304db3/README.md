# GPT-OSS tokenizer investigation

Source issue: https://github.com/sgl-project/sglang/issues/31271

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3444

## Result

The report is reproducible with the public `openai/gpt-oss-20b` tokenizer at
revision `6cee5e81ee83917806bbde320786a8fb61efebee` and the prepared
Transformers 5.12.1 environment. `AutoTokenizer` and SGLang both return the
runtime class `TokenizersBackend`. With `trust_remote_code=True`, current main
performs the fallback and logs:

```
Tokenizer for openai/gpt-oss-20b is still TokenizersBackend after retries with
--trust-remote-code. Model-specific tokenizer attributes may be missing.
```

This is a class-name compatibility warning, not measured functional loss.
Transformers 5.12.1 exposes `PreTrainedTokenizerFast` as an alias whose actual
class name is `TokenizersBackend`, and GPT-OSS explicitly declares
`"tokenizer_class": "PreTrainedTokenizerFast"`. Both trust settings retained
`get_vocab`, `encode`, `decode`, `apply_chat_template`, `chat_template`, and
`backend_tokenizer`. Encoding `Hello, GPT-OSS!` produced
`[13225, 11, 174803, 12, 26496, 0]`, decoded identically, and the vocabulary
contained 200019 entries.

Upstream PR https://github.com/sgl-project/sglang/pull/34451 already addresses
this exact issue by recognizing generic declared tokenizer classes. I inspected
it before making a change and did not duplicate the fix. Applying its source
change to the recorded base eliminated the warning and the unnecessary retry
for both `trust_remote_code=False` and `True`, while preserving the attributes,
IDs, round trip, and vocabulary size above. Its tests also preserve fallback
behavior for a specifically declared class and for missing tokenizer config.

## Evidence

- `raw/current_gpt_oss_retry_probe.txt`: current main, one resolver invocation
  for each trust setting and the warning with trust enabled.
- `raw/gpt_oss_tokenizer_probe_before.txt`: full public-tokenizer class,
  attribute, encode/decode, vocabulary, and chat-template probe.
- `raw/candidate_gpt_oss_probe.txt`: PR 34451 source applied locally; zero
  resolver invocations for both trust settings and unchanged functionality.
- `raw/upstream_pr_34451.diff` and `raw/upstream_pr_34451.json`: inspected
  candidate change and metadata.
- `raw/related_pr_search.json` and `raw/related_issue_search.json`: related
  change search output.

## Limitations

The reporter's private merged-LoRA checkpoint was not available and was not
accessed or reconstructed. Its tokenizer metadata may differ from the public
GPT-OSS asset. The exact B200/CUDA 13 serving reproduction was unavailable in
this ROCm environment, and model weights were intentionally not downloaded;
therefore full server startup, model architecture execution, merged-LoRA
behavior, and B200 behavior remain unverified. This tokenizer-only issue did
not require GPU execution or a native rebuild. A deterministic tiny Llama
server fixture would validate only transport/engine execution and cannot
qualify GPT-OSS tokenizer or architecture behavior, so it was not substituted
for the requested reproduction.
