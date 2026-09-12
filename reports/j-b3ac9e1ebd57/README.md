# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/31533

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2230

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, the XGrammar
factory passed only `model_config.hf_eos_token_id` to XGrammar. Request finish
detection also accepts `tokenizer.eos_token_id` and
`tokenizer.additional_stop_token_ids`, so the two components disagreed about
which tokens may terminate generation.

The real XGrammar matcher reproduction in `raw/xgrammar_mask_reproduction.log`
shows that Qwen2.5's tokenizer EOS (`<|im_end|>`, 151645) is masked after the
regex accepts `17` when only the model-style EOS (`<|endoftext|>`, 151643) is
provided. Supplying their union unmasks both tokens.

The correction unions model and tokenizer stop tokens once, when constructing
the XGrammar backend. It preserves `None` when no EOS information is available,
allowing XGrammar's existing tokenizer auto-detection. Regression tests cover
the reported mismatch, additional stop tokens, deduplication, and the existing
empty-input behavior.

Upstream PR https://github.com/sgl-project/sglang/pull/31534 proposes the same
core correction but remained open during this investigation; it was not
present in the prepared base.

## Limitations

The private Qwen3.5-MoE checkpoint and NVIDIA B200 environment from the report
were unavailable. No full-model HTTP serving reproduction was claimed. The
deterministic test exercises the actual installed XGrammar matcher and the
unit regression exercises the checked-out SGLang factory path. No GPU kernels
or native sources are involved in this fix, so the assigned gfx950 GPU was not
used. Per-request `sampling_params.stop_token_ids` remain outside the grammar
matcher and are intentionally not addressed.
