# Correction generation 2: Qwen preprocessing save/load review

This investigation reviewed candidate PR
https://github.com/amdpilot-org/sglang/pull/3235 at exact commit
`4b29cbe7f919d23dd82ff76f65257a85a72889a2` and independent review PR
https://github.com/amdpilot-org/sglang/pull/3277 against prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Result

The independent review's four concrete counterexamples were reproduced against
the candidate implementation:

- the same trusted image content ID with a different rendered prompt produces
  a different key and invokes preprocessing twice;
- a fresh processor instance cannot see an entry retained by the first;
- there is no explicit OpenAI save operation or caller-selected load operation;
- Qwen video and audio requests return no request-cache key and use the normal
  preprocessing path.

The candidate remains a valid, bounded optimization for repeated identical
Qwen image requests. Its four regression tests pass, including single-flight
coalescing and defensive copies. Those source, test, and documentation changes
are preserved here. They do not complete the original feature request.

No further source correction is claimed. Prompt-independent reuse requires a
validated split between Qwen media preprocessing and prompt-dependent token,
offset, and mRoPE construction. Cross-worker/restart reuse additionally needs a
service-level persistent store and lifecycle/API contract. No Qwen-VL weights
were available, and the deterministic tiny Llama fixture cannot validate those
Qwen-specific semantics. Making either change from mocks alone would be
speculative.

## Reproduction

Candidate regression:

```bash
/tmp/amdpilot-repo-j-f0b2f6628d2a/venv/bin/python -m pytest -q \
  test/registered/unit/multimodal/test_qwen_preprocess_request_cache.py
```

Result: 4 passed. See `raw/candidate-regression.log`.

The independent adversarial harness is retained in `raw/adversarial.log`; its
measured values were:

```text
same_content_id_prompt_keys_equal False
preprocess_calls_across_prompt_change 2
fresh_processor_preprocess_calls 1
video_cache_key None
audio_cache_key None
```

Compatibility suite:

```bash
/tmp/amdpilot-repo-j-f0b2f6628d2a/venv/bin/python -m pytest -q \
  test/registered/unit/multimodal/test_qwen_preprocess_request_cache.py \
  test/registered/unit/multimodal/test_media_artifact_processor.py \
  test/registered/unit/multimodal/test_preprocess_cache.py \
  test/registered/unit/managers/test_mm_hashes.py \
  test/registered/unit/parser/test_jinja_template_utils.py \
  test/registered/unit/entrypoints/openai/test_protocol.py
```

Result: 115 passed and 30 subtests passed. `compileall` and `git diff --check`
also passed. One AMD Instinct MI350X (`gfx950`) was visible, but no GPU execution
was used as evidence because qualifying Qwen-VL weights were unavailable. No
native source changed, so a native rebuild was not applicable.

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3282
