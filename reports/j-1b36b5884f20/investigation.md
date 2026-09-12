# Streaming ASR punctuation-growth correction

Upstream issue: https://github.com/sgl-project/sglang/issues/35295

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1541

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1421

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1504

The candidate was reproduced at exact commit
`7ca83d8652bac65ff85abac25501043711f5b451` before any correction. Its
word-revision and comma fixes worked, but closing quote, Unicode ellipsis, and
em dash growth each emitted the complete last word again and duplicated that
word in `emitted_text`.

The candidate used `needs_space()` both to render boundaries and to decide
whether a character-prefix extension was punctuation-only. That helper has an
intentionally curated spacing table, so it cannot recognize every Unicode
punctuation character. The correction uses Unicode general categories to
identify a non-empty punctuation-only suffix for state transitions and prompt
accumulation, while retaining `needs_space()` for ordinary word and CJK
boundaries.

The regression covers all three reviewed counterexamples, a multi-character
punctuation suffix, and punctuation rollback. The candidate's reported word
revision, comma, normal append, and word rollback coverage remains intact.

No GPU, model weights, live server, or native rebuild was needed or used. This
directly validates the deterministic shared Python state used by HTTP SSE and
realtime paths, but does not qualify Qwen3-ASR model semantics or live
transport behavior. The documented whitespace-free CJK rollback limitation is
unchanged.
