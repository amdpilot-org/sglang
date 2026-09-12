# Kimi-K2 candidate correction

This correction was prepared from main commit
`358c163250ad3b1f62939b01ce1314a0a31a0365` after independently testing PR
2519 at exact commit `a8cd992b18403cd79c7b1defed0d6862f91f813c`.

The candidate's valid split-end-marker fix is preserved. The independent
candidate run reproduced the four review observations: full one-character
fragmentation produced no calls and leaked markers, singular K2-Thinking
section markers leaked into content, a 600,000-character unclosed call retained
600,067 characters and emitted no content, and no explicit reset API existed.

The consolidated correction holds every proper special-token prefix (including
a one-character `<`), recognizes and strips singular section markers, releases
an unclosed section after the configurable `SGLANG_KIMI_PARSER_SECTION_MAX`
limit (512 KiB by default), and provides `reset()` to clear all request-scoped
streaming state before explicit detector reuse.

The parser is deterministic CPU string processing. No model weights, HTTP
server, GPU execution, or native rebuild were needed or used. Therefore this
does not claim Kimi model semantic accuracy, full serving-path behavior, or
distributed reproduction. Because the streaming API has no end-of-stream
signal, a final literal `<` is necessarily held until another increment
disambiguates it; it is emitted unchanged when the following text diverges.
