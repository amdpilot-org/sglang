# Streaming ASR word-revision investigation

The defect was reproduced in the prepared checkout at base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. `StreamingASRState.update()`
classified every character-prefix extension as append-only. Consequently,
changing a confirmed `cat` token to `caterpillar` emitted the character suffix
`erpillar`. `_record_emit()` also unconditionally inserted a space between
non-empty deltas, producing `hello world ,` for a punctuation-only suffix.

Upstream PR https://github.com/sgl-project/sglang/pull/35296 was inspected before
editing. It is still open and directly addresses the source issue. The prepared
`main` contains related punctuation-aware `needs_space()` and normalization
support, but does not contain that PR's boundary-aware state transition.

The correction uses the existing `needs_space()` boundary policy to distinguish
true append-only growth from an in-word revision, and to join the monotonic
prompt accumulator without separating punctuation. A whole-word prefix shrink
is retained as a temporary rollback instead of replacing `confirmed_text`, so a
later recovery does not re-emit already delivered words.

This is deterministic CPU string-state logic. No GPU kernels, model weights,
native extensions, or compiler behavior are involved. Qwen3-ASR model inference
and live HTTP/realtime server execution were not performed; the regression
directly exercises the shared state object used by those paths.
