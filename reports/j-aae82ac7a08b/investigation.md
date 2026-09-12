# Streaming ASR final punctuation correction

Upstream issue: https://github.com/sgl-project/sglang/issues/35295

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1718

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1593

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1685

The review claims were reproduced independently at exact candidate commit
`41d3846aea50aee5a478156a0797d6a7d3fa35dd`. After an incremental update had
emitted `hello world`, all six reviewed punctuation-only final transcripts
duplicated `world`. The comma failure was also reproduced through
`process_asr_chunk(..., is_last=True)`, the helper called by HTTP chunked
streaming and realtime.

The candidate's valid fixes were retained: complete-word emission for word
revisions, punctuation attachment without an extra space, Unicode punctuation
classification, normal append spacing, and monotonic handling of temporary
rollback. The consolidated correction applies the same boundary-aware
append/rollback classification during `finalize()` before falling back to
word-level revision handling.

The focused regression passes for comma, quote, ellipsis, em dash, `?!`, and
fullwidth closing parenthesis through direct finalization. It also passes the
comma case through the production helper. Independent cases verify a final
`cat` to `caterpillar` revision still emits the complete revised word, and a
final punctuation rollback does not repeat text.

This is deterministic Python string-state logic. No model weights, GPU
execution, or native rebuild were needed or used. Consequently this work does
not claim Qwen3-ASR semantic accuracy or live HTTP/realtime transport
validation. The existing whitespace-free CJK rollback limitation remains.
