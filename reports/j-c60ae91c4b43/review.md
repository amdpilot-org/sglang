# Independent review of PR 1421

Candidate reviewed: `7ca83d8652bac65ff85abac25501043711f5b451`

Upstream issue: https://github.com/sgl-project/sglang/issues/35295

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1459

## Recommendation

Request changes. The candidate is a real partial fix, not test-only hardening: it fixes the reported `cat` to `caterpillar` word revision, fixes comma/period spacing, preserves normal append behavior, and passes its regression plus the related focused suite. It does not fully implement the issue's punctuation-only growth contract.

The boundary decision delegates to `needs_space()`, whose `_NO_SPACE_BEFORE` set covers selected punctuation only. Common punctuation absent from that set is misclassified as a word revision. For example, `hello world` to `hello world…` emits `world…` and records `hello world world…`; the expected punctuation-only delta and prompt prefix are `…` and `hello world…`. Closing quote (`"`) and em dash (`—`) have the same failure.

## Evidence

The image-prepared checkout was clean, on `amdpilot/j-c60ae91c4b43`, and exactly at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference from the requested failing-before revision.

On that base, using `PYTHONPATH=/job/repo/python` and the prepared interpreter, the source import resolved to `/job/repo/python/sglang/srt/entrypoints/openai/streaming_asr.py`. The original reproduction returned `the cat`, then `erpillar`, with `emitted_text == "the cat erpillar"`. The punctuation reproduction returned `,` but recorded `emitted_text == "hello world ,"`.

At the exact detached candidate SHA, the same source import path was confirmed. The candidate's four committed state tests passed. The related ASR suite passed 31 tests. Independent cases passed for the reported word revision, comma growth, period growth, normal word append, and temporary rollback, but failed for closing quote, Unicode ellipsis, and em dash punctuation-only growth.

Raw command output and the independent executable test are preserved outside the revision-switching checkout under `/job/review-evidence-j-c60ae91c4b43/`:

- `base-reproduction.txt`
- `candidate-regression-suite.txt`
- `candidate-adversarial.txt`
- `adversarial_streaming_asr.py`

## Scope and limitations

This change touches Python only, so no native source changed and no native rebuild was applicable. The deterministic state transition does not require GPU execution; no GPU claim is made. Review ran on x86_64 with Python 3.12.3 and the prepared ROCm Torch environment. It did not run Qwen3-ASR model inference, use model weights, or launch live HTTP SSE/realtime servers. Direct state testing is sufficient to reproduce the original shared-state defect, while the focused existing serving tests provide integration regression coverage without qualifying model semantics or a live transport deployment.
