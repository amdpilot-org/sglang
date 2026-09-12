# Correction evidence

Candidate: https://github.com/amdpilot-org/sglang/pull/1564 at
`5e007ae4fb75b5b0b68c0c37a7ffe913239ca402`

Independent review: https://github.com/amdpilot-org/sglang/pull/1630

Upstream issue: https://github.com/sgl-project/sglang/issues/34740

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1664

## Candidate reproduction

The candidate's submitted regression and neighboring stop-trimming suite passed
(`11 passed`). An independent fixture against that exact commit then measured:

```text
oversized steps=32 window=57344 total=131073
oversized steps=256 window=53248 total=1048577
accept_index [[3, 4, 5, 6]]
num_correct_drafts [3]
predict_unique [100]
```

This independently confirms both review counterexamples: the hardcoded token is
unchanged, and an event-count limit is not a small retained-token bound.

## Failing before / passing after

Running the added regressions on base `358c163250ad3b1f62939b01ce1314a0a31a0365`
produced `5 failed, 2 passed`. The two detokenizer failures showed no committed
text for both normal and 4,096-token invalid-byte events. The fixed-token tests
showed that the base had no safe-token resolver or token-id input and retained
the hardcoded behavior.

After the correction, the new regressions and existing stop-trimming coverage
produced `15 passed`. On the assigned gfx950 GPU the actual tensor path resolved
token id 64, whose fixture decode is `a`, and produced `predict_unique=[64]`.

Raw logs are retained at `/job/review-evidence/j-69fb2eb5a4ec/`.

## Limitations

DeepSeek-V4-Pro weights/tokenizer and the reported eight-GPU topology were not
available. No claim is made about that architecture's semantic output, serving
throughput, TTFT, or distributed behavior. No native source changed.
