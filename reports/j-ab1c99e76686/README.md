# Independent review of PR 3148

Upstream issue: https://github.com/sgl-project/sglang/issues/33035

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3150

Candidate: https://github.com/amdpilot-org/sglang/pull/3148 at exact commit
`7c363ab28fa2e7b4b390cd1105ca2c95119e221b`.

Recommendation: **accept as a conservative partial fix**. The candidate fixes the
specific prior-review counterexample and provides real pre-body bounding for the
supported Python ASGI frontend. It does not fully resolve or verify the original
issue across the reported Kimi-K2.6 deployment, Rust frontend, or EPD
language-only architecture.

The recorded base contains no admission controller. On the exact candidate, the
focused suite passed 12 tests. An independent four-client case with a four-item
budget and four 4 MiB bodies observed only one application `receive()`, three
pre-receive HTTP 503 responses, 4 MiB rather than 16 MiB of retained body data,
and zero accounting after completion.

Imports were verified from `/job/repo/python/sglang`, not an installed stale
copy. No native source changed, so a native rebuild was not applicable. No GPU
execution was used: this is CPU/ASGI accounting, and the available text-only tiny
Llama fixture cannot qualify Kimi-K2.6 multimodal behavior.

The implementation reserves the complete budget until JSON parsing reveals the
true weight. That safely prevents under-accounting but means enabled request-body
admission is conservative: an existing nonzero lease prevents another unknown
body from entering even where the eventual weights might fit together.

Raw outputs are retained under `evidence/`; exact commands and classifications
are in `result.json`.
