# Independent review of PR 1836

Upstream issue: https://github.com/sgl-project/sglang/issues/38210

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1878

Candidate: https://github.com/amdpilot-org/sglang/pull/1836 at exact commit
`91a601b36576761643cbf825833165f168af3ca7`.

Recommendation: **accept**.

The recorded base commit reproduced the reported destructive serialization
boundary in both synchronous and asynchronous tracing: after an enabled trace
context was materialized in a process without a tracer/exporter and serialized
again, an initialized destination received a disabled context. The candidate
preserves the original wire state in relay-only processes, and an initialized
destination reconstructs an enabled, valid context.

This is the issue's actual boundary rather than a neighboring smoke test.
`MultiTokenizerRouter` and `MultiDetokenizerRouter` transparently receive and
send request objects through the pickle IPC helpers, without consuming or
altering their trace contexts. Independent tests exercised three consecutive
uninitialized relays, final initialized reconstruction, span/link state, and
genuinely disabled contexts for both implementations. The candidate's focused
suite also passed all 58 tests.

No native source changed, so a native rebuild was not applicable. Imports were
confirmed to resolve from `/job/repo/python/sglang`. The assigned GPU is an AMD
Instinct MI355X (`gfx950`), but this review did not execute a model or collect a
live end-to-end OTLP serving trace. The conclusion is based on deterministic
coverage of the exact pickle relay contract and inspection of the production
router/IPC path; it does not qualify another model architecture, semantic
accuracy, non-pickle IPC, or a distributed workload.

Raw evidence is retained outside the checkout under
`/tmp/amdpilot-repo-j-b2a330db286d/evidence/`.
