# Independent review of PR 879

Upstream issue: https://github.com/sgl-project/sglang/issues/38013

Mirror issue: https://github.com/amdpilot-org/sglang/issues/906

Candidate: https://github.com/amdpilot-org/sglang/pull/879 at exact commit
`7d454f646cb8f7d1c0778eec666a62ad1c375121`.

Recommendation: **request changes**. The candidate is a substantive partial
parser fix, not a full resolution of the original open issue. At recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, the actual checkout parser
forwarded the reported `arguments`, JSON-string `arguments`, `args`, and
repeated-envelope shapes unchanged. At the exact candidate commit, independent
one-shot and three-character streaming cases normalized those shapes, including
four clean nested envelopes, while retaining correct payloads, a schema-declared
`arguments` property, unknown keys, empty objects, and invalid JSON strings.

The candidate's regression file passed (10 tests, 6 subtests), the complete
pre-existing function-call parser file passed (243 tests, 4 subtests), and the
advertised combined command passed (253 tests, 10 subtests). Imports resolved
to `/job/repo/python/sglang/...`, so these tests exercised the checked-out
source. The candidate changes Python only; no native source or generated native
library changed, so a native rebuild was not applicable.

The original report also includes mid-stream duplication/corruption and a
corrupted quadruple-wrapped example. The candidate intentionally preserves
corrupted inner command text, and neither the named
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` weights nor an equivalent model run
was available to reproduce generation frequency, error-feedback escalation, or
semantic behavior. A tiny Llama fixture would exercise a different model
architecture and cannot close that gap. Accordingly, PR 879's committed
`outcome: fixed` should be narrowed to a partial/candidate-verified parser
hardening claim before acceptance.

One assigned AMD Instinct MI355X (`gfx950`) was visible. GPU execution was not
used because the reviewed change and deterministic reproduction are Python
parser-only; no full-model, vision, distributed, or native compiler claim is
made.
