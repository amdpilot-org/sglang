# Independent review of amdpilot-org/sglang PR 923

Upstream issue: https://github.com/sgl-project/sglang/issues/37912

Mirror issue: https://github.com/amdpilot-org/sglang/issues/959

Candidate reviewed: `9e190f3070c97e9aa680919ed29d96ddf08239af`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

Request changes. The candidate is a useful partial fix, but it does not resolve
the original seven-tier contract. It fixes the gateway-local Harmony clamping
and forwards the four tiers representable by the pinned protocol. It leaves
`openai-protocol = "=1.0.0"` unchanged, so `/v1/responses` still rejects
`none`, `xhigh`, and `max` while deserializing the request, before a handler can
run.

This is not merely an absent test: an independent candidate-side regression
using the production `ResponsesRequest` type failed on `none` with:

```text
unknown variant `none`, expected one of `minimal`, `low`, `medium`, `high`
```

The resolved crate source was
`/tmp/amdpilot-repo-j-803615524856/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/openai-protocol-1.0.0/src/responses.rs`.
Its `ReasoningEffort` enum contains only `Minimal`, `Low`, `Medium`, and `High`.
`cargo tree -i openai-protocol` also resolved version 1.0.0 for the compiled
candidate.

## What was verified

- On the recorded base, an actual `ResponsesRequest` rejected `none` during
  deserialization.
- On the recorded base, `responses_to_chat` dropped `minimal`, producing
  `reasoning_effort=None`.
- At the exact candidate commit, the candidate's four-tier forwarding test
  passed.
- At the exact candidate commit, its seven-string Harmony clamp test and the
  unknown-string boundary test passed.
- At the exact candidate commit, an independent seven-tier deserialization
  test still failed at `none` with the same four-variant error.
- Seven existing focused Responses/parser reasoning tests passed and rustfmt
  check passed.

Raw logs and the preserved candidate diff are under
`/tmp/j-803615524856-review-evidence/`.

## Scope and environment

This PR changes Rust request conversion code only. The candidate was compiled
from `/job/repo/sgl-model-gateway` with Rust 1.90.0; Cargo's target and registry
caches were kept under `/tmp/amdpilot-repo-j-803615524856/`. It changes no
Python extension, C++, FlyDSL, or other native source, so there was no native
library to rebuild and no Python import-path ambiguity to resolve.

The host is x86_64 and exposes the assigned AMD `gfx950` agent. No GPU execution
was performed because the reproduced defect occurs in CPU-side Rust JSON
deserialization and request conversion before model execution. Consequently,
this review makes no model-quality, model-architecture, serving-with-weights,
or distributed-workload claim.
