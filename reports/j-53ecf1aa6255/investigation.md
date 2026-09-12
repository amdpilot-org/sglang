# Seven-tier reasoning effort correction

Upstream issue: https://github.com/sgl-project/sglang/issues/37912

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1074

Parent candidate: https://github.com/amdpilot-org/sglang/pull/923 at
`9e190f3070c97e9aa680919ed29d96ddf08239af`

Independent review: https://github.com/amdpilot-org/sglang/pull/1041

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Reproduction

The candidate was applied exactly to the recorded base. A regression using the
production `openai_protocol::responses::ResponsesRequest` failed before handler
execution on `none`:

```text
failed to deserialize none: unknown variant `none`, expected one of `minimal`,
`low`, `medium`, `high`
```

An independent minimal program pinned to the candidate's exact
`openai-protocol = "=1.0.0"` dependency tested each reviewed counterexample and
reported the same rejection separately for `none`, `xhigh`, and `max`. Raw
outputs are in `candidate-seven-tier-before.log` and
`candidate-three-counterexamples.log`.

## Correction

The candidate's valid Harmony clamping and Responses-to-Chat forwarding changes
are retained. The latest published `openai-protocol` remains 1.13.0 and its
`ReasoningEffort` still contains only four variants. The exact smg #2410 commit
was already shown by the parent candidate to be an incompatible whole-crate
upgrade for this gateway snapshot.

To avoid importing those unrelated incompatible protocol changes, this change
vendors the already-pinned 1.0.0 crate and applies only the upstream
`ReasoningEffort` enum expansion plus `as_str()`. A `[patch.crates-io]` entry
ensures the gateway and its smg dependencies use one identical protocol crate,
avoiding duplicate Rust types. The downloaded 1.0.0 crate archive SHA-256 was
`b365334e1e57a6f57ed932951878b31c1fe9087a90149db6f5d071c3504f6dff`;
all vendored source files other than `responses.rs` were byte-identical to that
archive's extracted source.

The permanent regression deserializes all seven values through the production
request type and verifies exact forwarding to the Chat pipeline. Harmony tests
verify low/high clamping for the newly representable boundary tiers.

## Scope and limitations

This is a Rust request-deserialization and conversion fix. No GPU execution,
model weights, model architecture, semantic model-quality, or distributed
workload was needed or claimed. The assigned host exposes a `gfx950` agent, but
the failure occurs before handler or engine execution. No native source changed,
so no native rebuild was applicable.

The vendored compatibility backport should be removed when a published
`openai-protocol` release containing smg-project/smg#2410 is API-compatible with
the gateway and its pinned smg dependencies.
