# Reasoning effort investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37912

Mirror issue: https://github.com/amdpilot-org/sglang/issues/836

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365` still pinned
`openai-protocol = "=1.0.0"`. A temporary regression using the actual
`ResponsesRequest` type failed on `none` before handler execution:

```text
tier none should deserialize: unknown variant `none`, expected one of
`minimal`, `low`, `medium`, `high`
```

A second temporary regression showed `responses_to_chat` discarded an existing
`minimal` effort (`left: None`, `right: Some("minimal")`). The retained raw logs
are in the private runtime directory named in `result.json`.

The latest crates.io release was independently queried on 2026-09-12. Version
`1.13.0` still defines only `Minimal`, `Low`, `Medium`, and `High` in
`src/responses.rs`. The extracted release source is retained at
`/tmp/amdpilot-repo-j-54df984c6e53/openai-protocol-1.13.0/`.

The issue's git-dependency fallback was also tested at the exact smg #2410 merge
commit `73b1b2c779a8e088b00987c4fea6a0be24d612fd`. It is not a compatible drop-in:
compilation fails across the gateway because that protocol revision has broader
API changes (including missing `ResponsesGetParams`, `ResponseToolType`, and
`worker_spec`). Therefore this change does not claim full seven-tier Responses
support.

The delivered code fixes the independently verifiable gateway-local portions:

- Chat Harmony maps `none`/`minimal`/`low` to Low,
  `high`/`xhigh`/`max` to High, `medium` to Medium, and unknown strings to Medium.
- Responses-to-Chat preserves all four effort tiers representable by the pinned
  protocol.

Full `/v1/responses` acceptance of `none`, `xhigh`, and `max` remains blocked on
a protocol release compatible with this gateway snapshot (or a separately
maintained compatible backport of the protocol crate).
