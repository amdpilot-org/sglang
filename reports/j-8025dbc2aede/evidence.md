# Abort request router evidence

- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Private Rust/Cargo home: `/tmp/amdpilot-repo-j-8025dbc2aede/{rustup-home,cargo-home}`
- Native build target: `/tmp/amdpilot-repo-j-8025dbc2aede/cargo-target`
- Raw logs and HTTP captures: `/tmp/amdpilot-repo-j-8025dbc2aede/evidence`
- Patched native binary: `/tmp/amdpilot-repo-j-8025dbc2aede/cargo-target/debug/sgl-router`

The base binary returned `404` for `POST /abort_request`. After rebuilding,
the same route returned `200`, and the recording worker logged:

```text
/abort_request Bearer transport-test {"rid":"repro-request","abort_all":false,"future_field":{"x":1}}
```

GPU execution was not applicable: this patch changes only Rust HTTP fan-out
and makes no model-output or kernel claim.
