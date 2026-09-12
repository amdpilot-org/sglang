# Evidence

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared base contained an unconditional assignment in `prepare_mcp_tools_as_functions`:

```rust
obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
```

Failing-before non-streaming assertion:

```text
assertion `left == right` failed: initial request should preserve the client's tool_choice
  left: Some("auto")
 right: Some("required")
test result: FAILED. 0 passed; 1 failed; 86 filtered out
```

Failing-before streaming assertion:

```text
assertion `left == right` failed: initial request should preserve the client's tool_choice
  left: Some("auto")
 right: Some("required")
test result: FAILED. 0 passed; 1 failed; 86 filtered out
```

Passing-after focused results:

```text
test api::responses_api_test::test_non_streaming_mcp_minimal_e2e_with_persistence ... ok
test result: ok. 1 passed; 0 failed; 86 filtered out

test api::responses_api_test::test_streaming_with_mcp_tool_calls ... ok
test result: ok. 1 passed; 0 failed; 86 filtered out

test routers::openai::responses::mcp::tests::initial_tool_choice_defaults_to_auto_when_omitted ... ok
test routers::openai::responses::mcp::tests::initial_tool_choice_preserves_explicit_values ... ok
test result: ok. 2 passed; 0 failed; 392 filtered out
```

Both integration tests capture the actual JSON bodies received by the mock upstream worker and assert exactly two requests: initial `tool_choice="required"`, followed by `tool_choice="auto"` with a `function_call_output`. They also verify the completed MCP call and final assistant message. The streaming output contained `Tool result consumed; here is the final answer.`

Related-change review: upstream issue https://github.com/sgl-project/sglang/issues/31459 remains open and references open candidate PR https://github.com/sgl-project/sglang/pull/31469 at commit `1314fa761f6872e4f1c63912adaca57c91a78b09`. Its source correction matches the issue-specific behavior implemented here; this PR additionally retains direct boundary tests for explicit and omitted initial choices.

Toolchain and build paths:

```text
Rust toolchain: /tmp/amdpilot-repo-j-b7b243505374/rustup (pinned repository toolchain 1.90.0)
Cargo home: /tmp/amdpilot-repo-j-b7b243505374/cargo
Cargo target: /tmp/amdpilot-repo-j-b7b243505374/cargo-target
Source: /job/repo/sgl-model-gateway
```

This is a Rust request-routing change. No FlyDSL/C++ native library was changed or rebuilt, and no GPU execution was relevant or claimed.
