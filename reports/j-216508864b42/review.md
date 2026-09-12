# Independent review of PR 1691

Candidate: https://github.com/amdpilot-org/sglang/pull/1691 at `fac8ec3773db9c151b20afd6cff8133a9f64bea2`

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1724

## Verdict

Request changes. The candidate is a substantive partial fix, not test-only hardening: it adds parsing to the non-streaming HTTP worker path used by the issue and corrects the earlier gRPC `think_in_prefill=false` initialization bug. Its focused regressions pass from a freshly built Rust test binary. However, it only treats JSON `null` or a missing field as unparsed. A worker response with `reasoning_content: ""` is preserved as if already parsed, leaving `work</think>answer` in `content`. That violates the same gateway-side parsing contract and is a remaining counterexample.

## Evidence

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that revision, `Router::route_chat` directly returns `route_typed_request(..., "/v1/chat/completions", ...)` with no reasoning-parser processing. This is the original HTTP failure mechanism: worker-side parsing disabled means the raw worker message and null reasoning field pass through unchanged.

At the exact candidate commit, Cargo rebuilt the gateway library test binary from `/job/repo/sgl-model-gateway` into the private target directory `/tmp/amdpilot-repo-j-216508864b42/cargo-target`. The candidate's Qwen3 prefill tests passed (2/2), HTTP payload tests passed (2/2), and the thinking-toggle boundary test passed (1/1).

An independent temporary test supplied `{"content":"work</think>answer","reasoning_content":""}` to the candidate's HTTP payload parser. It expected `reasoning_content == "work"` and `content == "answer"`; instead the candidate retained the empty reasoning string and raw content. The companion explicit `<think>work</think>answer` case passed. The temporary tests were removed before returning to the review branch; their patch and raw output are retained in `/job/review-evidence-j-216508864b42/`.

The source cause is the candidate's guard that skips every non-null `reasoning_content`, including an empty string. Preserving a non-empty worker-produced reasoning value is correct, but empty is another common representation of “not populated” and must not suppress gateway parsing.

## Commands

```text
CARGO_TARGET_DIR=/tmp/amdpilot-repo-j-216508864b42/cargo-target /job/.cargo/bin/cargo test --manifest-path sgl-model-gateway/Cargo.toml --lib qwen3_prefill_reasoning -- --nocapture
# 2 passed

CARGO_TARGET_DIR=/tmp/amdpilot-repo-j-216508864b42/cargo-target /job/.cargo/bin/cargo test --manifest-path sgl-model-gateway/Cargo.toml --lib http_chat_response -- --nocapture
# 2 passed

CARGO_TARGET_DIR=/tmp/amdpilot-repo-j-216508864b42/cargo-target /job/.cargo/bin/cargo test --manifest-path sgl-model-gateway/Cargo.toml --lib test_reasoning_parser_start_follows_thinking_toggle -- --nocapture
# 1 passed

CARGO_TARGET_DIR=/tmp/amdpilot-repo-j-216508864b42/cargo-target /job/.cargo/bin/cargo test --manifest-path sgl-model-gateway/Cargo.toml --lib review_http_ -- --nocapture
# 1 passed, 1 failed: empty reasoning_content was not parsed
```

## Limitations

The reporter's Qwen/Qwen3.8-27B-FP8 weights and NVIDIA L40S environment were unavailable. The assigned host exposes an AMD/ROCm software environment (`torch 2.11.0+rocm7.2`, HIP 7.2), not the reported CUDA architecture. No model inference or GPU claim is made. The deterministic review validates the Rust HTTP/gRPC parsing logic and native compilation only. HTTP streaming is explicitly still passthrough in the candidate and was not validated as fixed. The candidate was not modified or merged.
