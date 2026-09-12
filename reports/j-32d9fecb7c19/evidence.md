# Payload-size feature evidence

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Source paths:

- `sgl-model-gateway/src/main.rs`
- `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py`
- `sgl-model-gateway/src/server.rs`
- `sgl-model-gateway/tests/routing/payload_size_test.rs`
- `sgl-model-gateway/tests/common/mock_worker.rs`

Native build path: `/tmp/amdpilot-repo-j-32d9fecb7c19/gateway-target`

Pinned compiler: Rust 1.90.0 (`1159e78c4 2025-09-14`), selected by `sgl-model-gateway/rust-toolchain.toml` and installed under `/tmp/amdpilot-repo-j-32d9fecb7c19` because the prepared image had no Rust compiler on `PATH`.

## Failing-before regression

The first run of `cargo test --test routing_tests configured_limit_overrides_axum_default -- --nocapture` sent a 3 MiB JSON body through a gateway configured for 4 MiB and received 413 instead of 200. Investigation showed the gateway accepted and forwarded the request, but the in-test mock worker independently applied Axum's 2 MiB default. The mock now disables that unrelated default so the regression measures the gateway boundary. The unchanged production gateway already applies both `DefaultBodyLimit::max(max_payload_size)` and `RequestBodyLimitLayer`.

## Passing validation

- Native CLI environment tests: 3 passed.
- Gateway payload integration tests: 5 passed.
- Python isolated parser checks: passed.
- Rust format check: passed.
- Git whitespace check: passed.

GPU execution was not applicable and was not performed.
