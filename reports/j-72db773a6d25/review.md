# Independent review of amdpilot-org/sglang PR 2422

- Upstream issue: https://github.com/sgl-project/sglang/issues/30781
- Candidate: https://github.com/amdpilot-org/sglang/pull/2422
- Candidate commit: `c89391648105d687cd6591fb2d501a7ab3c2c5a3`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **request changes**

## Finding

The candidate fixes enum-level deserialization for the 12 tool type strings currently listed by Python, and its focused tests pass. It does not fully fix the reported proxy contract because the gateway deserializes and then reserializes `ResponsesRequest` before forwarding it. The vendored Rust `ResponseTool` does not faithfully represent Python's tool object.

For the issue's exact custom tool payload, the candidate accepts the request but serializes the tool as only `{"type":"custom"}`, dropping `name` and `description`. An independent namespace boundary case similarly loses `name`, `description`, and nested `tools`. Thus the backend does not receive the request the client sent. This changes an explicit router rejection into silent request corruption and is only a partial fix.

The candidate's regression misses this because it checks deserialization and `Validate`, but never checks the serialized body sent by `route_typed_request`.

## Source and dependency evidence

- `server.rs` extracts `ValidatedJson<ResponsesRequest>` for `/v1/responses`.
- `routers/http/router.rs::route_responses` passes that typed request to `route_typed_request`, which serializes it for the backend.
- On the candidate, `cargo tree -i openai-protocol` resolves `openai-protocol v1.0.0` from `/job/repo/sgl-model-gateway/vendor/openai-protocol`, including direct and transitive gateway users. The tested build therefore used the candidate's vendored source rather than the registry crate.
- The Python contract at `/job/repo/python/sglang/srt/entrypoints/openai/protocol.py` includes `name`, `description`, `parameters`, `strict`, and nested `tools` on `ResponseTool`.
- No C/C++, HIP, or FlyDSL source changed; no separate native rebuild applies. Cargo rebuilt the changed Rust crate and gateway.

## Environment and limitations

The review host is x86_64 and exposes one AMD `gfx950` GPU. GPU execution was not used because the defect is CPU-only Rust JSON deserialization/serialization. No GLM-5.2 weights or four-node TP8 deployment were available, so the reporter's full serving topology was not reproduced. A live model/backend is unnecessary to establish the deterministic request-corruption counterexample, but downstream model/tool semantics remain untested.

The prepared checkout exactly matched the recorded base before review. The initially prepared image did not expose Cargo on PATH; an isolated Rust toolchain was installed and Cargo target artifacts were kept under `/tmp/amdpilot-repo-j-72db773a6d25/cargo-target`.
