# Independent review of PR 1892

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1923

Candidate: https://github.com/amdpilot-org/sglang/pull/1892 at `abca41a8c6819ed40de5d9814ea8f05dfdb69042`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a substantive partial fix, not test-only hardening: it adds non-streaming HTTP response parsing, initializes gRPC parsers for template-prefilled thinking, upgrades `reasoning-parser` from 1.0.0 to 1.3.0, and fixes the previously reported empty-string `reasoning_content` counterexample. Its focused native tests and independent payload boundary tests pass.

It does not fully resolve the gateway's HTTP contract. `parse_chat_reasoning_response` immediately returns for `request.is_stream()`, and the HTTP worker branch continues to pass the upstream SSE byte stream straight through. Thus an HTTP streaming response containing generated reasoning in `delta.content` is not split into `delta.reasoning_content` and clean `delta.content`.

## Evidence

On the exact recorded base, a temporary integration regression exercised the actual `OpenAIRouter::route_chat` path against a local mock worker returning `content: "work</think>answer"` and `reasoning_content: null`. The assertion expecting `reasoning_content: "work"` failed; the observed value was `Null`. The temporary test patch and raw failure are retained outside the checkout under `/job/review-evidence/j-b0204c427776/base/`.

At the exact candidate commit, a fresh Rust 1.90 build passed all three candidate HTTP payload tests, including null reasoning, empty-string reasoning, and preservation of non-empty worker reasoning. A temporary independent adversarial test also passed for missing `reasoning_content`, explicit `<think>`, multiple choices, non-string content, and preservation of worker-owned reasoning. The candidate's Qwen3 prefill parser tests passed in both non-streaming and incremental streaming parser modes.

Those parser-level streaming tests do not validate the HTTP transport path. Source inspection at the exact candidate shows the HTTP response parser is skipped for streaming requests and the worker byte stream is forwarded unchanged. That is a concrete remaining counterexample, not merely missing test coverage.

## Environment and limits

The review host is x86_64 with one AMD Instinct MI350X (`gfx950`) and PyTorch 2.11.0+ROCm 7.2. The reporter used an NVIDIA L40S/CUDA deployment. Qwen/Qwen3.8-27B-FP8 weights and its exact tokenizer assets were unavailable, so no full-model, semantic-quality, CUDA, or architecture-equivalence claim is made. The deterministic Rust tests validate gateway response transformation and parser behavior only.

No FlyDSL or C++ source changed. The relevant native artifact was rebuilt from source with the repository-pinned Rust 1.90 toolchain in a private target directory; tests executed the newly built `smg` library. The prepared Python import path remained `/job/repo/python`, but Python code was not involved in this gateway-only change.
