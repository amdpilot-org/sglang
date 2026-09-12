# Independent review of amdpilot-org/sglang#3437

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3437

Exact commit: `760ef21de6b20abcf4ee851c4e895d1c9467d512`

Upstream issue: https://github.com/sgl-project/sglang/issues/33055

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3434

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3438

Candidate lineage reviewed for context:
https://github.com/amdpilot-org/sglang/pull/3414 at
`c5812a14220bf0d4ebe1cf98cd2c63b818c98aa8`, and its independent review
https://github.com/amdpilot-org/sglang/pull/3433.

## Recommendation

**Accept.** The candidate fully resolves the original issue's additive Rust
`/server_info.kv_events` contract on the evidence available in this x86_64 CPU
environment. It is a source/native fix, not test-only hardening: typed
`ServerArgs` plumbing reaches the Rust serializer, valid ZMQ/TCP metadata is
emitted, unsafe configurations become JSON `null`, and the existing allowlist
continues to exclude raw configuration and secret canaries.

The original base at `358c163250ad3b1f62939b01ce1314a0a31a0365` was the
image-prepared checkout and contains no `kv_events` member in the Rust
serializer. The exact candidate adds it and passes both its regression tests
and an independently added (then removed) adversarial matrix. The prior
counterexample `tcp://[not-ipv6]:5557` independently returns `None` in Python
and `null` in candidate Rust; valid bracketed IPv6 remains advertised.

## Evidence

- `candidate-focused-rust.log`: all five candidate Rust serializer/parser
  regressions passed.
- `independent-review-tests.patch` and `candidate-adversarial-rust.log`: a
  temporary review-only test exercised port boundaries, malformed bracket and
  IPv6 forms, scheme rejection, malformed JSON types, and null serialization.
  It passed and was removed before leaving the candidate revision.
- `candidate-python-reference.log`: the same endpoint/schema matrix against
  the Python reference, including the reported malformed bracket case.
- `candidate-python-server-info.log`: the real Python handler suite passed (29
  tests and 12 subtests).
- `candidate-native-rebuild.log`: a forced release rebuild from the candidate
  checkout succeeded and the resulting PyO3 `_server` extension was imported
  from the private runtime cache.

## Limitations

No GPU was used. The reviewed behavior is CPU-only endpoint parsing and JSON
serialization; model architecture, weights, numerical kernels, and distributed
execution are not involved. The host is x86_64 Linux with Rust 1.92.0. A live
model-backed Axum server was not launched, and KV-event publishing itself was
not exercised because both are outside the original metadata-only contract.
The deterministic tiny-Llama fixture would only add transport/engine smoke
coverage and cannot qualify this serializer/parser behavior.
