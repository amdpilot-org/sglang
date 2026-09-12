# Independent review of amdpilot-org/sglang PR 3414

Reviewed candidate: `c5812a14220bf0d4ebe1cf98cd2c63b818c98aa8`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a substantive partial fix, not test-only hardening, but it does not fully resolve the original contract.

## Finding

The Rust endpoint parser accepts bracketed text without validating that it is IPv6. For `tcp://[not-ipv6]:5557`, Python's canonical `parse_advertisable_tcp` returns `None`, while the candidate Rust parser returns `Some(("[not-ipv6]", 5557))`. With a positive page size and `publisher="zmq"`, Rust therefore emits a non-null `kv_events` descriptor for an endpoint Python treats as malformed.

This contradicts the original requirement to return `null` whenever safe advertisement is unavailable and breaks Rust/Python parity. The independent failing Rust test and raw output are retained in `evidence/adversarial-invalid-bracketed-host.log`; the temporary test was removed before leaving the exact candidate checkout.

## What passed

- The exact recorded base lacks `kv_events` in the Rust serializer, reproducing the original missing-field behavior.
- The candidate's four focused Rust tests pass after compiling the changed crate with pinned Rust 1.92.0.
- A forced release build produced and imported a new PyO3 extension from the prepared repository, not an installed wheel. Its `ServerArgs` constructor includes `kv_events_config`, `page_size`, and `dp_size`.
- The Python `/server_info` route suite passes all 29 tests and 12 subtests.
- The candidate preserves the existing Rust internal-state allowlist and its canary test demonstrates that raw config and API secret values are not serialized.

## Environment and scope

The review ran on x86_64 Linux. The prepared stack reports Torch 2.11.0+rocm7.2 and HIP 7.2. No GPU was used because the changed path is CPU-side parsing/serialization, and no GPU/model numerical claim is involved. The host initially lacked Rust; pinned Rust 1.92.0 was installed in the job-local environment, with build targets and extension caches under `/tmp/amdpilot-repo-j-a64e132cdd59`.

A live model-backed Rust HTTP server was not launched. The rebuilt native extension and Rust serialization logic were directly exercised, and the Python handler was route-tested, but the candidate does not include an Axum route-level regression. KV-event publishing remains intentionally out of scope.
