# Correction generation 1

Candidate: https://github.com/amdpilot-org/sglang/pull/1639 at `aae252eb457f1f0358ae4b98897f3a8fa6eda50d`

Independent review: https://github.com/amdpilot-org/sglang/pull/1727

The duplicate-payload counterexample was independently reproduced on the exact candidate. The consolidated correction preserves its scheduler and HiCache changes, records the last applied immutable decision so only a byte-for-byte-equivalent duplicate is accepted, and rejects divergent or older replays immediately.

Source inspection also reproduced the router omission: the candidate retried every retryable response from a PD attempt, including responses returned after an upstream might have accepted the request. The correction ports the narrow retry-safety marker and no-auto-replay behavior tracked by upstream router PR 34570, with unit boundary coverage and an integration assertion that a failing decode worker receives exactly one generate request.

Python protocol regression evidence passes after the change. Rust execution remains explicitly unverified because the prepared image contains neither `cargo` nor `rustc`; see `result.json` and `evidence/` for commands and raw output.
