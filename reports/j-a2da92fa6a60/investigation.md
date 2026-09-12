# PD circuit-breaker investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/31206

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2277

The reported deployment combined engine nightly `b94ac87e` with the older
`sgl_model_gateway` v0.2.4 router. At prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, the production source already
contains the required fail-whole-request behavior:

- `sgl-model-gateway/src/routers/http/pd_router.rs::select_pd_pair` selects the
  prefill leg before the decode leg.
- `pick_worker_by_policy_arc` filters candidates with `Worker::is_available`.
- `Worker::is_available` requires both health and
  `circuit_breaker().can_execute()`.
- The shared policy filter independently applies the same breaker gate.

The added integration tests exercise the actual Axum router with one mock
prefill and one mock decode worker. They verify the reported state directly:
the prefill remains healthy, its breaker is forced open, and `/generate`
returns 503 within a three-second client bound. They also verify that a
recovered half-open prefill is dispatched again and closes after two successful
requests, plus the symmetric open-decode boundary.

For a failing-before check, both existing breaker filters were temporarily
changed to health-only. The decisive test then failed with `left: 200`,
`right: 503`, demonstrating the old decode-only outcome. Those source mutations
were reverted; only regression tests and this report remain.

This is deterministic router coverage, not a model, GPU, Mooncake, or
multi-node reproduction. Raw command output is retained beside this report.
