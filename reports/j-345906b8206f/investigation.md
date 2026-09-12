# Candidate correction investigation

Candidate: https://github.com/amdpilot-org/sglang/pull/2453 at `8b1804ecee27f2c877529f501f53d79a5c8ef1fc`

Independent review: https://github.com/amdpilot-org/sglang/pull/2535

Upstream issue: https://github.com/sgl-project/sglang/issues/30314

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2546

The exact candidate passed its original ten-test suite. Its deterministic cache
fixture also confirmed the review counterexample: with `N` concurrent requests,
`2N` slots are consumed by request-owned states and admission-locked matched
prefixes, leaving no victim for a donated-state allocation. Eviction is attempted
once and allocation raises `Can not alloc mamba cache`. At `3N`, allocation
succeeds; after decode makes matched prefixes evictable, `2N` also succeeds.

The candidate correctly made the existing safe decode-prefix unlock default-on,
but coupled that change to a one-slot-per-request reduction in configured pool
headroom. Decode unlocking occurs after the admission peak, so steady-state
savings cannot justify reducing admission sizing. The correction keeps the
candidate's unlock and rollback switch, while restoring ratios 3/4/5 for
no-buffer/lazy/overlap modes under both switch settings.

This also preserves the existing bounded startup behavior for undersized pools:
`resolve_max_num_reqs` divides capacity by the admission ratio and raises when
the result is zero. The scheduler's existing request-length bounds were tested
separately. No speculative change was made to hierarchical-cache I/O.

Raw failing-before and passing-after logs are retained in this directory.
