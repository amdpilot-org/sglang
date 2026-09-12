# HiCache prefetch timeout investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35012

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1454

## Finding

The prepared base already contains the behavioral fix. At the revision linked by
the issue, `can_terminate_prefetch` first recognized an expired timeout, then
overrode that result when `pool_transfers_done` was false. Commit
`829138a31e4f464563defc35b10e3468542e9aa9` removed that dependency while
reworking prefetch completion synchronization. On the prepared base, the
`timeout` policy returns the timeout predicate directly.

## Reproduction evidence

The focused pending-transfer case was executed against detached commit
`c3947eea` (`829138a3^`). It failed because the method returned false even though
`is_prefetch_timeout` returned true. The test printed the imported module path so
the historical source selection is auditable. See
`raw/test_before_fix.log` and `raw/test_before_fix.exit_code`.

The same invariant plus three independent boundaries pass against prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`: completed pool transfer, timeout not
yet reached, `best_effort`, and `wait_complete`. See `raw/test_current.log` and
`raw/test_current.exit_code`.

## Scope and limitations

This is a CPU-only decision method, so GPU execution would not strengthen the
claim and none was performed. A live HiCache plus Mooncake read-timeout scenario
was not available; therefore this report does not claim a full serving, model,
distributed, or Mooncake transport reproduction. It verifies the exact
issue-reported control-flow defect in the actual old and current implementations.
