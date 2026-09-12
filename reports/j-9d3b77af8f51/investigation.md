# Independent review of PR 1513

Upstream issue: https://github.com/sgl-project/sglang/issues/35012

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1549

Candidate: https://github.com/amdpilot-org/sglang/pull/1513 at
`92c8240901b80b66c06f044b8ff5cb5fedcaf368`.

## Recommendation

Accept. This is test-only hardening rather than a new runtime fix. Its parent is
the required prepared base, `358c163250ad3b1f62939b01ce1314a0a31a0365`,
which already contains the runtime correction from upstream commit
`829138a31e4f464563defc35b10e3468542e9aa9`.

The exact candidate changes only a focused CPU regression and its prior review
artifacts. It does not change Python runtime or native code. The test accurately
guards the original invariant: with the `timeout` policy, an expired timeout
terminates prefetch independently of `pool_transfers_done`.

## Independent evidence

The original faulty implementation was checked out at `c3947eea`, the parent of
the runtime fixing commit. An independent test used the actual linear timeout
calculation rather than mocking it. With the main KV transfer complete, an
expired timeout, and an auxiliary pool transfer still pending, the timeout
predicate returned true but `can_terminate_prefetch` returned false. The same
test also showed that a non-expired timeout remained false and that an expired
timeout with completed auxiliary transfers returned true.

The same three cases passed at the exact candidate commit. The candidate's own
four-case regression also passed there. Both runs imported
`/job/repo/python/sglang/srt/mem_cache/hiradix_cache.py`; the independent logs
record the source path. Inspection also found that the current
`UnifiedRadixCache._can_terminate_prefetch` timeout path does not consult pool
completion.

## Architecture and limitations

This decision is deterministic Python control flow and launches no GPU kernel,
so GPU execution is not claimed. The assigned device was visible as one AMD
Instinct MI350X (`gfx950:sramecc+:xnack-`) under Torch `2.11.0+rocm7.2` / HIP
`7.2.26015`. No native source changed in the candidate, so no native rebuild was
applicable. The import emitted an AITER path under the job-private cache, but the
tested method itself did not call AITER.

A live HiCache + Mooncake read-timeout deployment, full model, and distributed
PP/TP execution were not available and were not claimed. Those are integration
limitations, not remaining counterexamples to the reported timeout-decision
contract.
