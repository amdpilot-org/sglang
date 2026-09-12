# Investigation: decode `/query_dp_ranks` retry storm

Source issue: https://github.com/sgl-project/sglang/issues/33088

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1952

## Evidence

The prepared base already batches unresolved bootstrap rooms, reuses a thread-local
`requests.Session` per bootstrap address, and prefetches the lookup asynchronously.
It did not rate-limit a completed lookup that returned no ranks or failed
immediately. Consequently, each scheduler cycle could submit another real
`CommonKVReceiver.query_prefill_dp_ranks` call for the same prefill address.

The regression drives the actual `prefetch_prefill_dp_rank_queries` and
`_resolve_pending_reqs` path with an immediately completed empty response. Before
the correction, the focused test observed a query on every attempted cycle. The
saved pre-fix output is in `raw/failing-before.txt`.

The correction records lookup attempts per bootstrap address and permits one every
10 ms. The first lookup is immediate, separate prefill addresses do not block each
other, and timestamps are removed when an address no longer has pending requests.
Both the asynchronous prefetch and synchronous fallback honor the same limit.

Upstream PR https://github.com/sgl-project/sglang/pull/33114 independently proposed
the same 10 ms per-address throttle for the earlier synchronous implementation and
remains open. The implementation here adapts that issue-specific correction to the
current batched asynchronous source rather than copying the older call path.

## Scope and limitations

The assigned single GPU is an AMD Instinct MI350X reporting gfx950, but no GPU
execution was needed for this CPU-side HTTP scheduling regression. The reported
Qwen3.5-397B-A17B-FP8 weights, eight-GPU nodes, MoRI distributed topology, and a
second node were not available. Therefore the original two-node TP8/EP8/DP8 serving
run and its exact `[Errno 99]` symptom were not reproduced. This change prevents
the evidenced scheduler-cycle query flood; it does not make an unreachable
bootstrap reachable or establish model-level semantic correctness.

