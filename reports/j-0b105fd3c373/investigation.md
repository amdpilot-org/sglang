# Investigation notes

- Upstream issue: https://github.com/sgl-project/sglang/issues/33696
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1751
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Related merged change inspected: upstream PR #24462. Its PR body explicitly lists concurrent pause/continue on one worker and the shared `_pause_continue_future` as an unresolved follow-up.
- Base implementation evidence: `TokenizerWorker.pause_generation()` and `continue_generation()` both overwrite one `_pause_continue_future`; `_apply_pause_continue_broadcast()` resolves whichever future is current without identifying the originating operation.
- Correction: each operation gets a UUID carried in the request and router broadcast; the worker stores futures in an operation-ID-keyed map and removes entries in `finally` on success, cancellation, or dispatch failure. Broadcast state is always applied even when no waiter remains.
- Raw test logs are under `reports/j-0b105fd3c373/raw/`.
