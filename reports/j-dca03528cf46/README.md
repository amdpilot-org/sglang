# Correction review for PR 526

Upstream issue: https://github.com/sgl-project/sglang/issues/38319

Mirror issue: https://github.com/amdpilot-org/sglang/issues/692

Candidate: https://github.com/amdpilot-org/sglang/pull/526 at
`4d68007d1345b03fd894dcc23e56f3f2bc687f8f`

Independent review: https://github.com/amdpilot-org/sglang/pull/681

Outcome: **candidate rejected; no justified source correction**.

The candidate's two focused tests pass at its exact revision, but its new paged
test does not call either real `cache_unfinished_req` implementation. It queues a
request-table view directly, overwrites that view, and therefore manufactures a
dependency on `BaseTokenToKVPoolAllocator._copy_for_free_group`.

Independent reproduction against the candidate and the prepared base exercised
the real Python `RadixCache.cache_unfinished_req(..., chunked=True)` and native
`RadixCacheCpp.cache_unfinished_req(..., chunked=True)` paths. With the clone
deliberately replaced by identity, both paths freed the duplicate request page,
retained the canonical radix page, rebound the request row correctly, and
returned the canonical indices from a fresh radix lookup. This is expected:

- Python materializes the page-aligned values with
  `to(dtype=torch.int64, copy=True)` before insertion and converts the deferred
  free segment to page representatives.
- The C++ wrapper copies the request-table slice with
  `to(dtype=torch.int64, copy=True)` before insertion and deferred freeing.

The checked-out scheduler also serializes a chunked abort: `abort_request`
records `_pending_chunked_abort_req`; `process_pending_chunked_abort` runs at the
top of a later scheduling step, stops another chunk from launching, drains the
already-launched result, and then releases KV. The existing chunked-abort and
pause/retract unit suites passed. Those tests verify scheduler state transitions,
not QSA payloads.

No candidate test invokes an actual retract or abort lifecycle, and no available
test or artifact demonstrates corrupted QSA KV payload or repeated token 248319
through the reported serving stack. The assigned system is ROCm 7.2 on an AMD
Instinct MI355X (`gfx950`), not DGX Spark GB10/SM121, and the
`RadixArk/Qwen3.8-Flash-Next-NVFP4` weights and reported QSA KDA/EAGLE setup are
absent. Those limitations cannot justify a speculative source change.

Raw commands/results and the standalone real-path reproducer are retained in
`raw/`. No native source changed, so no native rebuild was applicable.
