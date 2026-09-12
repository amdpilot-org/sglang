# Zombie-request investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/36876

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2682

Outcome: **candidate_verified**. The prepared base already contains the merged
fix from upstream PR #35255, so this investigation does not change production
source.

The report's observable failure is real, but the issue snapshot attributes it
to a scheduler-local batch-transition window that request ingestion cannot
observe. The scheduler event loops call `ingest_requests()` only at the top of
an iteration. They then synchronously compute and run the local `batch`, and
publish it as `last_batch` before the next ingestion point. Adding
`cur_batch_for_debug` to `abort_request()` would therefore add a debug alias,
not close the reachable disconnect race.

The reachable race was in `TokenizerManager`: cancellation deleted
`rid_to_state` before the delayed abort, causing its state guard to suppress the
`AbortReq`. Current source records `ReqState.dispatched`, and
`_release_req_states_on_failure()` aborts dispatched requests while retaining
their state for scheduler cleanup. It also includes the deferred
chunked-prefill transition retry in `process_pending_chunked_abort()`. Those
changes arrived through merged upstream PR #35255 before this base.

## Evidence

The focused registered tests cover cancellation after dispatch, undispatched
cleanup, single and batch boundaries, dispatch failure, and a chunked request
moving out of its dedicated slot. They pass: 28 tests plus 3 subtests. Raw
output is in `evidence/unit-tests.txt`.

`run_disconnect_probe.py` launches the actual source checkout with a
deterministic two-layer random Llama on the assigned ROCm GPU, opens a streaming
HTTP request, waits for an SSE `data:` event, closes the TCP client, and polls
scheduler load. The retained result shows `num_reqs` changing from 1 to 0 in
0.251 seconds and zero occurrences of the exact reported error. The server log
contains real GPU prefill execution. The fixture weights were generated under
`/tmp/amdpilot-repo-j-7fffb3db9406`, outside the worktree, with SHA256
`6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`.

Reproduce the focused checks with:

```bash
/tmp/amdpilot-repo-j-7fffb3db9406/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  test/registered/unit/managers/test_scheduler_chunked_abort_race.py

/tmp/amdpilot-repo-j-7fffb3db9406/venv/bin/python \
  reports/j-7fffb3db9406/run_disconnect_probe.py \
  --fixture /tmp/amdpilot-repo-j-7fffb3db9406/tiny-random-llama \
  --output reports/j-7fffb3db9406/evidence/gpu-disconnect
```

## Limitations

This fixture qualifies the HTTP disconnect, tokenizer lifecycle, scheduler
execution, and GPU reclamation path only. DeepSeek-V4-Flash weights, DSPARK,
hierarchical cache, the production request concurrency, and CUDA hardware were
not available, so this is not a reproduction of that full deployment or a
claim about its model architecture. The exact historical broken revision was
not substituted for the prepared base; failing-before behavior is represented
by the regression added with the merged fix and the upstream production
evidence, while this checkout supplies passing-after evidence.
