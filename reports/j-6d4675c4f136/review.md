# Independent review of candidate PR 2430

- Upstream issue: https://github.com/sgl-project/sglang/issues/30760
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2369
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2466
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Exact candidate: `26709f5a5b8744a56ed9ad7e8972a3271dfd8540`
- Recommendation: **accept**

## Finding

The candidate fixes the specific call-count mismatch described by the original
issue. On the base, `check_prefetch_progress(req.rid)` is reached only after
rank-local capacity checks and a possible `batch_is_full` break. The candidate
instead evaluates the complete waiting queue before those exits and makes the
admission loop consume local results. Given the issue's stated invariant that
the waiting queue and per-request prefetch operations are shared across TP
ranks, local allocator divergence can no longer change the sequence of
prefetch-progress calls.

The candidate changes only Python scheduler code and tests. The measured import
path was `/job/repo/python/sglang/srt/managers/scheduler.py`; no native source or
native library changed, so a native rebuild was not applicable.

## Independent checks

At the recorded base, the prefetch check is visibly inside the locally bounded
admission loop. With the candidate test retained but the scheduler restored to
the base, the suite failed 5/5. Four failures are only evidence that the new
helper is absent; the useful integration failure is
`test_snapshots_before_batch_full_early_return`, which shows that the base
returns on `batch_is_full` without executing the queue-wide check.

At the exact candidate commit:

```text
python -m pytest -q test/registered/unit/managers/test_scheduler_hicache_prefetch_progress.py
5 passed, 17 warnings

python -m pytest -q \
  test/registered/unit/managers/test_scheduler_decision_batch_params.py \
  test/registered/unit/mem_cache/test_hiradix_pp_sync_drain.py \
  test/registered/unit/managers/test_scheduler_hicache_prefetch_progress.py
10 passed, 17 warnings
```

Independent adversarial review covered divergent local admission capacities,
an already-full batch before admission, unfinished prefetch results, disabled
storage, and an empty queue. In each relevant candidate path, the complete
waiting-queue snapshot occurs before the rank-local early exits, while the base
either varies the number of checks or performs none.

## Qualification and limitations

This is a source-level and deterministic scheduler-contract verification, not a
full production reproduction. The prepared host exposes one AMD Instinct
MI350X/gfx950 through ROCm 7.2. The reported deployment needs four NVIDIA GPUs,
GLM-5.2-FP8 and EAGLE weights, Mooncake/RDMA, and production traffic. Those
prerequisites were unavailable, so no TP=4 NCCL server deadlock or full-model
recovery is claimed, and a single-GPU serving smoke was not substituted for it.

The fix relies on the original issue's premise that TP ranks have the same
waiting queue and corresponding ongoing-prefetch membership. A separate
divergence in either state could still produce mismatched internal collectives
because `HiRadixCache.check_prefetch_progress` returns without a collective
when a request is absent from `ongoing_prefetch`; no such counterexample was
demonstrated for the reported incident. The separately mentioned watchdog gap
is outside this issue's requested deadlock fix and is unchanged.
