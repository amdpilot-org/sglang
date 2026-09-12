# Investigation evidence

## Scope and current-source review

- Upstream issue: https://github.com/sgl-project/sglang/issues/30760
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2369
- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Related open change reviewed: https://github.com/sgl-project/sglang/pull/33029

Current main still called `tree_cache.check_prefetch_progress(req.rid)` inside
the waiting-queue admission loop, after rank-local capacity checks and a
`batch_is_full` break. Both `HiRadixCache` and the current default
`UnifiedRadixCache` can perform distributed synchronization from that method
for the timeout policy. The related PR remained open and its injected-extra-
collective reproducer had been challenged by a maintainer, so its fix was not
copied.

The correction here snapshots progress for the complete waiting queue before
the first local capacity/fullness return or loop break, and the admission loop
uses the resulting local map. It adds no collective and applies to either cache
implementation through their existing interface.

## Failing-before regression

With the scheduler source restored to the recorded base behavior while leaving
the new regression present:

```text
$ /tmp/amdpilot-repo-j-eec6a6a299f2/venv/bin/python -m pytest -q \
    test/registered/unit/managers/test_scheduler_hicache_prefetch_progress.py
FFFF
4 failed, 17 warnings in 9.27s
```

All four tests failed because the base scheduler had no queue-wide progress
snapshot. The source change was then reapplied.

## Passing-after and boundary coverage

```text
$ /tmp/amdpilot-repo-j-eec6a6a299f2/venv/bin/python -m pytest -q \
    test/registered/unit/managers/test_scheduler_hicache_prefetch_progress.py
.....
5 passed, 17 warnings in 9.26s
```

The cases cover two simulated TP ranks with different local admission limits,
an unfinished prefetch result, disabled storage, an empty queue, and the
`batch_is_full` early-return boundary.

Adjacent focused suite:

```text
$ /tmp/amdpilot-repo-j-eec6a6a299f2/venv/bin/python -m pytest -q \
    test/registered/unit/managers/test_scheduler_decision_batch_params.py \
    test/registered/unit/mem_cache/test_hiradix_pp_sync_drain.py \
    test/registered/unit/managers/test_scheduler_hicache_prefetch_progress.py
..........
10 passed, 17 warnings in 9.39s
```

`git diff --check` and Python byte compilation also passed. The prepared
environment does not contain the `ruff` Python module.

## Hardware and reproduction limitation

Read-only hardware discovery reported Torch `2.11.0+rocm7.2`, HIP
`7.2.26015`, and exactly one `AMD Instinct MI355X` (`gfx950`). The report
requires TP=4 and originated on four NVIDIA GPUs with GLM-5.2-FP8, EAGLE, and
Mooncake/RDMA under production pressure. Those weights, backend, network, and
four GPUs are unavailable here. No full server/model reproduction was claimed,
and no single-GPU smoke was substituted for it. The deterministic regression
measures the issue's collective-call-count invariant in the actual scheduler
implementation.
