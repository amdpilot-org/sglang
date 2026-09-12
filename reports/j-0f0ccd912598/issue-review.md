Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1738

Related upstream PR inspected: https://github.com/sgl-project/sglang/pull/33784

At review time, the upstream issue and PR were both open. PR 33784 proposed
allowing a separate `flush_cache` while retract-paused. It did not make retract
self-contained and had no regression test. The prepared base still freed only
the request-owned suffix in `cache_finished_req(..., is_insert=False)`, dropped
the prefix lock, and performed only pressure-sized eviction from `release_req`.

Source path changed:
`python/sglang/srt/managers/schedule_batch.py`

Regression path added:
`test/registered/unit/managers/test_retract_all_cache_release.py`

Native path: not applicable; no native source was modified.
