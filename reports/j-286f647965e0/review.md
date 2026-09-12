# Independent review of amdpilot-org/sglang PR 1576

- Upstream issue: https://github.com/sgl-project/sglang/issues/34676
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1610
- Candidate: https://github.com/amdpilot-org/sglang/pull/1576
- Exact candidate commit: `b6ca9a3b9c47a1faaced6e588e46faec4693717f`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **request changes**

## Result

The candidate is a partial fix, not a full resolution of the original issue.
It correctly introduces a typed KV allocation error, catches that error around
late prefill allocation, requeues ordinary admitted requests, and releases
request-pool slots acquired by the failed allocation. Its focused test suite
passes.

However, the recovery incorrectly requeues an already-active chunked request
while retaining the same object in `Scheduler.chunked_req`. On the following
scheduler pass, that request can be presented once by `add_chunked_req()` and
again by the normal `waiting_queue` loop. This violates the scheduler's
single-owner queue invariant and can lead to duplicate admission/state mutation
under exactly the late-allocation pressure being fixed.

## Evidence

On the prepared base, an injected allocator miss through the real
`Scheduler._get_new_batch_prefill_raw()` escaped as `RuntimeError`. A second
test drove the real paged `alloc_for_extend()` path: the allocator returned
`None`, the original prefill OOM was raised, and the newly acquired request-pool
slot remained assigned. These reproduce the fatal late-miss behavior and its
cleanup gap without relying on the candidate's prose.

At the exact candidate commit, its two focused test modules passed: 34 tests
and 12 subtests. An independent adversarial test then started with an existing
`scheduler.chunked_req`, injected `KVCacheOOMError` from
`prepare_for_extend()`, and asserted that the active chunk must not also enter
`waiting_queue`. It failed: the request remained `scheduler.chunked_req` and
was appended to `waiting_queue`.

Raw logs, the extracted regression, the adversarial test, import-path records,
and the complete candidate diff were preserved outside the checkout under
`/job/review-evidence/` while revisions were switched.

## Environment and limits

The prepared interpreter was
`/tmp/amdpilot-repo-j-286f647965e0/venv/bin/python`. Both revisions imported
`scheduler.py` and `allocation.py` from `/job/repo/python/sglang/...`, not from
an installed wheel. Torch was `2.11.0+rocm7.2`, ROCm was `7.2.26015`, and the
single assigned GPU was reported as AMD Instinct MI350X, `gfx950:sramecc+:xnack-`.

The candidate changes Python scheduler/allocation control flow only; no C++,
FlyDSL, or other native source changed, so no native rebuild was required or
performed. The tests were deterministic CPU control-flow tests and no GPU
execution result is claimed.

The original Kimi-K3 TP8/DCP4 workload was not run because this environment has
one AMD gfx950 GPU rather than eight NVIDIA B300 GPUs and no Kimi-K3 weights.
The deterministic tiny Llama fixture cannot exercise hybrid Mamba allocation,
so it would only be an unrelated serving smoke and was not used as proof.

## Commands

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-286f647965e0/venv/bin/python -m pytest -vv /job/review-evidence/test_prefill_alloc_oom_requeue.py::TestSchedulerPrefillOOM::test_late_allocation_miss_requeues_instead_of_escaping
# base: failed; RuntimeError escaped Scheduler._get_new_batch_prefill_raw

PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-286f647965e0/venv/bin/python -m pytest -vv /job/review-evidence/test_prefill_alloc_oom_requeue.py::TestAllocForExtendOOM
# base: 1 passed, 1 failed; newly acquired request slot remained allocated

PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-286f647965e0/venv/bin/python -m pytest -q test/registered/unit/managers/test_prefill_alloc_oom_requeue.py test/registered/unit/managers/test_prefill_adder.py
# candidate: 34 passed, 12 subtests passed

PYTHONPATH=/job/repo/python:/job/review-evidence /tmp/amdpilot-repo-j-286f647965e0/venv/bin/python -m pytest -vv /job/review-evidence/test_candidate_adversarial.py
# candidate: failed; existing chunked request duplicated into waiting_queue
```
