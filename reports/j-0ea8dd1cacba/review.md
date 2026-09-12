# Independent review of PR 3511 at `ce81df839f9b10ad98b581f394d24cf04d346940`

Upstream issue: https://github.com/sgl-project/sglang/issues/38470

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3485

Review issue: https://github.com/amdpilot-org/sglang/issues/3512

## Verdict

Recommendation: **request changes**. The candidate fixes the primary unrecorded-event race and implements the default-off asynchronous handoff, but it does not fully satisfy the original failure-unblocking contract. If recovery from the first loader exception itself raises while recording release events or constructing its acknowledgement, that second exception escapes the daemon loop. The slot remains `enqueue_done == false`, no ack is published, and later slot rotation/loading can block forever.

## Evidence

The prepared checkout was exactly the requested recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`. Candidate testing used exact commit `ce81df839f9b10ad98b581f394d24cf04d346940`, then the checkout was restored to `amdpilot/j-0ea8dd1cacba` before this report was committed.

The imported candidate modules were the source checkout, not an installed wheel:

```text
/job/repo/python/sglang/__init__.py
/job/repo/python/sglang/srt/managers/cache_controller.py
```

No native/C++ files changed, `repository-environment.json` reports no prepared native component, and no native rebuild was applicable.

### Failing before / passing after on a real GPU

An independent script used `LayerLoadingEvent.wait()` before the producer thread recorded its event. The producer slept on the CPU, then filled a GPU tensor and recorded the layer event on a producer stream. A consumer stream waited through the controller interface and copied the tensor.

Recorded base result (exit 1):

```text
device=AMD Instinct MI350X hip=7.2.26015
matches=0/4096
```

Exact candidate result (exit 0):

```text
device=AMD Instinct MI350X hip=7.2.26015
matches=4096/4096
```

This directly confirms that the base can pass an unrecorded device event and that the candidate's CPU-side recorded flag prevents it on ROCm.

### Candidate regression suite

Command:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-0ea8dd1cacba/venv/bin/python -m pytest -q test/registered/unit/mem_cache/test_hicache_async_load_enqueue.py test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py
```

Result (exit 0): `19 passed, 1 skipped`. This covers the candidate's regression and the adjacent staged-transfer test. The candidate's committed evidence contains stale contradictory failing logs, but the exact commit passes in the prepared review environment.

### Independent failure and shutdown cases

A mock using the actual controller loop verified FIFO draining on normal shutdown and verified that an ordinary injected `_enqueue_load` failure records both layer releases, publishes an ack, and marks the slot enqueue-complete (2 tests passed).

An adversarial nested failure then made `producer_event.complete()` raise while the loader was handling the primary transfer exception. Exact candidate result (exit 1):

```text
loader_escaped=event record failure during abort
enqueue_done_calls=0
acks=0
candidate_abort_failure_exit=1
```

The implementation performs abort cleanup and `mark_enqueue_done()` inside the same unguarded `except` block rather than using a nested guard plus `finally`. Thus a device/event error during cleanup terminates the worker and leaves the producer slot permanently unavailable. This is a remaining counterexample to the issue's explicit failure acknowledgement/unblocking requirement.

## Scope and limitations

The candidate's producer handoff, per-layer recorded gate, normal slot backpressure, index-move stream context, fence-event capture, normal failure recovery, FIFO drain, and decode-restore enqueue check were inspected. Real GPU testing exercised event/copy ordering and the recorded gate on one AMD Instinct MI350X (`gfx950`, ROCm 7.2). No H100/CUDA device, TP=8 deployment, model weights, or `cudaMemcpyBatchAsync` direct-backend workload was available, so the upstream H100 throughput/latency numbers were not reproduced and are not claimed as local results. No serving-path smoke was used as proof.
