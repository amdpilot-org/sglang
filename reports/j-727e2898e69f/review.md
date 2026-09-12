# Independent review of amdpilot-org/sglang PR 1331

Candidate reviewed: `3e666d05dfeb40785ab8195f8e1d13ceccc17f87`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/35884

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1367

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1331

## Recommendation

Accept. The candidate fully resolves the original issue's timeout-path contract in the implementation paths exercised by this environment.

The base handler cancels only its local asyncio task, removes the rid from local state, and returns 503. Replaying the candidate's timeout regression against the recorded base produced a 503 with zero calls to `abort_request`, as expected for the reported defect. The success control passed.

At the candidate commit, the handler calls `TokenizerManager.abort_request(rid)` before removing the rid. This ordering matters because the single-tokenizer implementation deliberately skips unknown rids. The resulting `AbortReq` is dispatched to the scheduler.

The candidate also fixes a second necessary boundary: the scheduler's busy-path previously classified every health-prefixed object as a disposable health generation request. That included `AbortReq`, so merely adding the HTTP-side abort would allow a busy scheduler to swallow the cancellation. The new type-restricted `is_health_check_probe` classifier applies the shortcut only to tokenized generation and embedding probes.

An independent test invoked the actual undecorated `Scheduler.process_input_requests` implementation with a health-prefixed `AbortReq` while `is_fully_idle(for_health_check=True)` returned false. The abort reached `_request_dispatcher`, no health IPC acknowledgment was queued, and abort/lookalike objects were rejected by the probe classifier.

## Source and build validation

The prepared interpreter resolved `sglang`, `http_server.py`, and `scheduler.py` from `/job/repo/python/sglang`, so tests exercised the checked-out source rather than an installed SGLang copy. The imported AIter extension came from `/tmp/amdpilot-repo-j-727e2898e69f/cache/aiter/module_aiter_core.so`.

The candidate changes only Python, tests, and reports. No C++, HIP, CUDA, FlyDSL, or other native source changed, so a native rebuild was not applicable.

## Test evidence

- Base replay of `test_health_check_abort.py`: 1 failed, 1 passed. The timeout case failed because `tokenizer_manager.aborted` had length 0 instead of 1. The success control passed.
- Base replay of both candidate test files: collection also failed for `test_health_check_probe.py` because `is_health_check_probe` does not exist on the base, independently confirming the classifier is candidate code.
- Exact candidate focused tests: 5 passed.
- Exact candidate focused tests plus `test_tokenizer_manager_rid_cleanup.py`: 31 passed, 3 subtests passed.
- Independent busy-scheduler dispatch/type-boundary script: passed.

## Limitations

The assigned device is one AMD Instinct MI350X (gfx950), using Torch 2.11.0+rocm7.2 and HIP 7.2.26015. The report's NVIDIA RTX PRO 6000 Blackwell, Qwen3.8-27B-FP8 weights, 262144-token context, NEXTN speculative configuration, and multi-minute production stall were unavailable. No full-model serving run or paged-prefill numerical reproduction was performed. GPU execution was not needed for the process-boundary cancellation contract, and no GPU kernel was run as review evidence.

The candidate prevents timeout-created scheduler orphans. It does not claim to repair the separate paged-prefill shape mismatch or NVIDIA driver reset, and this review does not qualify either behavior.
