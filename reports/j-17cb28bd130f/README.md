# Investigation report: issue 35537

The checked-out scheduler still treated `ScheduleBatch.batch_is_full` as sticky
state for ordinary (non-hybrid, non-priority-preemption) generation. A focused
reproduction initialized the state produced by the final chunked-prefill pass:
three running requests, one waiting request, one currently allocatable slot, and
`batch_is_full=True`. Before the fix, `_get_new_batch_prefill_raw` returned
before consulting current capacity or the waiting queue.

The correction resets the previous pass's admission hint before the early return.
The existing slot and token-budget checks then derive current fullness. Regression
coverage also verifies that genuine zero capacity re-latches the flag and stops
admission, and that an empty waiting queue performs no capacity/admission work.

Upstream PR https://github.com/sgl-project/sglang/pull/35609 was inspected before
implementation. It proposes the same reset but remains open and is absent from
base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Evidence

- `raw/regression-before.txt`: all three new tests fail on the original source.
- `raw/regression-after.txt`: the complete scheduler test file passes (6 tests).
- `raw/gpu-serving/server.log`: a real gfx950 server log showing 32-token
  chunked-prefill passes and the queue changing from one waiting request to zero
  while prior requests still run.
- `raw/gpu-serving/probe.json`: four staggered 72-token prompts each complete 30
  generated tokens; load samples reach four running and zero waiting requests.
- `raw/gpu-info.txt` and `raw/gpu-after.txt`: assigned GPU identity and cleanup.

The GPU fixture is a deterministic two-layer random Llama created outside the
worktree from the qualified PR649 scripts at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. It validates this non-hybrid HTTP
and engine path only. The reported Qwen3.8-27B-NVFP4 hybrid GDN/mamba model,
NEXTN speculation, FP8 KV cache, WSL2/Docker environment, semantic accuracy,
and long production latency were not reproduced because those weights and that
environment were unavailable.
