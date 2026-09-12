# Independent review of PR 1728

- Upstream issue: https://github.com/sgl-project/sglang/issues/34676
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1764
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/1728
- Exact candidate commit: `9ee7239c734292e901b5dbd9792ce261c6ae27b9`
- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **request changes**

## Finding

The candidate is a partial fix. It preserves the parent candidate's typed
late-allocation exception, request-pool rollback, and scheduler catch/requeue
path. It also fixes the independently reported counterexample: a pre-existing
active `Scheduler.chunked_req` is no longer simultaneously inserted into the
ordinary `waiting_queue`. The candidate's focused suite passed (37 tests and
12 subtests), and an independent reproduction failed on parent commit
`b6ca9a3b9c47a1faaced6e588e46faec4693717f` before passing at the reviewed
commit.

However, an independent two-pass case shows that the original retry/backpressure
contract is not complete. If an ordinary prefill is the only work in the
scheduler and `prepare_for_extend()` raises `KVCacheOOMError`, the catch block
requeues the request and sets the empty `running_batch.batch_is_full` to true.
On the immediately following scheduling pass, the guard
`(running_batch.batch_is_full or len(waiting_queue) == 0) and chunked_req is None`
returns without attempting the requeued request. With no running decode batch
to make progress and clear the flag, later passes remain gated. The process no
longer crashes, but the request is neither retried nor failed request-locally.

## Evidence

- On the recorded base, an injected late `RuntimeError("Prefill out of memory")`
  escaped the real `_get_new_batch_prefill_raw()` method after admission, and
  the request was absent from `waiting_queue`. This reproduces the original
  scheduler-fatal control flow in a deterministic fixture.
- On parent commit `b6ca9a3`, the active chunk remained both `chunked_req` and a
  member of `waiting_queue`; the independent ownership assertion failed.
- At exact candidate `9ee7239`, the candidate focused suite passed: 37 tests and
  12 subtests.
- At exact candidate `9ee7239`, the independent idle-scheduler retry assertion
  failed. The first call caught the typed OOM and requeued the request; the
  second call returned `None` before `prepare_for_extend`, proving that the
  requeued request was blocked by the retained full flag.
- Imports resolved to `/job/repo/python/sglang/srt/managers/scheduler.py` and
  `/job/repo/python/sglang/srt/mem_cache/allocation.py`, so the checked-out
  candidate source—not an installed copy—was exercised.

Raw command output was preserved during revision switches under
`/job/review-evidence/j-ebc023b03be3/`.

## Environment and limitations

The available system has one AMD Instinct MI355X, Torch `2.11.0+rocm7.2`, and
ROCm/HIP `7.2.26015`. The reported environment used eight NVIDIA B300 GPUs,
CUDA 13, TP8/DCP4, Kimi-K3 weights, and sustained hybrid-Mamba traffic. Those
weights and that architecture were unavailable. The tiny Llama fixture cannot
exercise hybrid Mamba allocation and was not substituted as proof. No GPU
numerical or distributed/model-semantic claim is made.

The candidate changes Python and reports/tests only; it changes no C++, CUDA,
HIP, or FlyDSL native source. Therefore no native rebuild was required. The
prepared interpreter and explicit `PYTHONPATH=/job/repo/python` were used.

