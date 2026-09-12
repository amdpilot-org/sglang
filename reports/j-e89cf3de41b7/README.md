# Correction review: concurrent custom all-reduce

Candidate: https://github.com/amdpilot-org/sglang/pull/2371 at
`39673680441fb67c3934e0139f317742a2aa4353`

Independent review: https://github.com/amdpilot-org/sglang/pull/2469

The candidate's event-based eager-stream ordering is retained. The review's
host-thread counterexample was confirmed in the candidate source: guard state
was unlocked, and the guard returned before the native collective was enqueued.
Merely locking `maybe_serialize()` would still permit thread B to record its
event on thread A's stream before thread A enqueues the collective.

The correction adds one per-communicator host lock and holds it across both
stream/event bookkeeping and the legacy or V2 native launch. The regression in
`check_host_thread_serialization.py` fails against the exact candidate and
passes against this checkout. Focused unit tests also exercise two contending
host threads, both communicator launch hooks, the same-stream fast path, stream
switching, and the existing graph-capture boundary.

A real single-device probe used two host threads and two gfx950 streams. The
first thread deliberately paused while holding the communicator guard; the
second could not enqueue until the first enqueued its write. The GPU result was
`2`, equal to the CPU reference.

## Remaining limitations

- CUDA graph replay makes no Python communicator call. Two graphs containing
  collectives from the same communicator can therefore still replay
  concurrently, bypassing this host guard.
- The V2 push/pull and legacy AOT device polling loops remain unbounded. No CUDA
  device, CUDA compiler/ISA validation, A100 NVLink pair, or two-rank custom
  all-reduce execution was available. A speculative device deadline was not
  added.
- The assigned hardware was one AMD Instinct MI350X (`gfx950`), so the original
  CUDA deadlock, green-context workload, and multi-GPU protocol were not run.
  The GPU probe validates only host-thread/stream ordering.

Raw commands and outputs are retained in `raw/`.
