# Prometheus queue latency investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/6357

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2512

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the scheduler exports
`sglang:num_queue_reqs` and the completed-request
`sglang:queue_time_seconds` histogram, but does not define or update
`sglang:avg_request_queue_latency`. A regression importing the missing queue
calculation helper failed during collection before the implementation was
added (pytest exit 2).

The correction adds a gauge for the mean wait-so-far of requests in the
current scheduler waiting queue. Both prefill and decode stats paths use one
allocation-free helper. Requests whose timestamp is still the sentinel `0.0`
are excluded, an empty queue reports `0.0`, and a future timestamp is clamped
to zero rather than exporting a negative latency.

Related upstream work inspected before implementation:

- PR 6574 proposed only the missing collector log call and closed unmerged.
- PR 8627 added the old metric and merged.
- PR 11123 removed it during a metrics/time-stats cleanup.
- PR 22613 proposed the live-queue semantics but closed unmerged; its review
  requested a shared, allocation-free calculation helper. The current source
  has since moved reporting into `scheduler_components/metrics_reporter.py`,
  so this change is adapted to the actual prepared implementation.

## Evidence

- `test_queue_latency_metrics.py`: four deterministic numerical cases for a
  two-request mean, empty queue, uninitialized timestamp, and future timestamp.
- `evidence/metrics-while-queued.txt`: an actual `/metrics` response containing
  queue depth `1.0` and average live queue latency `0.003117550048045814`.
- `evidence/server.log`: source-checkout server startup and real gfx950
  prefill/decode execution.
- `evidence/run-metadata.json`: exact launch command, HTTP results, metric
  observations, and cleanup details.

The GPU probe used the qualified deterministic random two-layer Llama fixture
from amdpilot-org/sglang PR649 commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Weights were generated under
`/tmp/amdpilot-repo-j-c031fac99fc2`, outside the checkout. This validates the
single-GPU HTTP and scheduler metrics path only. It does not reproduce the
reported TP=8, DP-attention, NVIDIA, production-model deployment, model
semantics, or distributed behavior. The owned server needed SIGKILL after its
graceful-shutdown timeout; descendants were reaped and the GPU returned idle.
